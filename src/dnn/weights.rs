//! Weight blobs in the libopus format, and the layer constructors that
//! read them (libopus `dnn/parse_lpcnet_weights.c`).
//!
//! A blob is a sequence of records. Each record is a 64-byte header (the
//! `WeightHead` of `dnn/nnet.h:56-63`) followed by its data, padded with
//! zeros to `block_size` bytes:
//!
//! | Offset | Size | Field |
//! |---:|---:|---|
//! | 0 | 4 | `head`: `"DNNw"` |
//! | 4 | 4 | `version`: 0 |
//! | 8 | 4 | `type`: 0 float, 1 int, 3 int8 |
//! | 12 | 4 | `size`: data size in bytes |
//! | 16 | 4 | `block_size`: `size` rounded up to a multiple of 64 |
//! | 20 | 44 | `name`, zero-padded; byte 43 is 0 |
//!
//! The integers are little-endian. `WeightList::parse` accepts what
//! libopus's `parse_weights` accepts, with one difference: it refuses a
//! blob whose arrays do not all start 4-byte aligned in memory, so that
//! float and int arrays can be borrowed in place. A blob that starts
//! 4-byte aligned and whose `block_size` values are multiples of 4 (as the
//! libopus writer makes them) always passes this check. Like libopus, the
//! parser does not check `head`, `version` or `type`.
//!
//! `linear_init` and `conv2d_init` build layers as libopus does, and also
//! refuse, where libopus reads out of bounds, crashes or builds a layer
//! that cannot run:
//!
//! - a sparse index table that is missing, or that holds a negative block
//!   count or a negative position;
//! - int8 weights equal to -128 (`plan.md` section 3: the 1.6.1 weights
//!   have none, and with -128 the kernel variants disagree);
//! - a linear layer with neither int8 nor float weights, and a conv2d layer
//!   without weights;
//! - int8 weights whose shape the kernels cannot run (an output count that
//!   is not a multiple of 8, an input count that is not a multiple of 4 or
//!   above `MAX_INPUTS`), a `diag` array without `3 · nb_inputs` outputs,
//!   and shapes whose byte counts overflow.
//!
//! A failure builds nothing, so no layer is ever half-initialized (libopus
//! leaves the fields it found before the failure). The model `init`
//! functions of `crate::dnn::models` build a whole model or fail, and
//! `crate::dnn::models::ModelSet::load` replaces each model group only
//! when all of its models load.

use super::MAX_INPUTS;
use super::nnet::{Conv2dLayer, LinearLayer};

/// `WEIGHT_BLOB_VERSION`: the only record version.
pub const WEIGHT_BLOB_VERSION: i32 = 0;
/// `WEIGHT_BLOCK_SIZE`: the size of a record header, and the unit of
/// `block_size`.
pub const WEIGHT_BLOCK_SIZE: usize = 64;
/// The size of the `name` field of a record header.
pub const WEIGHT_NAME_SIZE: usize = 44;
/// `WEIGHT_TYPE_float`.
pub const WEIGHT_TYPE_FLOAT: i32 = 0;
/// `WEIGHT_TYPE_int`: 32-bit integers (sparse index tables).
pub const WEIGHT_TYPE_INT: i32 = 1;
/// `WEIGHT_TYPE_qweight`: not used by any 1.6.1 model.
pub const WEIGHT_TYPE_QWEIGHT: i32 = 2;
/// `WEIGHT_TYPE_int8`.
pub const WEIGHT_TYPE_INT8: i32 = 3;
/// `SPARSE_BLOCK_SIZE`: the bytes of one 8 × 4 int8 block.
pub const SPARSE_BLOCK_SIZE: usize = 32;

/// The largest number of blobs that one `WeightList` searches: one per
/// model of libopus 1.6.1 (six in opus1 today, two more for DRED).
pub const MAX_BLOBS: usize = 8;

/// The error of a blob that `parse_weights` of libopus refuses.
pub const ERR_INVALID_BLOB: &str = "Invalid DNN blob";
/// The error of a blob whose arrays do not start 4-byte aligned.
pub const ERR_NOT_ALIGNED: &str = "DNN blob is not aligned";
/// The error of more than `MAX_BLOBS` blobs in one list.
pub const ERR_TOO_MANY_BLOBS: &str = "Too many DNN blobs";
/// The error of a layer whose array is not in the list.
pub const ERR_MISSING_ARRAY: &str = "DNN weight array is missing";
/// The error of a layer whose array has another size than its shape needs.
pub const ERR_ARRAY_SIZE: &str = "DNN weight array has the wrong size";
/// The error of a malformed sparse index table.
pub const ERR_SPARSE_INDEX: &str = "Invalid DNN sparse index table";
/// The error of a layer without weights.
pub const ERR_NO_WEIGHTS: &str = "DNN layer has no weights";
/// The error of int8 weights that hold -128.
pub const ERR_INT8_MINUS_128: &str = "DNN int8 weight is -128";
/// The error of a shape that the kernels cannot run.
pub const ERR_LAYER_SHAPE: &str = "Invalid DNN layer shape";

/// One array of a blob (libopus `WeightArray`, `dnn/nnet.h:44-49`).
#[derive(Clone, Copy)]
pub struct WeightArray<'w> {
    /// The name, up to its first zero byte.
    pub name: &'w [u8],
    /// The record's `type` field (`WEIGHT_TYPE_*`). libopus never checks it.
    pub kind: i32,
    /// The data: `size` bytes, without the padding. It starts 4-byte
    /// aligned in memory.
    pub data: &'w [u8],
}

impl core::fmt::Debug for WeightArray<'_> {
    /// The name, the type and the size, without the data.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WeightArray")
            .field("name", &core::str::from_utf8(self.name).unwrap_or("(not UTF-8)"))
            .field("kind", &self.kind)
            .field("size", &self.data.len())
            .finish()
    }
}

fn read_i32(b: &[u8], at: usize) -> Option<i32> {
    let bytes = b.get(at..at + 4)?;
    Some(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

/// Reads the record that starts at `off`, with the checks of libopus
/// `parse_record` (`dnn/parse_lpcnet_weights.c:38-53`) and of
/// `parse_weights`, which also refuses `size == 0` (lines 62-75). Returns
/// the array and the offset of the next record.
fn read_record(blob: &[u8], off: usize) -> Result<(WeightArray<'_>, usize), &'static str> {
    let remaining = blob.len().checked_sub(off).ok_or(ERR_INVALID_BLOB)?;
    if remaining < WEIGHT_BLOCK_SIZE {
        return Err(ERR_INVALID_BLOB);
    }
    let head = &blob[off..off + WEIGHT_BLOCK_SIZE];
    let kind = read_i32(head, 8).ok_or(ERR_INVALID_BLOB)?;
    let size = read_i32(head, 12).ok_or(ERR_INVALID_BLOB)?;
    let block_size = read_i32(head, 16).ok_or(ERR_INVALID_BLOB)?;
    let name_field = &head[20..WEIGHT_BLOCK_SIZE];
    // The comparisons of libopus, in the same order, without its int
    // overflows: `block_size < size`, `block_size > len - 64`, the last
    // name byte, `size < 0`, then `size == 0` in `parse_weights`.
    if block_size < size
        || i64::from(block_size) > (remaining - WEIGHT_BLOCK_SIZE) as i64
        || name_field[WEIGHT_NAME_SIZE - 1] != 0
        || size <= 0
    {
        return Err(ERR_INVALID_BLOB);
    }
    // 0 < size <= block_size <= remaining - 64, so both casts are exact.
    let (size, block_size) = (size as usize, block_size as usize);
    let start = off + WEIGHT_BLOCK_SIZE;
    let data = &blob[start..start + size];
    if !(data.as_ptr() as usize).is_multiple_of(4) {
        return Err(ERR_NOT_ALIGNED);
    }
    let name_len = name_field.iter().position(|&b| b == 0).unwrap_or(WEIGHT_NAME_SIZE);
    let array = WeightArray { name: &name_field[..name_len], kind, data };
    Ok((array, start + block_size))
}

/// Checks one blob: every record is well formed and aligned, and the
/// records fill the blob exactly.
fn validate(blob: &[u8]) -> Result<(), &'static str> {
    let mut off = 0;
    while off < blob.len() {
        off = read_record(blob, off)?.1;
    }
    Ok(())
}

/// The records of one validated blob, in order.
#[derive(Clone)]
struct Records<'w> {
    blob: &'w [u8],
    off: usize,
}

impl<'w> Iterator for Records<'w> {
    type Item = WeightArray<'w>;

    fn next(&mut self) -> Option<WeightArray<'w>> {
        if self.off >= self.blob.len() {
            return None;
        }
        // The blob was validated, so this never fails; a failure would end
        // the iteration instead of panicking.
        let (array, next) = read_record(self.blob, self.off).ok()?;
        self.off = next;
        Some(array)
    }
}

/// A list of validated blobs, searched in order. Looking up a name returns
/// the first array with that name, as libopus's `find_array_entry` does.
///
/// One list can hold several blobs (at most `MAX_BLOBS`), so that the
/// built-in weights, one blob per model, need no copy. A list of several
/// blobs behaves like one list of their concatenation.
#[derive(Clone, Copy, Default)]
pub struct WeightList<'w> {
    blobs: [&'w [u8]; MAX_BLOBS],
    count: usize,
}

impl core::fmt::Debug for WeightList<'_> {
    /// The number of blobs, arrays and bytes, without the data.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WeightList")
            .field("blobs", &self.count)
            .field("arrays", &self.len())
            .field("bytes", &self.blobs().iter().map(|b| b.len()).sum::<usize>())
            .finish()
    }
}

impl<'w> WeightList<'w> {
    /// A list without arrays.
    pub const fn empty() -> Self {
        WeightList { blobs: [&[]; MAX_BLOBS], count: 0 }
    }

    /// Checks a blob as libopus `parse_weights` does
    /// (`dnn/parse_lpcnet_weights.c:38-79`), and that every array starts
    /// 4-byte aligned in memory. An empty blob gives an empty list, as in
    /// libopus. Never reads out of bounds and never panics.
    ///
    /// Errors: `ERR_INVALID_BLOB` for a record that libopus refuses (also
    /// when the records do not fill the blob exactly), `ERR_NOT_ALIGNED`
    /// for an array that does not start 4-byte aligned.
    pub fn parse(blob: &'w [u8]) -> Result<Self, &'static str> {
        Self::parse_all(&[blob])
    }

    /// Checks several blobs as `parse` does, and lists them in order.
    /// Fails when one of them fails, or with `ERR_TOO_MANY_BLOBS` for more
    /// than `MAX_BLOBS` blobs.
    pub fn parse_all(blobs: &[&'w [u8]]) -> Result<Self, &'static str> {
        let mut list = Self::empty();
        for &blob in blobs {
            list.push(blob)?;
        }
        Ok(list)
    }

    /// Checks a blob as `parse` does and appends it to the list. Its
    /// arrays come after those already in the list. On failure the list is
    /// unchanged.
    pub fn push(&mut self, blob: &'w [u8]) -> Result<(), &'static str> {
        if self.count == MAX_BLOBS {
            return Err(ERR_TOO_MANY_BLOBS);
        }
        validate(blob)?;
        self.blobs[self.count] = blob;
        self.count += 1;
        Ok(())
    }

    /// The validated blobs, in order.
    pub fn blobs(&self) -> &[&'w [u8]] {
        &self.blobs[..self.count]
    }

    /// The first array named `name` (libopus `find_array_entry`).
    pub fn find(&self, name: &str) -> Option<WeightArray<'w>> {
        self.iter().find(|a| a.name == name.as_bytes())
    }

    /// Every array, in blob order.
    pub fn iter(&self) -> impl Iterator<Item = WeightArray<'w>> + use<'w> {
        let (blobs, count) = (self.blobs, self.count);
        blobs.into_iter().take(count).flat_map(|blob| Records { blob, off: 0 })
    }

    /// The number of arrays.
    pub fn len(&self) -> usize {
        self.iter().count()
    }

    /// Whether the list has no array.
    pub fn is_empty(&self) -> bool {
        self.iter().next().is_none()
    }
}

/// `data` as a slice of `T`, when it starts aligned for `T` and its length
/// is a multiple of the size of `T`.
///
/// `T` is `f32`, `i32` or `i8`: plain numbers, for which every bit pattern
/// is a valid value.
fn cast<T: Pod>(data: &[u8]) -> Option<&[T]> {
    // SAFETY: `T` is `f32`, `i32` or `i8` (the private trait `Pod`), which
    // have no padding and no invalid bit patterns, so any bytes in the
    // middle part are valid values. `align_to` keeps the lifetime of `data`.
    let (head, mid, tail) = unsafe { data.align_to::<T>() };
    (head.is_empty() && tail.is_empty()).then_some(mid)
}

/// The number types that `cast` may produce. The trait is private, so no
/// other type can have it.
trait Pod: Copy {}
impl Pod for f32 {}
impl Pod for i32 {}
impl Pod for i8 {}

/// libopus `find_array_check`: the array `name` with exactly `size` bytes.
fn find_sized<'w, T: Pod>(list: &WeightList<'w>, name: &str, size: usize) -> Result<&'w [T], &'static str> {
    let a = list.find(name).ok_or(ERR_MISSING_ARRAY)?;
    if a.data.len() != size {
        return Err(ERR_ARRAY_SIZE);
    }
    // The parser checked the alignment, and `size` is a multiple of the
    // size of `T` (it comes from a count of `T` values).
    cast(a.data).ok_or(ERR_NOT_ALIGNED)
}

/// libopus `opt_array_check`: like `find_sized`, but a missing array is
/// not an error. An array with the wrong size still is.
fn find_optional<'w, T: Pod>(list: &WeightList<'w>, name: &str, size: usize) -> Result<Option<&'w [T]>, &'static str> {
    match find_sized(list, name, size) {
        Ok(a) => Ok(Some(a)),
        Err(ERR_MISSING_ARRAY) => Ok(None),
        Err(e) => Err(e),
    }
}

/// libopus `find_idx_check` (`dnn/parse_lpcnet_weights.c:99-122`): the
/// index table of a sparse layer and its total number of 8 × 4 blocks.
///
/// For each block of 8 rows the table holds a count `n`, then `n` column
/// positions. Each position is a multiple of 4 with `pos + 3 < nb_inputs`,
/// and the table covers exactly `nb_outputs / 8` row blocks. Unlike
/// libopus, a missing table, a negative count and a negative position are
/// refused (libopus then reads out of bounds). Trailing bytes that do not
/// make a whole `int` are ignored, as in libopus.
fn find_idx<'w>(list: &WeightList<'w>, name: &str, nb_inputs: usize, nb_outputs: usize) -> Result<(&'w [i32], usize), &'static str> {
    let a = list.find(name).ok_or(ERR_MISSING_ARRAY)?;
    let whole = a.data.len() / 4 * 4;
    let idx: &[i32] = cast(&a.data[..whole]).ok_or(ERR_NOT_ALIGNED)?;
    let mut rest = idx;
    let mut rows_left = nb_outputs;
    let mut total_blocks = 0usize;
    while let Some((&count, tail)) = rest.split_first() {
        let count = usize::try_from(count).map_err(|_| ERR_SPARSE_INDEX)?;
        if tail.len() < count {
            return Err(ERR_SPARSE_INDEX);
        }
        let (positions, tail) = tail.split_at(count);
        for &pos in positions {
            let pos = usize::try_from(pos).map_err(|_| ERR_SPARSE_INDEX)?;
            if !pos.is_multiple_of(4) || pos.checked_add(3).is_none_or(|end| end >= nb_inputs) {
                return Err(ERR_SPARSE_INDEX);
            }
        }
        rows_left = rows_left.checked_sub(8).ok_or(ERR_SPARSE_INDEX)?;
        total_blocks += count;
        rest = tail;
    }
    if rows_left != 0 {
        return Err(ERR_SPARSE_INDEX);
    }
    Ok((idx, total_blocks))
}

/// Builds a layer from the arrays of `list` (libopus `linear_init`,
/// `dnn/parse_lpcnet_weights.c:124-177`). Each name is optional, as the
/// `NULL` arguments of the C function:
///
/// - `bias`, `subias`, `diag`: when named, the array must exist with
///   `nb_outputs` floats.
/// - `weights_idx`: when named, a valid sparse index table (see the module
///   documentation); `weights` and `float_weights` then hold
///   `32 · blocks` values instead of `nb_inputs · nb_outputs`.
/// - `weights` (int8): when named, the array must exist with the right
///   size, and `scale` must exist with `nb_outputs` floats.
/// - `float_weights`: optional even when named, as in libopus, but an
///   array of the wrong size is an error.
///
/// Fails, unlike libopus, when the layer ends up with neither int8 nor
/// float weights, when an int8 weight is -128, when the int8 shape cannot
/// run (`nb_outputs % 8 != 0`, `nb_inputs % 4 != 0` or
/// `nb_inputs > MAX_INPUTS`), when `diag` is named but
/// `nb_outputs != 3 · nb_inputs`, and when a byte count overflows. A
/// failure builds no layer, so nothing is left half-initialized.
pub fn linear_init<'w>(
    list: &WeightList<'w>,
    bias: Option<&str>,
    subias: Option<&str>,
    weights: Option<&str>,
    float_weights: Option<&str>,
    weights_idx: Option<&str>,
    diag: Option<&str>,
    scale: Option<&str>,
    nb_inputs: usize,
    nb_outputs: usize,
) -> Result<LinearLayer<'w>, &'static str> {
    let out_bytes = nb_outputs.checked_mul(4).ok_or(ERR_LAYER_SHAPE)?;
    let mut layer = LinearLayer { nb_inputs, nb_outputs, ..LinearLayer::default() };
    if let Some(name) = bias {
        layer.bias = Some(find_sized(list, name, out_bytes)?);
    }
    if let Some(name) = subias {
        layer.subias = Some(find_sized(list, name, out_bytes)?);
    }
    // The number of weights: every value of the matrix, or 32 per block
    // listed in the index table.
    let nb_weights = if let Some(name) = weights_idx {
        let (idx, blocks) = find_idx(list, name, nb_inputs, nb_outputs)?;
        layer.weights_idx = Some(idx);
        blocks.checked_mul(SPARSE_BLOCK_SIZE)
    } else {
        nb_inputs.checked_mul(nb_outputs)
    }
    .ok_or(ERR_LAYER_SHAPE)?;
    if let Some(name) = weights {
        layer.weights = Some(find_sized(list, name, nb_weights)?);
    }
    if let Some(name) = float_weights {
        let bytes = nb_weights.checked_mul(4).ok_or(ERR_LAYER_SHAPE)?;
        layer.float_weights = find_optional(list, name, bytes)?;
    }
    if let Some(name) = diag {
        layer.diag = Some(find_sized(list, name, out_bytes)?);
        if nb_inputs.checked_mul(3) != Some(nb_outputs) {
            return Err(ERR_LAYER_SHAPE);
        }
    }
    if weights.is_some() {
        // libopus passes `scale` to `strcmp` even when it is NULL.
        layer.scale = Some(find_sized(list, scale.ok_or(ERR_MISSING_ARRAY)?, out_bytes)?);
    }
    if let Some(w) = layer.weights {
        if w.contains(&-128) {
            return Err(ERR_INT8_MINUS_128);
        }
        if !nb_outputs.is_multiple_of(8) || !nb_inputs.is_multiple_of(4) || nb_inputs > MAX_INPUTS {
            return Err(ERR_LAYER_SHAPE);
        }
    }
    if layer.weights.is_none() && layer.float_weights.is_none() {
        return Err(ERR_NO_WEIGHTS);
    }
    Ok(layer)
}

/// Builds a conv2d layer from the arrays of `list` (libopus `conv2d_init`,
/// `dnn/parse_lpcnet_weights.c:179-202`). `bias`, when named, must exist
/// with `out_channels` floats. The weights must exist with
/// `in_channels · out_channels · ktime · kheight` floats; unlike libopus,
/// a missing weight array is an error (libopus builds the layer and then
/// dereferences a null pointer). A failure builds no layer.
pub fn conv2d_init<'w>(
    list: &WeightList<'w>,
    bias: Option<&str>,
    float_weights: Option<&str>,
    in_channels: usize,
    out_channels: usize,
    ktime: usize,
    kheight: usize,
) -> Result<Conv2dLayer<'w>, &'static str> {
    let mut layer = Conv2dLayer { in_channels, out_channels, ktime, kheight, ..Conv2dLayer::default() };
    if let Some(name) = bias {
        let bytes = out_channels.checked_mul(4).ok_or(ERR_LAYER_SHAPE)?;
        layer.bias = Some(find_sized(list, name, bytes)?);
    }
    if let Some(name) = float_weights {
        let bytes = in_channels
            .checked_mul(out_channels)
            .and_then(|n| n.checked_mul(ktime))
            .and_then(|n| n.checked_mul(kheight))
            .and_then(|n| n.checked_mul(4))
            .ok_or(ERR_LAYER_SHAPE)?;
        layer.float_weights = find_optional(list, name, bytes)?;
    }
    if layer.float_weights.is_none() {
        return Err(ERR_NO_WEIGHTS);
    }
    Ok(layer)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 64-byte aligned buffer for test blobs, without `alloc`.
    #[repr(C, align(64))]
    struct Buf([u8; 1024]);

    /// Writes a record at `off` and returns the offset after it.
    fn record(b: &mut [u8], off: usize, name: &str, kind: i32, data: &[u8]) -> usize {
        let block = data.len().div_ceil(64) * 64;
        b[off..off + 4].copy_from_slice(b"DNNw");
        b[off + 4..off + 8].copy_from_slice(&0i32.to_le_bytes());
        b[off + 8..off + 12].copy_from_slice(&kind.to_le_bytes());
        b[off + 12..off + 16].copy_from_slice(&(data.len() as i32).to_le_bytes());
        b[off + 16..off + 20].copy_from_slice(&(block as i32).to_le_bytes());
        b[off + 20..off + 64].fill(0);
        b[off + 20..off + 20 + name.len()].copy_from_slice(name.as_bytes());
        b[off + 64..off + 64 + data.len()].copy_from_slice(data);
        b[off + 64 + data.len()..off + 64 + block].fill(0);
        off + 64 + block
    }

    fn floats(v: &[f32], out: &mut [u8]) -> usize {
        for (i, x) in v.iter().enumerate() {
            out[4 * i..4 * i + 4].copy_from_slice(&x.to_le_bytes());
        }
        4 * v.len()
    }

    #[test]
    fn parse_find_and_iterate() {
        let mut buf = Buf([0; 1024]);
        let mut tmp = [0u8; 64];
        let n = floats(&[1.0, 2.0], &mut tmp);
        let mut off = record(&mut buf.0, 0, "l_bias", WEIGHT_TYPE_FLOAT, &tmp[..n]);
        off = record(&mut buf.0, off, "l_bias", WEIGHT_TYPE_FLOAT, &tmp[..4]);
        let list = WeightList::parse(&buf.0[..off]).unwrap();
        assert_eq!(list.len(), 2);
        // The first record of a name wins.
        assert_eq!(list.find("l_bias").unwrap().data.len(), 8);
        assert!(list.find("l_bia").is_none());
        assert!(list.find("l_bias_").is_none());
        let names = list.iter().filter(|a| a.name == b"l_bias").count();
        assert_eq!(names, 2);
        assert!(WeightList::parse(&[]).unwrap().is_empty());
    }

    #[test]
    fn libopus_refusals() {
        let mut buf = Buf([0; 1024]);
        let off = record(&mut buf.0, 0, "a", WEIGHT_TYPE_INT8, &[1, 2, 3]);
        assert_eq!(off, 128);
        let good = buf.0;
        let check = |edit: &dyn Fn(&mut [u8]), len: usize| {
            let mut b = Buf(good);
            edit(&mut b.0);
            WeightList::parse(&b.0[..len]).map(|_| ())
        };
        assert_eq!(check(&|_| {}, 128), Ok(()));
        // Truncated: in the header, in the data, in the padding.
        for len in [1, 63, 64, 66, 127] {
            assert_eq!(check(&|_| {}, len), Err(ERR_INVALID_BLOB), "length {len}");
        }
        // Extra bytes after the last record.
        assert_eq!(check(&|_| {}, 129), Err(ERR_INVALID_BLOB));
        assert_eq!(check(&|_| {}, 192), Err(ERR_INVALID_BLOB));
        // size 0, negative size, block_size below size, block_size past the end,
        // negative block_size, name without its final zero.
        let set = |at: usize, v: i32| move |b: &mut [u8]| b[at..at + 4].copy_from_slice(&v.to_le_bytes());
        assert_eq!(check(&set(12, 0), 128), Err(ERR_INVALID_BLOB));
        assert_eq!(check(&set(12, -1), 128), Err(ERR_INVALID_BLOB));
        assert_eq!(check(&set(16, 2), 128), Err(ERR_INVALID_BLOB));
        assert_eq!(check(&set(16, 128), 128), Err(ERR_INVALID_BLOB));
        assert_eq!(check(&set(16, i32::MAX), 128), Err(ERR_INVALID_BLOB));
        assert_eq!(check(&set(16, -64), 128), Err(ERR_INVALID_BLOB));
        assert_eq!(check(&|b: &mut [u8]| b[63] = b'x', 128), Err(ERR_INVALID_BLOB));
        // Accepted as in libopus: another head, version or type; a short
        // block_size that is still a multiple of 4.
        assert_eq!(check(&|b: &mut [u8]| b[0] = b'x', 128), Ok(()));
        assert_eq!(check(&set(4, 7), 128), Ok(()));
        assert_eq!(check(&set(8, 99), 128), Ok(()));
        assert_eq!(check(&set(16, 4), 68), Ok(()));
    }

    #[test]
    fn alignment() {
        let mut buf = Buf([0; 1024]);
        // A block_size of 5 puts the next record's data at 64 + 5 + 64.
        let mut off = record(&mut buf.0, 0, "a", WEIGHT_TYPE_INT8, &[1, 2, 3, 4, 5]);
        buf.0[16..20].copy_from_slice(&5i32.to_le_bytes());
        off = off - 64 + 5;
        let end = record(&mut buf.0, off, "b", WEIGHT_TYPE_INT8, &[1]);
        assert_eq!(WeightList::parse(&buf.0[..end]).map(|_| ()), Err(ERR_NOT_ALIGNED));
        // The last record may have any block_size.
        assert!(WeightList::parse(&buf.0[..off]).is_ok());
        // A blob that does not start 4-byte aligned.
        let mut buf2 = Buf([0; 1024]);
        let end = record(&mut buf2.0, 1, "a", WEIGHT_TYPE_INT8, &[1]);
        assert_eq!(WeightList::parse(&buf2.0[1..end]).map(|_| ()), Err(ERR_NOT_ALIGNED));
    }

    #[test]
    fn several_blobs() {
        let mut a = Buf([0; 1024]);
        let mut b = Buf([0; 1024]);
        let ea = record(&mut a.0, 0, "x", WEIGHT_TYPE_INT8, &[1]);
        let eb = record(&mut b.0, 0, "x", WEIGHT_TYPE_INT8, &[2, 2]);
        let eb = record(&mut b.0, eb, "y", WEIGHT_TYPE_INT8, &[3]);
        let list = WeightList::parse_all(&[&a.0[..ea], &b.0[..eb]]).unwrap();
        assert_eq!(list.blobs().len(), 2);
        assert_eq!(list.len(), 3);
        assert_eq!(list.find("x").unwrap().data, &[1]);
        assert_eq!(list.find("y").unwrap().data, &[3]);
        let mut full = WeightList::empty();
        for _ in 0..MAX_BLOBS {
            full.push(&a.0[..ea]).unwrap();
        }
        assert_eq!(full.push(&a.0[..ea]), Err(ERR_TOO_MANY_BLOBS));
        assert_eq!(WeightList::parse_all(&[&a.0[..ea], &b.0[..eb - 1]]).map(|_| ()), Err(ERR_INVALID_BLOB));
    }

    #[test]
    fn layers() {
        let mut buf = Buf([0; 1024]);
        let mut tmp = [0u8; 256];
        // An int8 layer with 4 inputs and 8 outputs, and a float layer with
        // 2 inputs and 3 outputs.
        let mut off = 0;
        let n = floats(&[0.5; 8], &mut tmp);
        off = record(&mut buf.0, off, "i_bias", WEIGHT_TYPE_FLOAT, &tmp[..n]);
        off = record(&mut buf.0, off, "i_scale", WEIGHT_TYPE_FLOAT, &tmp[..n]);
        off = record(&mut buf.0, off, "i_weights_int8", WEIGHT_TYPE_INT8, &[1u8; 32]);
        let n = floats(&[0.25; 6], &mut tmp);
        off = record(&mut buf.0, off, "f_weights_float", WEIGHT_TYPE_FLOAT, &tmp[..n]);
        let n = floats(&[0.25; 3], &mut tmp);
        off = record(&mut buf.0, off, "f_bias", WEIGHT_TYPE_FLOAT, &tmp[..n]);
        let list = WeightList::parse(&buf.0[..off]).unwrap();
        let i8l = |list: &WeightList<'_>, nin, nout| {
            linear_init(list, Some("i_bias"), None, Some("i_weights_int8"), Some("i_weights_float"), None, None, Some("i_scale"), nin, nout)
                .map(|_| ())
        };
        let l = linear_init(&list, Some("i_bias"), None, Some("i_weights_int8"), Some("i_weights_float"), None, None, Some("i_scale"), 4, 8).unwrap();
        assert_eq!((l.nb_inputs, l.nb_outputs), (4, 8));
        assert_eq!(l.weights.unwrap(), &[1i8; 32]);
        assert_eq!(l.scale.unwrap(), &[0.5f32; 8]);
        assert!(l.float_weights.is_none() && l.subias.is_none() && l.weights_idx.is_none());
        // Wrong shapes give wrong sizes.
        assert_eq!(i8l(&list, 8, 8), Err(ERR_ARRAY_SIZE));
        assert_eq!(i8l(&list, 2, 16), Err(ERR_ARRAY_SIZE));
        assert_eq!(i8l(&list, 2, usize::MAX), Err(ERR_LAYER_SHAPE));
        let r = linear_init(&list, None, None, Some("i_weights_int8"), None, None, None, Some("i_scale"), usize::MAX, 2);
        assert_eq!(r.map(|_| ()), Err(ERR_LAYER_SHAPE));
        // int8 weights need a scale.
        let r = linear_init(&list, None, None, Some("i_weights_int8"), None, None, None, None, 4, 8);
        assert_eq!(r.map(|_| ()), Err(ERR_MISSING_ARRAY));
        // A float layer, and the same without its weights.
        let f = linear_init(&list, Some("f_bias"), None, None, Some("f_weights_float"), None, None, None, 2, 3).unwrap();
        assert_eq!(f.float_weights.unwrap().len(), 6);
        let r = linear_init(&list, Some("f_bias"), None, None, Some("none_weights_float"), None, None, None, 2, 3);
        assert_eq!(r.map(|_| ()), Err(ERR_NO_WEIGHTS));
        let r = linear_init(&list, Some("f_bias"), None, None, Some("f_weights_float"), None, None, None, 3, 3);
        assert_eq!(r.map(|_| ()), Err(ERR_ARRAY_SIZE));
        // A conv2d layer: 1 input channel, 2 output channels, 1 × 3.
        let c = conv2d_init(&list, Some("i_scale"), Some("f_weights_float"), 1, 2, 1, 3);
        assert_eq!(c.map(|_| ()), Err(ERR_ARRAY_SIZE));
        let c = conv2d_init(&list, None, Some("f_weights_float"), 1, 2, 1, 3).unwrap();
        assert_eq!(c.float_weights.unwrap().len(), 6);
        assert_eq!(conv2d_init(&list, None, None, 1, 2, 1, 3).map(|_| ()), Err(ERR_NO_WEIGHTS));
        assert_eq!(conv2d_init(&list, None, Some("x"), 1, 2, 1, 3).map(|_| ()), Err(ERR_NO_WEIGHTS));
        assert_eq!(conv2d_init(&list, None, Some("x"), usize::MAX, 2, 1, 3).map(|_| ()), Err(ERR_LAYER_SHAPE));
    }
}
