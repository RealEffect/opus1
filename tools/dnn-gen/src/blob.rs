//! The blob format of libopus: what `write_weights` of
//! `dnn/write_lpcnet_weights.c` writes and `parse_weights` of
//! `dnn/parse_lpcnet_weights.c` reads.
//!
//! Each record is a 64-byte header followed by the data, padded with zeros
//! to `block_size` bytes:
//!
//! | Offset | Size | Field |
//! |---:|---:|---|
//! | 0 | 4 | `"DNNw"` |
//! | 4 | 4 | version, 0 |
//! | 8 | 4 | type: 0 float, 1 int, 3 int8 |
//! | 12 | 4 | size of the data in bytes |
//! | 16 | 4 | `block_size`: the size rounded up to a multiple of 64 |
//! | 20 | 44 | the name, padded with zeros |
//!
//! The integers are little-endian, as on every target of libopus 1.6.1.

/// `WEIGHT_BLOCK_SIZE` of `dnn/nnet.h`.
pub const BLOCK: usize = 64;
/// The size of the name field.
pub const NAME_SIZE: usize = 44;

/// Appends one record to `out`.
pub fn write_record(out: &mut Vec<u8>, name: &str, kind: i32, data: &[u8]) -> Result<(), String> {
    // write_weights warns when the name does not fit with room to spare,
    // and then cuts it. The generator refuses instead.
    if name.len() >= NAME_SIZE - 1 {
        return Err(format!("array name `{name}` is too long for a blob record"));
    }
    if name.as_bytes().contains(&0) {
        return Err(format!("array name `{name}` holds a zero byte"));
    }
    let size = i32::try_from(data.len()).map_err(|_| format!("array `{name}` is too large"))?;
    let block_size = data.len().div_ceil(BLOCK) * BLOCK;
    out.extend_from_slice(b"DNNw");
    out.extend_from_slice(&0i32.to_le_bytes());
    out.extend_from_slice(&kind.to_le_bytes());
    out.extend_from_slice(&size.to_le_bytes());
    out.extend_from_slice(&(block_size as i32).to_le_bytes());
    let mut name_field = [0u8; NAME_SIZE];
    name_field[..name.len()].copy_from_slice(name.as_bytes());
    out.extend_from_slice(&name_field);
    out.extend_from_slice(data);
    out.resize(out.len() + block_size - data.len(), 0);
    Ok(())
}

/// One record read back from a blob.
#[derive(Debug, PartialEq, Eq)]
pub struct Record<'a> {
    pub name: &'a [u8],
    pub kind: i32,
    pub data: &'a [u8],
}

fn le_i32(b: &[u8], at: usize) -> i32 {
    i32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// Reads a blob with the rules of `parse_weights`. Used to check what the
/// generator wrote.
pub fn read(blob: &[u8]) -> Result<Vec<Record<'_>>, String> {
    let mut out = Vec::new();
    let mut off = 0usize;
    while off < blob.len() {
        let rem = blob.len() - off;
        if rem < BLOCK {
            return Err(format!("record at {off}: only {rem} bytes left"));
        }
        let h = &blob[off..off + BLOCK];
        let kind = le_i32(h, 8);
        let size = le_i32(h, 12);
        let block_size = le_i32(h, 16);
        if &h[..4] != b"DNNw" || le_i32(h, 4) != 0 {
            return Err(format!("record at {off}: bad head or version"));
        }
        if size <= 0 || block_size < size || block_size as usize > rem - BLOCK || h[BLOCK - 1] != 0 {
            return Err(format!("record at {off}: bad sizes or name"));
        }
        let name_field = &h[20..BLOCK];
        let name_len = name_field.iter().position(|&b| b == 0).unwrap_or(NAME_SIZE);
        let data = &blob[off + BLOCK..off + BLOCK + size as usize];
        if blob[off + BLOCK + size as usize..off + BLOCK + block_size as usize].iter().any(|&b| b != 0) {
            return Err(format!("record at {off}: padding is not zero"));
        }
        out.push(Record { name: &name_field[..name_len], kind, data });
        off += BLOCK + block_size as usize;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut b = Vec::new();
        write_record(&mut b, "a_bias", 0, &[1, 2, 3, 4]).unwrap();
        write_record(&mut b, "a_weights_int8", 3, &[7u8; 65]).unwrap();
        assert_eq!(b.len(), 64 + 64 + 64 + 128);
        let r = read(&b).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r[0], Record { name: b"a_bias", kind: 0, data: &[1, 2, 3, 4] });
        assert_eq!(r[1].data.len(), 65);
        assert!(write_record(&mut b, &"x".repeat(43), 0, &[0]).is_err());
    }
}
