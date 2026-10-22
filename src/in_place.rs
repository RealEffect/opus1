//! Building large structs directly in their final memory.
//!
//! Without the `heap` feature an `OpusDecoder` holds all of its state
//! inline: 60 KB, or 160 KB with the neural network features. An
//! `OpusEncoder` holds 150 KB, or 210 KB with `dred`. A constructor that
//! returns such a struct by value builds it on the stack and copies it out,
//! often more than once. `OpusDecoder::init` and `OpusEncoder::init`
//! instead write every field through a raw pointer into memory the caller
//! provides. The large fields are filled by functions that take a pointer
//! to the field, so no large value is ever built on the stack.
//!
//! [`write_fields!`] writes the fields. It also checks at compile time that
//! the list names every field of the struct, so that adding a field
//! without writing it in place fails to compile instead of leaving memory
//! uninitialized.
//!
//! [`write_zero`] fills a field with zero bytes, for numbers, arrays of
//! numbers and the structs that `zeroable_new!` checks. The shared tables
//! (the CELT mode, the built-in neural network models) are built in place
//! too, through `compat::OnceCell::get_or_write`.
//!
//! With the feature `heap` the large fields are boxes. `boxed` and
//! `write_box` write their contents in place too. `OpusEncoder::reset_state`
//! writes a new encoder over the old one with `HeapMemory::Reuse`, which
//! keeps the memory of every box, so a reset allocates nothing.
//!
//! These functions keep large values off the stack in an optimized build.
//! An unoptimized build may still copy some through the stack.

/// Writes every field of the struct that `$this` (a `*mut Self`) points
/// to. Each entry is either `field: value`, which writes `value`, or
/// `field = init`, which calls `init` (a closure or function) with a raw
/// pointer to the field; `init` must initialize the whole field.
///
/// The list must name every field of `Self`, with the same `#[cfg]`
/// attributes as the struct; otherwise the struct expression in the
/// unused closure below does not compile.
///
/// Must be used inside an `unsafe` block, in an `impl` of the struct, and
/// `$this` must be valid for writes of `Self`.
macro_rules! write_fields {
    ($this:ident, { $( $(#[$attr:meta])* $field:ident $op:tt $value:expr ),* $(,)? }) => {
        $(
            $(#[$attr])*
            $crate::in_place::write_fields!(@one $this, $field, $op, $value);
        )*
        // Never called: a struct expression must name every field, so this
        // fails to compile when the list above misses one.
        let _every_field_is_written = || -> Self {
            Self { $( $(#[$attr])* $field: $crate::in_place::unreachable_value(), )* }
        };
    };
    (@one $this:ident, $field:ident, :, $value:expr) => {
        (&raw mut (*$this).$field).write($value)
    };
    (@one $this:ident, $field:ident, =, $init:expr) => {
        ($init)(&raw mut (*$this).$field)
    };
}
pub(crate) use write_fields;

/// A value of any type, for the struct expression of [`write_fields!`]
/// that is never evaluated.
pub(crate) fn unreachable_value<T>() -> T {
    unreachable!("the field check of write_fields! is never called")
}

#[cfg(all(feature = "heap", not(feature = "std")))]
use alloc::boxed::Box;

/// A `Box<T>` whose contents `init` writes in place, so that `T` is never
/// built on the stack.
///
/// # Safety
/// `init` must initialize the whole `T` that its pointer points to.
#[cfg(feature = "heap")]
pub(crate) unsafe fn boxed<T>(init: impl FnOnce(*mut T)) -> Box<T> {
    let mut b = Box::<T>::new_uninit();
    init(b.as_mut_ptr());
    // SAFETY: the caller guarantees that `init` initialized the value.
    unsafe { b.assume_init() }
}

/// What the function that writes an encoder in place does with the heap
/// memory of the feature `heap`.
///
/// - `Allocate`: the encoder is new, so each box is allocated.
/// - `Reuse`: the memory holds an encoder already, which is being reset.
///   Each box it holds keeps its memory, and only the contents are written
///   anew, so a reset allocates nothing, as libopus `OPUS_RESET_STATE`.
///
/// Without `heap` an encoder holds no boxes, and both write the same.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum HeapMemory {
    Allocate,
    Reuse,
}

/// Writes a `Box<T>` to `p` whose contents `init` writes in place.
///
/// - With [`HeapMemory::Allocate`] the box is new, as from [`boxed`], and
///   `p` may be uninitialized.
/// - With [`HeapMemory::Reuse`] `p` holds a box already. Its contents are
///   dropped, and `init` writes the new contents into the same memory, so
///   nothing is allocated.
///
/// # Safety
/// `p` must be valid for writes, and with `Reuse` it must hold a valid
/// `Box<T>`. `init` must initialize the whole `T` that its pointer points
/// to. With `Reuse` it must not panic, because the box then holds a value
/// that was dropped.
#[cfg(feature = "heap")]
pub(crate) unsafe fn write_box<T>(p: *mut Box<T>, memory: HeapMemory, init: impl FnOnce(*mut T)) {
    match memory {
        // SAFETY: the caller guarantees that `p` is valid for writes and
        // what `init` does.
        HeapMemory::Allocate => unsafe { p.write(boxed(init)) },
        // SAFETY: `p` holds a valid box, so its contents are a valid `T`
        // that may be dropped once. `init` then writes a new `T` into the
        // same memory before anything reads it.
        HeapMemory::Reuse => unsafe {
            let contents: *mut T = &raw mut **p;
            core::ptr::drop_in_place(contents);
            init(contents);
        },
    }
}

/// Types for which zero bytes are a valid value: the value their
/// constructor makes, if they have one. [`write_zero`] writes it in place.
///
/// # Safety
/// Zero bytes must be a valid value of the type, and the type must have no
/// destructor.
pub(crate) unsafe trait Zeroable {}

macro_rules! zeroable_numbers {
    ($($t:ty),*) => {$(
        // SAFETY: zero bytes are the number 0 (or `false`).
        unsafe impl Zeroable for $t {}
    )*};
}
zeroable_numbers!(f32, i8, i16, i32, u8, u32, usize, bool);
// SAFETY: an array of zero values is zero bytes.
unsafe impl<T: Zeroable, const N: usize> Zeroable for [T; N] {}

/// Implements [`Zeroable`] for structs with a `const fn new()`, after a
/// check at compile time that `new()` returns only zero bytes. The check
/// fails to compile when a field gets a value other than zero, or a type
/// with padding (whose bytes are not all initialized).
#[cfg(feature = "deep-plc")]
macro_rules! zeroable_new {
    ($($t:ty),* $(,)?) => {$(
        const _: () = assert!(
            !core::mem::needs_drop::<$t>()
                // SAFETY: a value is read as its bytes; the read fails at
                // compile time if one of them is not initialized.
                && $crate::in_place::all_zero(&unsafe {
                    core::mem::transmute::<$t, [u8; core::mem::size_of::<$t>()]>(<$t>::new())
                }),
            concat!(stringify!($t), "::new() is not all zero bytes")
        );
        // SAFETY: the check above found that `new()` is a value of zero
        // bytes and that the type has no destructor.
        unsafe impl $crate::in_place::Zeroable for $t {}
    )*};
}
#[cfg(feature = "deep-plc")]
pub(crate) use zeroable_new;

/// Whether every byte is 0, for [`zeroable_new!`].
#[cfg(feature = "deep-plc")]
pub(crate) const fn all_zero(bytes: &[u8]) -> bool {
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != 0 {
            return false;
        }
        i += 1;
    }
    true
}

/// Writes zero bytes to `p`: for a [`Zeroable`] type, a valid value (the
/// one its constructor makes), written in place.
///
/// # Safety
/// `p` must be valid for writes of `T`.
pub(crate) unsafe fn write_zero<T: Zeroable>(p: *mut T) {
    // SAFETY: the caller guarantees that `p` is valid for writes, and
    // `Zeroable` that zero bytes are a valid `T`.
    unsafe { p.write_bytes(0, 1) }
}
