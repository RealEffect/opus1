//! Minimal `Vec`-like growable buffer backed by **inline fixed-size storage**.
//!
//! `FixedVec<T, N>` stores its elements in a `[core::mem::MaybeUninit<T>; N]` with a
//! tracked length — no global allocator is required. It mirrors the subset of the
//! `Vec` API used throughout opus1 (`resize`, `push`, `truncate`, `len`,
//! indexing, and `Deref<Target = [T]>` over the *initialized* range), so it can
//! replace a `Vec` field for field without the heap.
//!
//! Elements are never dropped, so the element type must not have a
//! destructor; building a buffer of any other type fails to compile. The codec
//! stores numbers, slices and plain structs of them.
//!
//! This is what lets opus1 be `#![no_std]` with **no `alloc`** at all.

use core::mem::MaybeUninit;
use core::ops::{Deref, DerefMut};

#[derive(Debug)]
pub struct FixedVec<T, const N: usize> {
    len: usize,
    buf: [MaybeUninit<T>; N],
}

impl<T, const N: usize> FixedVec<T, N> {
    /// The compile-time capacity (number of elements that can be stored).
    pub const CAPACITY: usize = N;

    /// Evaluated by every constructor: the buffer never drops its elements.
    const NO_DESTRUCTOR: () = assert!(
        !core::mem::needs_drop::<T>(),
        "FixedVec elements must not have a destructor"
    );

    /// Whether `from_value` and `from_slice` fill the buffer through a
    /// separate call (see `init_from_value`). Below 1 kB the move of the
    /// result costs no more than a call, so the buffer is built inline.
    const FILL_IN_PLACE: bool = core::mem::size_of::<Self>() > 1024;

    /// Create an empty buffer. `const`-constructible so it can initialize `static`s
    /// and struct fields in `const` contexts.
    pub const fn new() -> Self {
        let () = Self::NO_DESTRUCTOR;
        Self {
            len: 0,
            buf: [const { MaybeUninit::uninit() }; N],
        }
    }

    /// Create a buffer of length `len` where every element is a clone of `value`.
    #[inline]
    pub fn from_value(value: T, len: usize) -> Self
    where
        T: Clone,
    {
        assert!(len <= N, "FixedVec capacity exceeded");
        if Self::FILL_IN_PLACE {
            let mut s = MaybeUninit::<Self>::uninit();
            // SAFETY: `s` is valid for writes and `len <= N` was checked
            // above. `init_from_value` writes the length and the first `len`
            // elements; the other elements are `MaybeUninit` and need no
            // value.
            unsafe {
                Self::init_from_value(s.as_mut_ptr(), value, len);
                s.assume_init()
            }
        } else {
            let mut s = Self::new();
            fill_with_clones(&mut s.buf[..len], value);
            s.len = len;
            s
        }
    }

    /// Writes a buffer of `len` clones of `value` to `this`.
    ///
    /// This is a separate call on purpose. A function that builds the buffer
    /// in a local and returns it copies all `N` elements to the caller (1.9 kB
    /// for 480 floats), whatever `len` is. A call that fills memory through a
    /// pointer, followed by one move of the result, is a pattern the compiler
    /// turns into "fill the caller's variable directly".
    ///
    /// Panics if `len > N`, before anything is written.
    ///
    /// # Safety
    /// `this` must be valid for writes of `Self`.
    #[inline(never)]
    unsafe fn init_from_value(this: *mut Self, value: T, len: usize)
    where
        T: Clone,
    {
        let () = Self::NO_DESTRUCTOR;
        // SAFETY: the caller guarantees that `this` is valid for writes. The
        // element array holds `MaybeUninit`, so a reference to it is valid
        // before any element is written.
        let buf = unsafe { &mut (*this).buf };
        fill_with_clones(&mut buf[..len], value);
        // SAFETY: as above, `this` is valid for writes.
        unsafe { (&raw mut (*this).len).write(len) };
    }

    /// Copy a slice into a fresh buffer.
    #[inline]
    pub fn from_slice(src: &[T]) -> Self
    where
        T: Clone,
    {
        assert!(src.len() <= N, "FixedVec capacity exceeded");
        if Self::FILL_IN_PLACE {
            let mut s = MaybeUninit::<Self>::uninit();
            // SAFETY: `s` is valid for writes and `src.len() <= N` was
            // checked above. `init_from_slice` writes the length and the
            // first `src.len()` elements.
            unsafe {
                Self::init_from_slice(s.as_mut_ptr(), src);
                s.assume_init()
            }
        } else {
            let mut s = Self::new();
            write_clones(&mut s.buf[..src.len()], src);
            s.len = src.len();
            s
        }
    }

    /// Writes a copy of `src` to `this`. A separate call for the same reason
    /// as `init_from_value`. Panics if `src.len() > N`, before anything is
    /// written.
    ///
    /// # Safety
    /// `this` must be valid for writes of `Self`.
    #[inline(never)]
    unsafe fn init_from_slice(this: *mut Self, src: &[T])
    where
        T: Clone,
    {
        // SAFETY: the caller's guarantees are those of `write_from_slice`.
        unsafe { Self::write_from_slice(this, src) }
    }

    /// Writes a copy of `src` to `this`, which may be uninitialized memory.
    ///
    /// For code that builds a larger struct in place and has a `FixedVec`
    /// field to fill (the range decoder does). Panics if `src.len() > N`,
    /// before anything is written.
    ///
    /// # Safety
    /// `this` must be valid for writes of `Self`.
    #[inline(always)]
    pub(crate) unsafe fn write_from_slice(this: *mut Self, src: &[T])
    where
        T: Clone,
    {
        let () = Self::NO_DESTRUCTOR;
        // SAFETY: the caller guarantees that `this` is valid for writes; see
        // `init_from_value` for the reference to the element array.
        let buf = unsafe { &mut (*this).buf };
        write_clones(&mut buf[..src.len()], src);
        // SAFETY: as above, `this` is valid for writes.
        unsafe { (&raw mut (*this).len).write(src.len()) };
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// View the initialized elements as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `buf[..len]` holds initialized `T`s; `MaybeUninit<T>` has the
        // same layout as `T`, so the cast is sound for the initialized prefix.
        unsafe { core::slice::from_raw_parts(self.buf.as_ptr() as *const T, self.len) }
    }

    /// View the initialized elements as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: as in `as_slice`; `&mut self` gives exclusive access.
        unsafe { core::slice::from_raw_parts_mut(self.buf.as_mut_ptr() as *mut T, self.len) }
    }

    /// Raw pointer to the start of the (full) backing storage. Useful for code that
    /// writes into the buffer with explicit bounds tracking (e.g. the range coder).
    #[inline]
    pub fn as_storage_ptr(&self) -> *const T {
        self.buf.as_ptr() as *const T
    }

    #[inline]
    pub fn as_storage_mut_ptr(&mut self) -> *mut T {
        self.buf.as_mut_ptr() as *mut T
    }

    #[inline]
    pub fn push(&mut self, value: T) {
        // `assert!` (not `debug_assert!`): in release builds a silent overflow
        // here would be out-of-bounds UB, e.g. `KissFftState::new(nfft)` with
        // nfft > KISS_MAX_N.
        assert!(self.len < N, "FixedVec push past capacity");
        self.buf[self.len].write(value);
        self.len += 1;
    }

    /// Resize the logical length. When growing, new slots are filled with clones of
    /// `value`.
    #[inline]
    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        assert!(new_len <= N, "FixedVec capacity exceeded");
        if new_len > self.len {
            fill_with_clones(&mut self.buf[self.len..new_len], value);
            self.len = new_len;
        } else {
            self.truncate(new_len);
        }
    }

    #[inline]
    pub fn truncate(&mut self, new_len: usize) {
        self.len = self.len.min(new_len);
    }

    #[inline]
    pub fn clear(&mut self) {
        self.truncate(0);
    }

    /// Set the logical length without initializing/deinitializing elements.
    ///
    /// Mirrors `Vec::set_len`. The caller is responsible for ensuring elements in
    /// `..new_len` are initialized (or treated as raw bytes). Used by the range
    /// coder which manages its own byte-level writes.
    ///
    /// # Safety
    /// After this call, `as_slice()` will expose `new_len` elements. The caller
    /// must guarantee that the `..new_len` range holds valid `T`s (or that they are
    /// only ever read as raw bytes / overwritten before being read as `T`).
    #[inline]
    pub unsafe fn set_len(&mut self, new_len: usize) {
        assert!(new_len <= N, "FixedVec capacity exceeded");
        self.len = new_len;
    }

    /// Append all elements from `src` (cloning).
    #[inline]
    pub fn extend_from_slice(&mut self, src: &[T])
    where
        T: Clone,
    {
        assert!(src.len() <= N - self.len, "FixedVec capacity exceeded");
        let new_len = self.len + src.len();
        write_clones(&mut self.buf[self.len..new_len], src);
        self.len = new_len;
    }
}

/// Writes a clone of `value` to every slot.
///
/// The loop has no index check per element, so for plain number types it
/// compiles to a fill.
#[inline(always)]
fn fill_with_clones<T: Clone>(slots: &mut [MaybeUninit<T>], value: T) {
    for slot in slots {
        slot.write(value.clone());
    }
}

/// Writes a clone of `src[i]` to `slots[i]`. Both must have the same length.
///
/// `zip` stops at the shorter side, so the loop has no index check per
/// element: for plain number types it compiles to one block copy.
#[inline(always)]
fn write_clones<T: Clone>(slots: &mut [MaybeUninit<T>], src: &[T]) {
    debug_assert_eq!(slots.len(), src.len());
    for (slot, v) in slots.iter_mut().zip(src) {
        slot.write(v.clone());
    }
}

impl<T, const N: usize> Default for FixedVec<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> Deref for FixedVec<T, N> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, const N: usize> DerefMut for FixedVec<T, N> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: Clone, const N: usize> Clone for FixedVec<T, N> {
    #[inline]
    fn clone(&self) -> Self {
        Self::from_slice(self.as_slice())
    }
}

// `FixedVec<T, N>` is `Send` and `Sync` exactly when `T` is: its fields are a
// `usize` and an array of `MaybeUninit<T>`, and the compiler derives both
// traits from them. `auto_traits_follow_the_element_type` checks it.

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    /// `Send` and `Sync` come from the element type.
    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn auto_traits_follow_the_element_type() {
        fn is_send<T: Send>() {}
        fn is_sync<T: Sync>() {}
        is_send::<FixedVec<u8, 16>>();
        is_sync::<FixedVec<u8, 16>>();
        is_send::<FixedVec<f32, 2048>>();
        is_sync::<FixedVec<f32, 2048>>();
        // `Cell<u8>` is `Send` but not `Sync`; a buffer of them must not be
        // `Sync` either. `Rc` is neither.
        is_send::<FixedVec<Cell<u8>, 4>>();
        trait NotSync {
            const IS_SYNC: bool = false;
        }
        impl<T: ?Sized> NotSync for T {}
        struct Probe<T: ?Sized>(core::marker::PhantomData<T>);
        #[allow(dead_code)]
        impl<T: ?Sized + Sync> Probe<T> {
            const IS_SYNC: bool = true;
        }
        assert!(Probe::<FixedVec<u8, 4>>::IS_SYNC);
        assert!(!Probe::<FixedVec<Cell<u8>, 4>>::IS_SYNC);
        assert!(!Probe::<FixedVec<Rc<u8>, 4>>::IS_SYNC);
    }

    /// The constructors and `clone` give the same elements as a `Vec`, for
    /// capacities on both sides of the size at which the buffer is filled
    /// through a separate call, and for every length from 0 to the capacity.
    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn constructors_match_vec_small_and_large() {
        fn check<const N: usize>() {
            let src: Vec<u32> = (0..N as u32).map(|i| i.wrapping_mul(2654435761)).collect();
            for len in 0..=N {
                let a = FixedVec::<u32, N>::from_slice(&src[..len]);
                assert_eq!(a.len(), len);
                assert_eq!(a.as_slice(), &src[..len]);

                let b = FixedVec::<u32, N>::from_value(0xDEAD_BEEF, len);
                assert_eq!(b.len(), len);
                assert!(b.iter().all(|&v| v == 0xDEAD_BEEF));

                let c = a.clone();
                assert_eq!(c.len(), len);
                assert_eq!(c.as_slice(), a.as_slice());
            }
        }
        // 4 * 8 + 8 = 40 bytes: built inline.
        check::<8>();
        // 4 * 254 + 8 = 1024 bytes: the largest buffer built inline.
        check::<254>();
        // 4 * 255 + 8 = 1028 bytes: the smallest buffer filled through a call.
        check::<255>();
        // 4 * 600 + 8 bytes: filled through a call.
        check::<600>();
        assert!(!FixedVec::<u32, 254>::FILL_IN_PLACE);
        assert!(FixedVec::<u32, 255>::FILL_IN_PLACE);
    }

    #[test]
    fn resize_extend_truncate_match_vec() {
        let mut f = FixedVec::<i16, 300>::new();
        let mut v: Vec<i16> = Vec::new();
        let steps: [(usize, i16); 9] = [
            (0, 1),
            (1, 2),
            (7, 3),
            (7, 4),
            (300, 5),
            (299, 6),
            (16, 7),
            (17, 8),
            (0, 9),
        ];
        for &(len, value) in &steps {
            f.resize(len, value);
            v.resize(len, value);
            assert_eq!(f.as_slice(), v.as_slice());
        }
        for chunk in [&[1i16, 2, 3][..], &[], &[4], &[5; 200]] {
            f.extend_from_slice(chunk);
            v.extend_from_slice(chunk);
            assert_eq!(f.as_slice(), v.as_slice());
        }
        f.truncate(250);
        v.truncate(250);
        assert_eq!(f.as_slice(), v.as_slice());
        f.truncate(10);
        v.truncate(10);
        assert_eq!(f.as_slice(), v.as_slice());
        f.clear();
        assert!(f.is_empty());
    }

    #[test]
    #[should_panic(expected = "FixedVec capacity exceeded")]
    fn from_slice_past_capacity_panics() {
        let _ = FixedVec::<u8, 2000>::from_slice(&[0u8; 2001]);
    }

    #[test]
    #[should_panic(expected = "FixedVec capacity exceeded")]
    fn from_value_past_capacity_panics() {
        let _ = FixedVec::<u8, 4>::from_value(0u8, 5);
    }

    #[test]
    #[should_panic(expected = "FixedVec capacity exceeded")]
    fn extend_past_capacity_panics() {
        let mut f = FixedVec::<u8, 600>::from_value(0u8, 599);
        f.extend_from_slice(&[1, 2]);
    }

    #[test]
    #[should_panic(expected = "FixedVec capacity exceeded")]
    fn resize_past_capacity_panics() {
        let mut f = FixedVec::<u8, 4>::new();
        f.resize(5, 0);
    }

    /// Linear congruential generator (Knuth's MMIX constants), high bits.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 11
        }
    }

    /// Element types of other sizes and alignments than `u32`: one byte,
    /// eight bytes, sixteen bytes with 16-byte alignment, and an array. Each
    /// is checked on both sides of the 1 kB limit and at lengths 0, 1,
    /// capacity - 1 and capacity, plus random lengths.
    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn constructors_match_vec_for_other_element_types() {
        fn check<T, const N: usize>(make: impl Fn(u64) -> T, rng: &mut Lcg)
        where
            T: Clone + PartialEq + core::fmt::Debug,
        {
            let src: Vec<T> = (0..N).map(|_| make(rng.next())).collect();
            let fill = make(rng.next());
            let mut lens = vec![0, 1, N / 2, N - 1, N];
            for _ in 0..12 {
                lens.push(rng.next() as usize % (N + 1));
            }
            for len in lens {
                let a = FixedVec::<T, N>::from_slice(&src[..len]);
                assert_eq!(a.as_slice(), &src[..len]);
                let b = FixedVec::<T, N>::from_value(fill.clone(), len);
                assert_eq!(b.len(), len);
                assert!(b.iter().all(|v| *v == fill));
                let c = b.clone();
                assert_eq!(c.as_slice(), b.as_slice());

                // Grow a copy to the capacity and shrink it again.
                let mut d = a.clone();
                d.resize(N, fill.clone());
                assert_eq!(d[..len], src[..len]);
                assert!(d[len..].iter().all(|v| *v == fill));
                d.truncate(len / 2);
                assert_eq!(d.as_slice(), &src[..len / 2]);
                d.extend_from_slice(&src[len / 2..len]);
                assert_eq!(d.as_slice(), &src[..len]);
            }
        }
        let mut rng = Lcg(0x0BAD_5EED_0BAD_5EED);
        // u8: 1000 + 8 bytes is built inline, 1017 + 8 through a call.
        check::<u8, 1000>(|r| r as u8, &mut rng);
        check::<u8, 1017>(|r| r as u8, &mut rng);
        assert!(!FixedVec::<u8, 1016>::FILL_IN_PLACE);
        assert!(FixedVec::<u8, 1017>::FILL_IN_PLACE);
        // f64: 100 elements inline, 143 through a call.
        check::<f64, 100>(|r| r as f64 * 0.37 - 1e6, &mut rng);
        check::<f64, 143>(|r| r as f64 * 0.37 - 1e6, &mut rng);
        // u128 is aligned to 16 bytes, so the length is followed by padding.
        check::<u128, 9>(|r| (r as u128) << 61 | r as u128, &mut rng);
        check::<u128, 71>(|r| (r as u128) << 61 | r as u128, &mut rng);
        assert!(FixedVec::<u128, 71>::FILL_IN_PLACE);
        // An array element: 6 bytes with 2-byte alignment.
        check::<[i16; 3], 11>(|r| [r as i16, (r >> 16) as i16, (r >> 32) as i16], &mut rng);
        check::<[i16; 3], 333>(|r| [r as i16, (r >> 16) as i16, (r >> 32) as i16], &mut rng);
    }
}
