// -*- mode: rust; coding: utf-8; -*-
//
// This file is part of curve25519-dalek.
// See LICENSE for licensing information.

//! Field arithmetic modulo \\(p = 2\^{255} - 19\\), using the PolkaVM (PVM)
//! 256-bit wide-arithmetic instructions on saturated \\(4 \times 64\\)-bit
//! limbs.
//!
//! A field element is held as an arbitrary 256-bit residue of its value
//! modulo \\(p\\): the limbs are *not* kept canonical (they may encode any
//! value below \\(2\^{256}\\)), and are only canonicalized on serialization
//! in [`FieldElement4x64::as_bytes`]. Reductions after multiplication fold
//! the value modulo \\(2\^{256} - 38 = 2p\\), which preserves congruence
//! modulo \\(p\\).

use core::fmt::Debug;
use core::ops::Neg;
use core::ops::{Add, AddAssign};
use core::ops::{Mul, MulAssign};
use core::ops::{Sub, SubAssign};

use subtle::Choice;
use subtle::ConditionallySelectable;

#[cfg(feature = "zeroize")]
use zeroize::Zeroize;

/// Wrappers around the five PVM 256-bit wide-arithmetic instructions.
///
/// All operands are little-endian 4×u64 (or 8×u64 for 512-bit values) limb
/// buffers in memory. On the PVM guest target (`riscv64` with the `e` target
/// feature, as targeted by `polkavm-derive`) the operations are emitted as
/// `.insn`-encoded instructions in the custom-0 (`0xb`) opcode space; the
/// destination buffer may alias the sources (read-all-then-write semantics),
/// although these safe wrappers always use a fresh destination.
///
/// On every other target, pure-Rust `u128`-based fallbacks are used instead.
/// The fallbacks are copied from `polkavm-common`'s `operation.rs`
/// (`wide_mul256`, `wide_add256`, `wide_sub256`, `wide_mul256_by_u64`,
/// `wide_redc256`), which are the normative reference semantics for these
/// instructions (the PVM interpreter runs exactly that code). They MUST be
/// kept in sync with `polkavm-common`.
#[allow(dead_code)] // Which of `redc256`/`mul256_by_u64` is used depends on cfg(pvm_redc).
mod intrinsics {
    #[cfg(all(target_arch = "riscv64", target_feature = "e"))]
    mod imp {
        use core::arch::asm;
        use core::mem::MaybeUninit;

        /// Full 512-bit product: returns `lhs * rhs`.
        #[inline(always)]
        pub fn mul256(lhs: &[u64; 4], rhs: &[u64; 4]) -> [u64; 8] {
            let mut dst = MaybeUninit::<[u64; 8]>::uninit();
            unsafe {
                asm!(
                    ".insn r 0xb, 4, 0, {d}, {a}, {b}",
                    d = in(reg) dst.as_mut_ptr(),
                    a = in(reg) lhs.as_ptr(),
                    b = in(reg) rhs.as_ptr(),
                    options(nostack),
                );
                dst.assume_init()
            }
        }

        /// Folds a 512-bit value modulo `2^256 - k`; the result is congruent
        /// to `src` and always below `2^256`, but not fully canonicalized.
        #[inline(always)]
        pub fn redc256(src: &[u64; 8], k: u64) -> [u64; 4] {
            let mut dst = MaybeUninit::<[u64; 4]>::uninit();
            unsafe {
                asm!(
                    ".insn r 0xb, 4, 1, {d}, {s}, {k}",
                    d = in(reg) dst.as_mut_ptr(),
                    s = in(reg) src.as_ptr(),
                    k = in(reg) k,
                    options(nostack),
                );
                dst.assume_init()
            }
        }

        /// Returns `(lhs + rhs) mod 2^256` and the carry-out (0 or 1).
        #[inline(always)]
        pub fn add256(lhs: &[u64; 4], rhs: &[u64; 4]) -> ([u64; 4], u64) {
            let mut dst = MaybeUninit::<[u64; 4]>::uninit();
            let carry: u64;
            unsafe {
                asm!(
                    ".insn r4 0xb, 5, 0, {c}, {a}, {b}, {d}",
                    c = out(reg) carry,
                    a = in(reg) lhs.as_ptr(),
                    b = in(reg) rhs.as_ptr(),
                    d = in(reg) dst.as_mut_ptr(),
                    options(nostack),
                );
                (dst.assume_init(), carry)
            }
        }

        /// Returns `(lhs - rhs) mod 2^256` and the borrow-out (0 or 1).
        #[inline(always)]
        pub fn sub256(lhs: &[u64; 4], rhs: &[u64; 4]) -> ([u64; 4], u64) {
            let mut dst = MaybeUninit::<[u64; 4]>::uninit();
            let borrow: u64;
            unsafe {
                asm!(
                    ".insn r4 0xb, 5, 1, {c}, {a}, {b}, {d}",
                    c = out(reg) borrow,
                    a = in(reg) lhs.as_ptr(),
                    b = in(reg) rhs.as_ptr(),
                    d = in(reg) dst.as_mut_ptr(),
                    options(nostack),
                );
                (dst.assume_init(), borrow)
            }
        }

        /// Returns `low256(lhs * rhs)` and bits 256..319 of the full product.
        #[inline(always)]
        pub fn mul256_by_u64(lhs: &[u64; 4], rhs: u64) -> ([u64; 4], u64) {
            let mut dst = MaybeUninit::<[u64; 4]>::uninit();
            let hi: u64;
            unsafe {
                asm!(
                    ".insn r4 0xb, 5, 2, {hi}, {a}, {m}, {d}",
                    hi = out(reg) hi,
                    a = in(reg) lhs.as_ptr(),
                    m = in(reg) rhs,
                    d = in(reg) dst.as_mut_ptr(),
                    options(nostack),
                );
                (dst.assume_init(), hi)
            }
        }
    }

    // Pure-Rust fallbacks, copied verbatim (modulo names) from
    // `polkavm-common/src/operation.rs`. These are the normative semantics of
    // the PVM instructions and MUST be kept in sync with that file.
    #[cfg(not(all(target_arch = "riscv64", target_feature = "e")))]
    mod imp {
        /// Full 512-bit product: returns `lhs * rhs`.
        ///
        /// Sync with `wide_mul256` in `polkavm-common/src/operation.rs`.
        #[inline]
        pub fn mul256(lhs: &[u64; 4], rhs: &[u64; 4]) -> [u64; 8] {
            let mut result = [0u64; 8];
            for i in 0..4 {
                let mut carry: u128 = 0;
                for j in 0..4 {
                    let value =
                        u128::from(result[i + j]) + u128::from(lhs[i]) * u128::from(rhs[j]) + carry;
                    result[i + j] = value as u64;
                    carry = value >> 64;
                }
                result[i + 4] = carry as u64;
            }
            result
        }

        /// Returns `(lhs + rhs) mod 2^256` and the carry-out (0 or 1).
        ///
        /// Sync with `wide_add256` in `polkavm-common/src/operation.rs`.
        #[inline]
        pub fn add256(lhs: &[u64; 4], rhs: &[u64; 4]) -> ([u64; 4], u64) {
            let mut result = [0u64; 4];
            let mut carry: u128 = 0;
            for i in 0..4 {
                let value = u128::from(lhs[i]) + u128::from(rhs[i]) + carry;
                result[i] = value as u64;
                carry = value >> 64;
            }
            (result, carry as u64)
        }

        /// Returns `(lhs - rhs) mod 2^256` and the borrow-out (0 or 1).
        ///
        /// Sync with `wide_sub256` in `polkavm-common/src/operation.rs`.
        #[inline]
        pub fn sub256(lhs: &[u64; 4], rhs: &[u64; 4]) -> ([u64; 4], u64) {
            let mut result = [0u64; 4];
            let mut borrow: u64 = 0;
            for i in 0..4 {
                let (value, b1) = lhs[i].overflowing_sub(rhs[i]);
                let (value, b2) = value.overflowing_sub(borrow);
                result[i] = value;
                borrow = u64::from(b1) | u64::from(b2);
            }
            (result, borrow)
        }

        /// Returns `low256(lhs * rhs)` and bits 256..319 of the full product.
        ///
        /// Sync with `wide_mul256_by_u64` in `polkavm-common/src/operation.rs`.
        #[inline]
        pub fn mul256_by_u64(lhs: &[u64; 4], rhs: u64) -> ([u64; 4], u64) {
            let mut result = [0u64; 4];
            let mut carry: u128 = 0;
            for i in 0..4 {
                let value = u128::from(lhs[i]) * u128::from(rhs) + carry;
                result[i] = value as u64;
                carry = value >> 64;
            }
            (result, carry as u64)
        }

        /// Folds a 512-bit value modulo `2^256 - k`. The result is congruent
        /// to `src` (mod `2^256 - k`) and always < `2^256`, but is not
        /// guaranteed to be fully canonicalized (it may still be
        /// >= `2^256 - k`).
        ///
        /// Sync with `wide_redc256` in `polkavm-common/src/operation.rs`.
        #[inline]
        pub fn redc256(src: &[u64; 8], k: u64) -> [u64; 4] {
            // t = t_lo + k·t_hi (5 limbs; ≤ 2^320 + 2^256)
            let mut t = [0u64; 5];
            let mut carry: u128 = 0;
            for i in 0..4 {
                let value = u128::from(src[i]) + u128::from(k) * u128::from(src[4 + i]) + carry;
                t[i] = value as u64;
                carry = value >> 64;
            }
            t[4] = carry as u64;

            // h = t >> 256 (≤ k); u = (t mod 2^256) + k·h (≤ 2^256 - 1 + k²)
            let kh = u128::from(k) * u128::from(t[4]);
            let (u, c) = add256(
                &[t[0], t[1], t[2], t[3]],
                &[kh as u64, (kh >> 64) as u64, 0, 0],
            );

            // dst = (u mod 2^256) + k·c; c ∈ {0, 1} and k² + k < 2^128, so this never carries.
            let (dst, overflow) = add256(&u, &[if c != 0 { k } else { 0 }, 0, 0, 0]);
            debug_assert_eq!(overflow, 0);
            dst
        }
    }

    pub(super) use imp::*;
}

/// A `FieldElement4x64` represents an element of the field
/// \\( \mathbb Z / (2\^{255} - 19)\\).
///
/// In the `pvm64` backend, a `FieldElement` is represented as four `u64`
/// limbs in little-endian order (radix \\(2\^{64}\\)); the value is an
/// arbitrary (not necessarily canonical) 256-bit residue of the field
/// element.
///
/// # Note
///
/// The `curve25519_dalek::field` module provides a type alias
/// `curve25519_dalek::field::FieldElement` to the backend-specific field
/// element type.
///
/// The backend-specific type `FieldElement4x64` should not be used
/// outside of the `curve25519_dalek::field` module.
#[derive(Copy, Clone)]
pub struct FieldElement4x64(pub(crate) [u64; 4]);

/// The prime \\(p = 2\^{255} - 19\\) as 4×64-bit little-endian limbs.
const P: [u64; 4] = [
    0xffff_ffff_ffff_ffed,
    0xffff_ffff_ffff_ffff,
    0xffff_ffff_ffff_ffff,
    0x7fff_ffff_ffff_ffff,
];

/// Fold the excess above \\(2\^{256}\\) back into the low 256 bits, using
/// \\(2\^{256} \equiv 38 \pmod p\\). Requires `excess <= 38`.
///
/// The first fold can itself carry (when the limbs are close to
/// \\(2\^{256}\\)), but after such a carry the low limbs are tiny, so the
/// second fold can never carry.
#[inline(always)]
fn fold_carry(limbs: [u64; 4], excess: u64) -> [u64; 4] {
    debug_assert!(excess <= 38);
    let (limbs, carry) = intrinsics::add256(&limbs, &[38 * excess, 0, 0, 0]);
    let (limbs, carry) = intrinsics::add256(&limbs, &[38 * carry, 0, 0, 0]);
    debug_assert_eq!(carry, 0);
    limbs
}

/// Fold a `sub256` borrow back into the low 256 bits, by subtracting
/// \\(38 \cdot \mathrm{borrow}\\) (since \\(-2\^{256} \equiv -38 \pmod p\\)).
///
/// The first fold can itself borrow (when the limbs are tiny), but after such
/// a borrow the value is close to \\(2\^{256}\\), so the second fold can
/// never borrow.
#[inline(always)]
fn fold_borrow(limbs: [u64; 4], borrow: u64) -> [u64; 4] {
    debug_assert!(borrow <= 1);
    let (limbs, borrow) = intrinsics::sub256(&limbs, &[38 * borrow, 0, 0, 0]);
    let (limbs, borrow) = intrinsics::sub256(&limbs, &[38 * borrow, 0, 0, 0]);
    debug_assert_eq!(borrow, 0);
    limbs
}

/// Reduce a 512-bit product to a 256-bit residue modulo
/// \\(2\^{256} - 38 = 2p\\).
#[cfg(pvm_redc)]
#[inline(always)]
fn reduce_wide(wide: &[u64; 8]) -> [u64; 4] {
    intrinsics::redc256(wide, 38)
}

/// Reduce a 512-bit product to a 256-bit residue modulo
/// \\(2\^{256} - 38 = 2p\\), without the `redc256` instruction.
#[cfg(not(pvm_redc))]
#[inline(always)]
fn reduce_wide(wide: &[u64; 8]) -> [u64; 4] {
    let lo = [wide[0], wide[1], wide[2], wide[3]];
    let hi = [wide[4], wide[5], wide[6], wide[7]];
    // lo + 2^256·hi ≡ lo + 38·hi (mod 2^256 - 38)
    let (h, h_hi) = intrinsics::mul256_by_u64(&hi, 38); // h_hi ≤ 37
    let (t, c) = intrinsics::add256(&lo, &h);
    // t + 2^256·(h_hi + c), with h_hi + c ≤ 38
    fold_carry(t, h_hi + c)
}

impl Debug for FieldElement4x64 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "FieldElement4x64({:?})", &self.0[..])
    }
}

#[cfg(feature = "zeroize")]
impl Zeroize for FieldElement4x64 {
    fn zeroize(&mut self) {
        self.0.zeroize();
    }
}

impl<'b> AddAssign<&'b FieldElement4x64> for FieldElement4x64 {
    fn add_assign(&mut self, rhs: &'b FieldElement4x64) {
        let (sum, carry) = intrinsics::add256(&self.0, &rhs.0);
        self.0 = fold_carry(sum, carry);
    }
}

impl<'a, 'b> Add<&'b FieldElement4x64> for &'a FieldElement4x64 {
    type Output = FieldElement4x64;
    fn add(self, rhs: &'b FieldElement4x64) -> FieldElement4x64 {
        let mut output = *self;
        output += rhs;
        output
    }
}

impl<'b> SubAssign<&'b FieldElement4x64> for FieldElement4x64 {
    fn sub_assign(&mut self, rhs: &'b FieldElement4x64) {
        let (diff, borrow) = intrinsics::sub256(&self.0, &rhs.0);
        self.0 = fold_borrow(diff, borrow);
    }
}

impl<'a, 'b> Sub<&'b FieldElement4x64> for &'a FieldElement4x64 {
    type Output = FieldElement4x64;
    fn sub(self, rhs: &'b FieldElement4x64) -> FieldElement4x64 {
        let mut output = *self;
        output -= rhs;
        output
    }
}

impl<'b> MulAssign<&'b FieldElement4x64> for FieldElement4x64 {
    fn mul_assign(&mut self, rhs: &'b FieldElement4x64) {
        self.0 = reduce_wide(&intrinsics::mul256(&self.0, &rhs.0));
    }
}

impl<'a, 'b> Mul<&'b FieldElement4x64> for &'a FieldElement4x64 {
    type Output = FieldElement4x64;
    fn mul(self, rhs: &'b FieldElement4x64) -> FieldElement4x64 {
        FieldElement4x64(reduce_wide(&intrinsics::mul256(&self.0, &rhs.0)))
    }
}

impl<'a> Neg for &'a FieldElement4x64 {
    type Output = FieldElement4x64;
    fn neg(self) -> FieldElement4x64 {
        let mut output = *self;
        output.negate();
        output
    }
}

impl ConditionallySelectable for FieldElement4x64 {
    fn conditional_select(
        a: &FieldElement4x64,
        b: &FieldElement4x64,
        choice: Choice,
    ) -> FieldElement4x64 {
        FieldElement4x64([
            u64::conditional_select(&a.0[0], &b.0[0], choice),
            u64::conditional_select(&a.0[1], &b.0[1], choice),
            u64::conditional_select(&a.0[2], &b.0[2], choice),
            u64::conditional_select(&a.0[3], &b.0[3], choice),
        ])
    }

    fn conditional_swap(a: &mut FieldElement4x64, b: &mut FieldElement4x64, choice: Choice) {
        u64::conditional_swap(&mut a.0[0], &mut b.0[0], choice);
        u64::conditional_swap(&mut a.0[1], &mut b.0[1], choice);
        u64::conditional_swap(&mut a.0[2], &mut b.0[2], choice);
        u64::conditional_swap(&mut a.0[3], &mut b.0[3], choice);
    }

    fn conditional_assign(&mut self, rhs: &FieldElement4x64, choice: Choice) {
        self.0[0].conditional_assign(&rhs.0[0], choice);
        self.0[1].conditional_assign(&rhs.0[1], choice);
        self.0[2].conditional_assign(&rhs.0[2], choice);
        self.0[3].conditional_assign(&rhs.0[3], choice);
    }
}

impl FieldElement4x64 {
    pub(crate) const fn from_limbs(limbs: [u64; 4]) -> FieldElement4x64 {
        FieldElement4x64(limbs)
    }

    /// The scalar \\( 0 \\).
    pub const ZERO: FieldElement4x64 = FieldElement4x64::from_limbs([0, 0, 0, 0]);
    /// The scalar \\( 1 \\).
    pub const ONE: FieldElement4x64 = FieldElement4x64::from_limbs([1, 0, 0, 0]);
    /// The scalar \\( -1 \\).
    pub const MINUS_ONE: FieldElement4x64 = FieldElement4x64::from_limbs([
        0xffff_ffff_ffff_ffec,
        0xffff_ffff_ffff_ffff,
        0xffff_ffff_ffff_ffff,
        0x7fff_ffff_ffff_ffff,
    ]);

    /// Invert the sign of this field element.
    pub fn negate(&mut self) {
        *self = &FieldElement4x64::ZERO - self;
    }

    /// Load a `FieldElement4x64` from the low 255 bits of a 256-bit input.
    ///
    /// # Warning
    ///
    /// This function does not check that the input used the canonical
    /// representative.  It masks the high bit, but it will happily decode
    /// 2^255 - 18 to 1.  Applications that require a canonical encoding of
    /// every field element should decode, re-encode to the canonical
    /// encoding, and check that the input was canonical.
    pub fn from_bytes(bytes: &[u8; 32]) -> FieldElement4x64 {
        let mut limbs = [0u64; 4];
        for (i, chunk) in bytes.chunks_exact(8).enumerate() {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(chunk);
            limbs[i] = u64::from_le_bytes(buf);
        }
        limbs[3] &= 0x7fff_ffff_ffff_ffff;
        FieldElement4x64(limbs)
    }

    /// Serialize this `FieldElement4x64` to a 32-byte array.  The
    /// encoding is canonical.
    pub fn as_bytes(&self) -> [u8; 32] {
        // The value v is an arbitrary residue below 2^256. Canonicalize it by
        // twice applying the reduction step
        //   q = ⌊(v + 19) / 2^255⌋ ∈ {0, 1, 2};  v ← v - q·(2^255 - 19).
        // The first application brings any v < 2^256 below 2^255 (but for
        // v ∈ [2p, 2p + 19) it yields a value in [p, p + 19), which is why a
        // single pass is not enough); the second brings any v < 2^255 below
        // p. This is cold-path glue, so plain u128 arithmetic is used instead
        // of the wide-arithmetic intrinsics.
        fn reduce_once(v: [u64; 4]) -> [u64; 4] {
            // q = (v + 19) >> 255
            let mut t = [0u64; 4];
            let mut carry: u128 = 19;
            for i in 0..4 {
                let acc = u128::from(v[i]) + carry;
                t[i] = acc as u64;
                carry = acc >> 64;
            }
            let q = ((carry as u64) << 1) | (t[3] >> 63);
            debug_assert!(q <= 2);

            // r = v - q·(2^255 - 19) = (v + 19·q) - q·2^255. Since r < 2^255,
            // the bits ≥ 255 of (v + 19·q) are exactly q; clear them.
            let mut r = [0u64; 4];
            let mut carry: u128 = u128::from(19 * q);
            for i in 0..4 {
                let acc = u128::from(v[i]) + carry;
                r[i] = acc as u64;
                carry = acc >> 64;
            }
            debug_assert_eq!(((carry as u64) << 1) | (r[3] >> 63), q);
            r[3] &= 0x7fff_ffff_ffff_ffff;
            r
        }

        let r = reduce_once(reduce_once(self.0));

        // Verify that the result is fully reduced: r < p.
        debug_assert!(
            !(r[3] == P[3] && r[2] == P[2] && r[1] == P[1] && r[0] >= P[0]),
            "canonicalization failed: result not below p"
        );

        let mut bytes = [0u8; 32];
        for (chunk, limb) in bytes.chunks_exact_mut(8).zip(r.iter()) {
            chunk.copy_from_slice(&limb.to_le_bytes());
        }
        bytes
    }

    /// Given `k > 0`, return `self^(2^k)`.
    pub fn pow2k(&self, k: u32) -> FieldElement4x64 {
        debug_assert!(k > 0);
        let mut output = *self;
        for _ in 0..k {
            output = output.square();
        }
        output
    }

    /// Returns the square of this field element.
    pub fn square(&self) -> FieldElement4x64 {
        FieldElement4x64(reduce_wide(&intrinsics::mul256(&self.0, &self.0)))
    }

    /// Returns 2 times the square of this field element.
    pub fn square2(&self) -> FieldElement4x64 {
        let square = self.square();
        &square + &square
    }
}

#[cfg(test)]
mod intrinsics_test {
    use super::*;

    /// Check the raw intrinsics against the test vectors from
    /// `polkavm-common/src/operation.rs`.
    #[test]
    fn intrinsics_reference_vectors() {
        // mul256: (2^256 - 1)² = 2^512 - 2^257 + 1
        let max = [u64::MAX; 4];
        let product = intrinsics::mul256(&max, &max);
        assert_eq!(
            product,
            [1, 0, 0, 0, u64::MAX - 1, u64::MAX, u64::MAX, u64::MAX]
        );

        // add256 carry-out, sub256 borrow-out
        let one = [1, 0, 0, 0];
        assert_eq!(intrinsics::add256(&max, &one), ([0, 0, 0, 0], 1));
        assert_eq!(intrinsics::sub256(&[0, 0, 0, 0], &one), (max, 1));

        // mul256_by_u64: (2^256 - 1)(2^64 - 1) = 2^320 - 2^256 - 2^64 + 1
        let (lo, hi) = intrinsics::mul256_by_u64(&max, u64::MAX);
        assert_eq!(lo, [1, u64::MAX, u64::MAX, u64::MAX]);
        assert_eq!(hi, u64::MAX - 1);

        // redc256: k = 0 degenerates to src mod 2^256
        let src = [1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(intrinsics::redc256(&src, 0), [1, 2, 3, 4]);

        // redc256 with k = 38: src = 2^256 → 38 (mod 2^256 - 38)
        let two_pow_256 = [0, 0, 0, 0, 1, 0, 0, 0];
        assert_eq!(intrinsics::redc256(&two_pow_256, 38), [38, 0, 0, 0]);

        // redc256 never returns >= 2^256 even for all-ones input, largest k.
        let all_ones = [u64::MAX; 8];
        let _ = intrinsics::redc256(&all_ones, u64::MAX);
    }
}

// Comparison tests against the default `u64` backend's `FieldElement51`.
// That backend is not compiled when the `pvm` backend is selected, so these
// only run in ordinary (non-pvm) 64-bit test builds; see
// `backend/serial/mod.rs`.
#[cfg(all(test, not(curve25519_dalek_backend = "pvm")))]
mod test {
    use super::*;
    use crate::backend::serial::u64::field::FieldElement51;

    /// A tiny deterministic xorshift64 PRNG for test inputs.
    struct Xorshift64(u64);

    impl Xorshift64 {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        fn next_bytes(&mut self) -> [u8; 32] {
            let mut bytes = [0u8; 32];
            for chunk in bytes.chunks_exact_mut(8) {
                chunk.copy_from_slice(&self.next_u64().to_le_bytes());
            }
            bytes
        }
    }

    /// LE encoding of p = 2^255 - 19.
    fn p_bytes() -> [u8; 32] {
        let mut bytes = [0xffu8; 32];
        bytes[0] = 0xed;
        bytes[31] = 0x7f;
        bytes
    }

    fn assert_same(a: &FieldElement4x64, b: &FieldElement51, what: &str) {
        assert_eq!(a.as_bytes(), b.as_bytes(), "mismatch in {}", what);
    }

    #[test]
    fn roundtrip_and_edge_bytes() {
        let mut edge_cases: [[u8; 32]; 6] = [[0u8; 32]; 6];
        edge_cases[1] = [0xffu8; 32]; // from_bytes masks bit 255
        edge_cases[2] = p_bytes();
        edge_cases[3] = p_bytes();
        edge_cases[3][0] = 0xec; // p - 1
        edge_cases[4] = p_bytes();
        edge_cases[4][0] = 0xee; // p + 1
        edge_cases[5] = p_bytes();
        edge_cases[5][0] = 0xed;
        edge_cases[5][31] = 0xff; // p with the high bit set (masked away)

        for bytes in edge_cases.iter() {
            let ours = FieldElement4x64::from_bytes(bytes);
            let theirs = FieldElement51::from_bytes(bytes);
            assert_same(&ours, &theirs, "roundtrip of edge bytes");
        }

        // Non-canonical encodings ≥ p: p..p+37 and the top of the range.
        for i in 0..38u8 {
            let mut bytes = p_bytes();
            // p ends in 0xed; p + i for i ≤ 18 stays in the low byte.
            bytes[0] = 0xed_u8.wrapping_add(i);
            if bytes[0] < 0xed {
                // wrapped: encode p + i via carry into the second byte — not
                // representable this way; skip (covered by 2^255 - 1 below).
                continue;
            }
            let ours = FieldElement4x64::from_bytes(&bytes);
            let theirs = FieldElement51::from_bytes(&bytes);
            assert_same(&ours, &theirs, "roundtrip of bytes >= p");
        }

        // 2^255 - 1 (largest value from_bytes can produce)
        let mut bytes = [0xffu8; 32];
        bytes[31] = 0x7f;
        let ours = FieldElement4x64::from_bytes(&bytes);
        let theirs = FieldElement51::from_bytes(&bytes);
        assert_same(&ours, &theirs, "roundtrip of 2^255 - 1");

        // Random roundtrips
        let mut rng = Xorshift64(0x0123_4567_89ab_cdef);
        for _ in 0..1000 {
            let bytes = rng.next_bytes();
            let ours = FieldElement4x64::from_bytes(&bytes);
            let theirs = FieldElement51::from_bytes(&bytes);
            assert_same(&ours, &theirs, "random roundtrip");
        }
    }

    #[test]
    fn ops_match_u64_backend() {
        let mut rng = Xorshift64(0xdead_beef_cafe_f00d);
        for i in 0..1000 {
            let (a_bytes, b_bytes) = if i % 8 == 7 {
                // Include non-canonical inputs ≥ p (from_bytes of bytes
                // encoding values in p..2^255-1).
                let mut a = [0xffu8; 32];
                a[0] = 0xed_u8.wrapping_add((rng.next_u64() % 19) as u8);
                a[31] = 0x7f;
                let mut b = rng.next_bytes();
                b[31] |= 0x7f;
                (a, b)
            } else {
                (rng.next_bytes(), rng.next_bytes())
            };

            let a = FieldElement4x64::from_bytes(&a_bytes);
            let b = FieldElement4x64::from_bytes(&b_bytes);
            let a51 = FieldElement51::from_bytes(&a_bytes);
            let b51 = FieldElement51::from_bytes(&b_bytes);

            assert_same(&(&a * &b), &(&a51 * &b51), "mul");
            assert_same(&a.square(), &a51.square(), "square");
            assert_same(&a.square2(), &a51.square2(), "square2");
            assert_same(&(&a + &b), &(&a51 + &b51), "add");
            assert_same(&(&a - &b), &(&a51 - &b51), "sub");
            assert_same(&(&b - &a), &(&b51 - &a51), "sub (reversed)");
            assert_same(&(-&a), &(-&a51), "negate");
            for k in 1..=8 {
                assert_same(&a.pow2k(k), &a51.pow2k(k), "pow2k");
            }
        }
    }

    #[test]
    fn carry_borrow_edge_cases() {
        // 2^256 - 1 ≡ 2·19 - 1 = 37 (mod p), since 2^256 ≡ 38.
        let max = FieldElement4x64::from_limbs([u64::MAX; 4]);
        assert_eq!(max.as_bytes()[0], 37);
        assert_eq!(max.as_bytes()[1..], [0u8; 31][..]);

        // 2^256 - 38 ≡ 0 (mod p)
        let two_p = FieldElement4x64::from_limbs([u64::MAX - 37, u64::MAX, u64::MAX, u64::MAX]);
        assert_eq!(two_p.as_bytes(), [0u8; 32]);

        // (2^256 - 1) + 1 carries out of add256 and folds to 38 ≡ 38 (mod p).
        let sum = &max + &FieldElement4x64::ONE;
        assert_eq!(sum.as_bytes()[0], 38);
        assert_eq!(sum.as_bytes()[1..], [0u8; 31][..]);

        // (2^256 - 2) + (2^256 - 1): carry with near-maximal low limbs,
        // exercising the double carry fold.
        let max_minus_1 =
            FieldElement4x64::from_limbs([u64::MAX - 1, u64::MAX, u64::MAX, u64::MAX]);
        let sum = &max_minus_1 + &max;
        let expected = (37u64 - 1) + 37; // both canonical values, mod p
        assert_eq!(sum.as_bytes()[0], expected as u8);
        assert_eq!(sum.as_bytes()[1..], [0u8; 31][..]);

        // Multiplication with maximal non-canonical inputs.
        let max51 = {
            // canonical value of 2^256 - 1 is 37
            FieldElement51::from_limbs([37, 0, 0, 0, 0])
        };
        assert_same(&(&max * &max), &(&max51 * &max51), "mul of 2^256 - 1");
        assert_same(&max.square(), &max51.square(), "square of 2^256 - 1");

        // 0 - (2^256 - 11): sub256 borrows and the fold borrows again
        // (diff = 11 < 38), exercising the double borrow fold.
        let b = FieldElement4x64::from_limbs([u64::MAX - 10, u64::MAX, u64::MAX, u64::MAX]);
        let ours = &FieldElement4x64::ZERO - &b;
        let b51 = FieldElement51::from_limbs([2 * 19 - 11, 0, 0, 0, 0]); // 2^256 - 11 ≡ 27
        let theirs = &FieldElement51::ZERO - &b51;
        assert_same(&ours, &theirs, "double borrow fold");

        // Values near 2^256 - 38 constructed via repeated additions of
        // MINUS_ONE, cross-checked against the u64 backend at every step.
        let mut ours = FieldElement4x64::from_bytes(&{
            let mut b = [0xffu8; 32];
            b[31] = 0x7f;
            b
        });
        let mut theirs = FieldElement51::from_bytes(&{
            let mut b = [0xffu8; 32];
            b[31] = 0x7f;
            b
        });
        for i in 0..300 {
            ours += &FieldElement4x64::MINUS_ONE;
            theirs += &FieldElement51::MINUS_ONE;
            assert_same(&ours, &theirs, "repeated addition of MINUS_ONE");
            if i % 3 == 0 {
                ours = ours.square();
                theirs = theirs.square();
                assert_same(&ours, &theirs, "square after repeated additions");
            }
        }
    }

    /// Spot-check the constant conversion formula used by
    /// `scripts/convert_constants_pvm64.py` on EDWARDS_D.
    #[test]
    fn edwards_d_constant_conversion() {
        // 4×64-limb form of -121665/121666 (mod p), from the converter.
        let d = FieldElement4x64::from_limbs([
            0x75eb4dca135978a3,
            0x00700a4d4141d8ab,
            0x8cc740797779e898,
            0x52036cee2b6ffe73,
        ]);
        // Radix-2^51 form from the u64 backend's constants.
        let d51 = FieldElement51::from_limbs([
            929955233495203,
            466365720129213,
            1662059464998953,
            2033849074728123,
            1442794654840575,
        ]);
        assert_same(&d, &d51, "EDWARDS_D");
    }
}
