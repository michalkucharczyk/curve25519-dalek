// -*- mode: rust; -*-
//
// This file is part of curve25519-dalek.
// See LICENSE for licensing information.

//! The `pvm64` backend uses the PolkaVM (PVM) 256-bit wide-arithmetic
//! instructions for field arithmetic, with field elements represented as
//! saturated 4×64-bit little-endian limbs.
//!
//! On non-PVM targets the wide-arithmetic intrinsics fall back to pure-Rust
//! `u128`-based implementations with identical semantics, so this backend
//! compiles (and its unit tests run) everywhere.
//!
//! Scalar arithmetic reuses the `u64` backend's `Scalar52` unchanged.

// Scalar arithmetic is identical to the `u64` backend; reuse it verbatim.
// (Same trick as `fiat_u64`.) Only compiled when this backend is actually
// selected, to avoid duplicating the `u64` backend's scalar/constants in
// ordinary test builds where `pvm64::field` is compiled for comparison tests.
#[cfg(curve25519_dalek_backend = "pvm")]
#[path = "../u64/scalar.rs"]
pub mod scalar;

pub mod field;

#[cfg(curve25519_dalek_backend = "pvm")]
pub mod constants;
