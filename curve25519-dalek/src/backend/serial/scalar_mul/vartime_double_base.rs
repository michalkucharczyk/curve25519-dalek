// -*- mode: rust; -*-
//
// This file is part of curve25519-dalek.
// Copyright (c) 2016-2021 isis lovecruft
// Copyright (c) 2016-2019 Henry de Valence
// See LICENSE for licensing information.
//
// Authors:
// - isis agora lovecruft <isis@patternsinthevoid.net>
// - Henry de Valence <hdevalence@hdevalence.ca>
#![allow(non_snake_case)]

use core::cmp::Ordering;

use crate::backend::serial::curve_models::{ProjectiveNielsPoint, ProjectivePoint};
use crate::constants;
use crate::edwards::EdwardsPoint;
use crate::scalar::Scalar;
use crate::traits::Identity;
use crate::window::NafLookupTable5;

/// Compute \\(aA + bB\\) in variable time, where \\(B\\) is the Ed25519 basepoint.
#[cfg(not(curve25519_dalek_backend = "pvm"))]
pub fn mul(a: &Scalar, A: &EdwardsPoint, b: &Scalar) -> EdwardsPoint {
    let a_naf = a.non_adjacent_form(5);

    #[cfg(feature = "precomputed-tables")]
    let b_naf = b.non_adjacent_form(8);
    #[cfg(not(feature = "precomputed-tables"))]
    let b_naf = b.non_adjacent_form(5);

    // Find starting index
    let mut i: usize = 255;
    for j in (0..256).rev() {
        i = j;
        if a_naf[i] != 0 || b_naf[i] != 0 {
            break;
        }
    }

    let table_A = NafLookupTable5::<ProjectiveNielsPoint>::from(A);
    #[cfg(feature = "precomputed-tables")]
    let table_B = &constants::AFFINE_ODD_MULTIPLES_OF_BASEPOINT;
    #[cfg(not(feature = "precomputed-tables"))]
    let table_B =
        &NafLookupTable5::<ProjectiveNielsPoint>::from(&constants::ED25519_BASEPOINT_POINT);

    let mut r = ProjectivePoint::identity();
    loop {
        let mut t = r.double();

        match a_naf[i].cmp(&0) {
            Ordering::Greater => t = &t.as_extended() + &table_A.select(a_naf[i] as usize),
            Ordering::Less => t = &t.as_extended() - &table_A.select(-a_naf[i] as usize),
            Ordering::Equal => {}
        }

        match b_naf[i].cmp(&0) {
            Ordering::Greater => t = &t.as_extended() + &table_B.select(b_naf[i] as usize),
            Ordering::Less => t = &t.as_extended() - &table_B.select(-b_naf[i] as usize),
            Ordering::Equal => {}
        }

        r = t.as_projective();

        if i == 0 {
            break;
        }
        i -= 1;
    }

    r.as_extended()
}

/// Compute \\(aA + bB\\) in variable time, where \\(B\\) is the Ed25519 basepoint.
///
/// Identical algorithm to the generic version above, but restructured to
/// avoid large point-struct copies (`memcpy` calls), which are
/// disproportionately expensive on the PVM target: table entries are
/// selected by reference, and the `CompletedPoint` → `ProjectivePoint`
/// conversion at the end of each loop iteration writes `r`'s coordinates in
/// place instead of moving a 96-byte struct through a temporary.
#[cfg(curve25519_dalek_backend = "pvm")]
pub fn mul(a: &Scalar, A: &EdwardsPoint, b: &Scalar) -> EdwardsPoint {
    let a_naf = a.non_adjacent_form(5);

    #[cfg(feature = "precomputed-tables")]
    let b_naf = b.non_adjacent_form(8);
    #[cfg(not(feature = "precomputed-tables"))]
    let b_naf = b.non_adjacent_form(5);

    // Find starting index
    let mut i: usize = 255;
    for j in (0..256).rev() {
        i = j;
        if a_naf[i] != 0 || b_naf[i] != 0 {
            break;
        }
    }

    let table_A = NafLookupTable5::<ProjectiveNielsPoint>::from(A);
    #[cfg(feature = "precomputed-tables")]
    let table_B = &constants::AFFINE_ODD_MULTIPLES_OF_BASEPOINT;
    #[cfg(not(feature = "precomputed-tables"))]
    let table_B =
        &NafLookupTable5::<ProjectiveNielsPoint>::from(&constants::ED25519_BASEPOINT_POINT);

    let mut r = ProjectivePoint::identity();
    loop {
        let mut t = r.double();

        // `t = &t.as_extended() ± table.select(x)`, but with the temporary
        // extended point made explicit and the sum written into `t` in
        // place, so that no 128-byte `CompletedPoint` moves through a stack
        // temporary.
        match a_naf[i].cmp(&0) {
            Ordering::Greater => {
                let e = t.as_extended();
                t.set_add(&e, table_A.select_ref(a_naf[i] as usize));
            }
            Ordering::Less => {
                let e = t.as_extended();
                t.set_sub(&e, table_A.select_ref(-a_naf[i] as usize));
            }
            Ordering::Equal => {}
        }

        match b_naf[i].cmp(&0) {
            Ordering::Greater => {
                let e = t.as_extended();
                let entry = table_B.select_ref(b_naf[i] as usize);
                #[cfg(feature = "precomputed-tables")]
                t.set_add_affine(&e, entry);
                #[cfg(not(feature = "precomputed-tables"))]
                t.set_add(&e, entry);
            }
            Ordering::Less => {
                let e = t.as_extended();
                let entry = table_B.select_ref(-b_naf[i] as usize);
                #[cfg(feature = "precomputed-tables")]
                t.set_sub_affine(&e, entry);
                #[cfg(not(feature = "precomputed-tables"))]
                t.set_sub(&e, entry);
            }
            Ordering::Equal => {}
        }

        // r = t.as_projective(), with the coordinates written in place.
        r.X = &t.X * &t.T;
        r.Y = &t.Y * &t.Z;
        r.Z = &t.Z * &t.T;

        if i == 0 {
            break;
        }
        i -= 1;
    }

    r.as_extended()
}
