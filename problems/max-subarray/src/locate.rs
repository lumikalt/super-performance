//! Finding the best sum first and the run that has it afterwards.

use crate::Span;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use crate::avx2::gather as simd_gather;
use crate::kadane::{kadane_two_best, step};

/// Finds the best sum with a fast version first, then walks the input again to see where it was.
/// The second pass keeps no best sum at all, it stops at the first value where the run reaches it.
fn locate(input: &[i32], find_sum: fn(&[i32]) -> i64) -> Span {
    assert!(!input.is_empty(), "the subarray has to be non-empty");

    let target = find_sum(input);
    let mut start = 0;
    let mut ending_here = input[0] as i64;

    if ending_here == target {
        return Span { start, end: 1, sum: target };
    }

    for (i, &x) in input.iter().enumerate().skip(1) {
        let x = x as i64;

        if ending_here < 0 {
            ending_here = x;
            start = i;
        } else {
            ending_here += x;
        }

        if ending_here == target {
            return Span { start, end: i + 1, sum: target };
        }
    }

    unreachable!("the best sum is the sum of some run");
}

pub fn locate_two_best(input: &[i32]) -> Span {
    locate(input, kadane_two_best)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub fn locate_simd(input: &[i32]) -> Span {
    locate(input, simd_gather)
}

/// `locate` with the restart marked as rare.
fn locate_branch(input: &[i32], find_sum: fn(&[i32]) -> i64) -> Span {
    assert!(!input.is_empty(), "the subarray has to be non-empty");

    let target = find_sum(input);
    let mut start = 0;
    let mut ending_here = input[0] as i64;

    if ending_here == target {
        return Span { start, end: 1, sum: target };
    }

    for (i, &x) in input.iter().enumerate().skip(1) {
        step(&mut ending_here, &mut start, i, x);

        if ending_here == target {
            return Span { start, end: i + 1, sum: target };
        }
    }

    unreachable!("the best sum is the sum of some run");
}

pub fn locate_branch_two_best(input: &[i32]) -> Span {
    locate_branch(input, kadane_two_best)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub fn locate_branch_simd(input: &[i32]) -> Span {
    locate_branch(input, simd_gather)
}
