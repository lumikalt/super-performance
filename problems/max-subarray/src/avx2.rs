//! Eight stretches at the same time, one AVX2 instruction advances all of them.

use crate::summary::Summary;

use std::arch::x86_64::*;

const LANES: usize = 8;

/// Steps per lane before the sums are widened, `MAX_ABS * STEPS` fits an `i32` with room to spare.
const STEPS: usize = 4096;

/// Eight stretches at the same time like `chains::<8>`, but one AVX2 instruction advances all of them.
/// A block is `LANES` stretches of `STEPS` values, the stretches of a block are neighbours.
/// Each step gathers one value from each stretch.
pub fn gather(input: &[i32]) -> i64 {
    summary(input).best
}

/// The summary of the whole input, so that results of several stretches can be joined.
pub fn summary(input: &[i32]) -> Summary {
    assert!(!input.is_empty(), "the subarray has to be non-empty");

    let block = LANES * STEPS;
    let mut total: Option<Summary> = None;
    let mut start = 0;

    while start + block <= input.len() {
        // SAFETY.
        // The block is inside `input` and this module is only built with AVX2.
        let lanes = unsafe { block_gather(&input[start..start + block]) };

        for lane in lanes {
            total = Some(total.map_or(lane, |t| t.then(lane)));
        }

        start += block;
    }

    // What does not fill a block is short, plain Kadane steps are fine for it.
    if start < input.len() {
        let mut rest = Summary::of(input[start]);

        for &x in &input[start + 1..] {
            rest.push(x);
        }

        total = Some(total.map_or(rest, |t| t.then(rest)));
    }

    total.unwrap()
}

#[target_feature(enable = "avx2")]
fn block_gather(block: &[i32]) -> [Summary; LANES] {
    let offsets = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let offsets = _mm256_mullo_epi32(offsets, _mm256_set1_epi32(STEPS as i32));

    // SAFETY.
    // `i * 1 + lane * STEPS` stays below `LANES * STEPS`, the length of `block`.
    let load = |i: usize| unsafe { _mm256_i32gather_epi32::<4>(block.as_ptr().add(i), offsets) };

    let first = load(0);
    let (mut sum, mut prefix, mut suffix, mut best) = (first, first, first, first);

    for i in 1..STEPS {
        let x = load(i);

        sum = _mm256_add_epi32(sum, x);
        prefix = _mm256_max_epi32(prefix, sum);
        suffix = _mm256_max_epi32(x, _mm256_add_epi32(suffix, x));
        best = _mm256_max_epi32(best, suffix);
    }

    let mut out = [Summary::of(0); LANES];
    let (mut s, mut p, mut u, mut b) = ([0i32; LANES], [0i32; LANES], [0i32; LANES], [0i32; LANES]);

    // SAFETY.
    // Each array holds exactly eight `i32`.
    unsafe {
        _mm256_storeu_si256(s.as_mut_ptr().cast(), sum);
        _mm256_storeu_si256(p.as_mut_ptr().cast(), prefix);
        _mm256_storeu_si256(u.as_mut_ptr().cast(), suffix);
        _mm256_storeu_si256(b.as_mut_ptr().cast(), best);
    }

    for lane in 0..LANES {
        out[lane] =
            Summary { sum: s[lane] as i64, prefix: p[lane] as i64, suffix: u[lane] as i64, best: b[lane] as i64 };
    }

    out
}
