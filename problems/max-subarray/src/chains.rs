//! Kadane on several stretches at the same time.

use crate::kadane::kadane;
use crate::summary::Summary;

/// Kadane's algorithm on `N` stretches at the same time.
/// Each step of the loop advances every stretch by one value.
/// The steps of one stretch depend on each other but the stretches do not, so the processor can overlap them.
fn chains<const N: usize>(input: &[i32]) -> i64 {
    assert!(!input.is_empty(), "the subarray has to be non-empty");

    let len = input.len() / N;

    if len == 0 {
        return kadane(input);
    }

    let mut stretches: [Summary; N] = std::array::from_fn(|c| Summary::of(input[c * len]));

    for i in 1..len {
        for (c, stretch) in stretches.iter_mut().enumerate() {
            stretch.push(input[c * len + i]);
        }
    }

    // The values left over by the division belong to the last stretch.
    for &x in &input[N * len..] {
        stretches[N - 1].push(x);
    }

    stretches.into_iter().reduce(Summary::then).unwrap().best
}

/// The same chains as `chains`, reading the input without bounds checks.
/// The checked version spends about two instructions per value on them.
fn chains_unchecked<const N: usize>(input: &[i32]) -> i64 {
    assert!(!input.is_empty(), "the subarray has to be non-empty");

    let len = input.len() / N;

    if len == 0 {
        return kadane(input);
    }

    let base = input.as_ptr();

    // SAFETY.
    // Every index below is smaller than `N * len`, which is at most `input.len()`.
    let read = |index: usize| unsafe { *base.add(index) };

    let mut stretches: [Summary; N] = std::array::from_fn(|c| Summary::of(read(c * len)));

    for i in 1..len {
        for (c, stretch) in stretches.iter_mut().enumerate() {
            stretch.push(read(c * len + i));
        }
    }

    // The values left over by the division belong to the last stretch.
    for &x in &input[N * len..] {
        stretches[N - 1].push(x);
    }

    stretches.into_iter().reduce(Summary::then).unwrap().best
}

pub fn chains2_unchecked(input: &[i32]) -> i64 {
    chains_unchecked::<2>(input)
}

pub fn chains4_unchecked(input: &[i32]) -> i64 {
    chains_unchecked::<4>(input)
}

pub fn chains8_unchecked(input: &[i32]) -> i64 {
    chains_unchecked::<8>(input)
}

pub fn chains16_unchecked(input: &[i32]) -> i64 {
    chains_unchecked::<16>(input)
}

pub fn chains2(input: &[i32]) -> i64 {
    chains::<2>(input)
}

pub fn chains4(input: &[i32]) -> i64 {
    chains::<4>(input)
}

pub fn chains8(input: &[i32]) -> i64 {
    chains::<8>(input)
}

pub fn chains16(input: &[i32]) -> i64 {
    chains::<16>(input)
}
