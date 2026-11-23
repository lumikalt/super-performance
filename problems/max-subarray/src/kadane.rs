//! Kadane's algorithm and the ways of writing it.

use crate::Span;

/// One pass that remembers the best sum of a run ending at the current value.
/// At each value the run either continues or starts over here, whichever is bigger.
pub fn kadane(input: &[i32]) -> i64 {
    assert!(!input.is_empty(), "the subarray has to be non-empty");

    let mut best = input[0] as i64;
    let mut ending_here = best;

    for &x in &input[1..] {
        ending_here = (x as i64).max(ending_here + x as i64);
        best = best.max(ending_here);
    }

    best
}

/// Kadane's algorithm with two accumulators for the best sum, used by alternate values.
/// The running sum is the same, so the answer is the same.
/// The chain of compares and conditional moves that updates the best sum is half as long.
pub fn kadane_two_best(input: &[i32]) -> i64 {
    assert!(!input.is_empty(), "the subarray has to be non-empty");

    let mut ending_here = input[0] as i64;
    let mut best_even = ending_here;
    let mut best_odd = i64::MIN;
    let (pairs, rest) = input[1..].as_chunks::<2>();

    for pair in pairs {
        ending_here = (pair[0] as i64).max(ending_here + pair[0] as i64);
        best_odd = best_odd.max(ending_here);
        ending_here = (pair[1] as i64).max(ending_here + pair[1] as i64);
        best_even = best_even.max(ending_here);
    }

    for &x in rest {
        ending_here = (x as i64).max(ending_here + x as i64);
        best_odd = best_odd.max(ending_here);
    }

    best_even.max(best_odd)
}

/// Kadane's algorithm that also remembers where the current run began.
/// When the best sum is beaten, the start and the end go with it.
pub fn kadane_span(input: &[i32]) -> Span {
    assert!(!input.is_empty(), "the subarray has to be non-empty");

    let mut best = Span { start: 0, end: 1, sum: input[0] as i64 };
    let mut start = 0;
    let mut ending_here = best.sum;

    for (i, &x) in input.iter().enumerate().skip(1) {
        let x = x as i64;

        // The run starts over when what came before is below zero.
        if ending_here < 0 {
            ending_here = x;
            start = i;
        } else {
            ending_here += x;
        }

        if ending_here > best.sum {
            best = Span { start, end: i + 1, sum: ending_here };
        }
    }

    best
}

/// Kadane's algorithm with two accumulators for the best run, used by alternate values.
/// The accumulators are compared at the end, and the one that ends first wins a tie.
pub fn kadane_span_two_best(input: &[i32]) -> Span {
    assert!(!input.is_empty(), "the subarray has to be non-empty");

    let first = Span { start: 0, end: 1, sum: input[0] as i64 };
    let mut best_even = first;
    let mut best_odd = Span { start: 0, end: 0, sum: i64::MIN };
    let mut start = 0;
    let mut ending_here = first.sum;
    let (pairs, rest) = input[1..].as_chunks::<2>();

    for (p, pair) in pairs.iter().enumerate() {
        // The index of the first value of the pair.
        let i = 1 + 2 * p;

        if ending_here < 0 {
            ending_here = pair[0] as i64;
            start = i;
        } else {
            ending_here += pair[0] as i64;
        }

        if ending_here > best_odd.sum {
            best_odd = Span { start, end: i + 1, sum: ending_here };
        }

        if ending_here < 0 {
            ending_here = pair[1] as i64;
            start = i + 1;
        } else {
            ending_here += pair[1] as i64;
        }

        if ending_here > best_even.sum {
            best_even = Span { start, end: i + 2, sum: ending_here };
        }
    }

    for (j, &x) in rest.iter().enumerate() {
        let i = 1 + 2 * pairs.len() + j;

        if ending_here < 0 {
            ending_here = x as i64;
            start = i;
        } else {
            ending_here += x as i64;
        }

        if ending_here > best_odd.sum {
            best_odd = Span { start, end: i + 1, sum: ending_here };
        }
    }

    if best_odd.sum > best_even.sum || (best_odd.sum == best_even.sum && best_odd.end < best_even.end) {
        best_odd
    } else {
        best_even
    }
}

/// Takes in the next value, starting a new run when the one so far is below zero.
/// Written as the compare that `kadane` uses, which the compiler turns into a branch.
/// The restart is marked as rare, so the branch costs nothing while it is guessed right.
/// The version in `kadane_span` is tested as `ending_here < 0` and becomes conditional moves instead.
#[inline(always)]
pub(crate) fn step(ending_here: &mut i64, start: &mut usize, i: usize, x: i32) {
    let x = x as i64;
    let continued = *ending_here + x;

    // The same test as the run being below zero, but written as the compare that kadane uses.
    if x > continued {
        std::hint::cold_path();
        *ending_here = x;
        *start = i;
    } else {
        *ending_here = continued;
    }
}

/// `kadane_span` with the restart marked as rare.
pub fn kadane_span_branch(input: &[i32]) -> Span {
    assert!(!input.is_empty(), "the subarray has to be non-empty");

    let mut best = Span { start: 0, end: 1, sum: input[0] as i64 };
    let mut start = 0;
    let mut ending_here = best.sum;

    for (i, &x) in input.iter().enumerate().skip(1) {
        step(&mut ending_here, &mut start, i, x);

        if ending_here > best.sum {
            best = Span { start, end: i + 1, sum: ending_here };
        }
    }

    best
}

/// `kadane_span_two_best` with the restart marked as rare.
pub fn kadane_span_branch_two_best(input: &[i32]) -> Span {
    assert!(!input.is_empty(), "the subarray has to be non-empty");

    let first = Span { start: 0, end: 1, sum: input[0] as i64 };
    let mut best_even = first;
    let mut best_odd = Span { start: 0, end: 0, sum: i64::MIN };
    let mut start = 0;
    let mut ending_here = first.sum;
    let (pairs, rest) = input[1..].as_chunks::<2>();

    for (p, pair) in pairs.iter().enumerate() {
        let i = 1 + 2 * p;

        step(&mut ending_here, &mut start, i, pair[0]);

        if ending_here > best_odd.sum {
            best_odd = Span { start, end: i + 1, sum: ending_here };
        }

        step(&mut ending_here, &mut start, i + 1, pair[1]);

        if ending_here > best_even.sum {
            best_even = Span { start, end: i + 2, sum: ending_here };
        }
    }

    for (j, &x) in rest.iter().enumerate() {
        let i = 1 + 2 * pairs.len() + j;

        step(&mut ending_here, &mut start, i, x);

        if ending_here > best_odd.sum {
            best_odd = Span { start, end: i + 1, sum: ending_here };
        }
    }

    if best_odd.sum > best_even.sum || (best_odd.sum == best_even.sum && best_odd.end < best_even.end) {
        best_odd
    } else {
        best_even
    }
}
