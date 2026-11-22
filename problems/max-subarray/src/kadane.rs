//! Kadane's algorithm and the ways of writing it.

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
