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
