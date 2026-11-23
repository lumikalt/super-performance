//! The solutions that try every run, slow but obviously right.

use crate::Span;

/// Tries every start and every end and keeps the best sum.
/// Quadratic, so it is only usable on small inputs, but it is obviously right.
pub fn naive(input: &[i32]) -> i64 {
    assert!(!input.is_empty(), "the subarray has to be non-empty");

    let mut best = i64::MIN;

    for start in 0..input.len() {
        let mut sum = 0i64;

        for &x in &input[start..] {
            sum += x as i64;
            best = best.max(sum);
        }
    }

    best
}

/// Tries every end and every start, with prefix sums so a sum costs one subtraction.
pub fn naive_span(input: &[i32]) -> Span {
    assert!(!input.is_empty(), "the subarray has to be non-empty");

    let mut prefix = vec![0i64; input.len() + 1];

    for (i, &x) in input.iter().enumerate() {
        prefix[i + 1] = prefix[i] + x as i64;
    }

    let mut best = Span { start: 0, end: 1, sum: input[0] as i64 };

    for end in 1..=input.len() {
        for start in 0..end {
            let sum = prefix[end] - prefix[start];

            if sum > best.sum {
                best = Span { start, end, sum };
            }
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leetcode_examples() {
        assert_eq!(naive(&[-2, 1, -3, 4, -1, 2, 1, -5, 4]), 6);
        assert_eq!(naive(&[1]), 1);
        assert_eq!(naive(&[5, 4, -1, 7, 8]), 23);
    }

    #[test]
    fn spans_are_the_first_best_run() {
        assert_eq!(naive_span(&[-2, 1, -3, 4, -1, 2, 1, -5, 4]), Span { start: 3, end: 7, sum: 6 });
        // Ends first, then starts first, so the zero at the front belongs to the run.
        assert_eq!(naive_span(&[0, 3, -3, 3]), Span { start: 0, end: 2, sum: 3 });
        assert_eq!(naive_span(&[-5, -2, -2]), Span { start: 1, end: 2, sum: -2 });
    }
}
