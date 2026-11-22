//! The solutions that try every run, slow but obviously right.

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leetcode_examples() {
        assert_eq!(naive(&[-2, 1, -3, 4, -1, 2, 1, -5, 4]), 6);
        assert_eq!(naive(&[1]), 1);
        assert_eq!(naive(&[5, 4, -1, 7, 8]), 23);
    }
}
