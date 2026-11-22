//! Maximum Subarray (LeetCode 53).
//!
//! The largest sum of a non-empty run of neighbouring values.
//! Every step of the write-up adds one function and one entry in `VARIANTS`.

mod kadane;
mod naive;

pub use kadane::*;
pub use naive::*;

/// One way of solving the problem.
pub struct Variant {
    pub name: &'static str,
    pub run: fn(&[i32]) -> i64,
    /// The largest input this variant is benchmarked on, for the slow ones.
    pub max_n: usize,
}

pub const VARIANTS: &[Variant] = &[
    Variant { name: "naive", run: naive, max_n: 1 << 14 },
    Variant { name: "kadane", run: kadane, max_n: usize::MAX },
];

#[cfg(test)]
mod tests {
    use super::*;
    use harness::data::random_i32s;

    #[test]
    fn variants_agree_on_edge_cases() {
        // All negative inputs are where a solution that allows an empty subarray goes wrong.
        let cases: &[&[i32]] = &[&[-1], &[7], &[-3, -1, -2], &[-5, -5], &[0], &[0, 0, 0], &[i32::MIN, i32::MAX]];

        for case in cases {
            for variant in VARIANTS {
                assert_eq!((variant.run)(case), naive(case), "{} differs on {case:?}", variant.name);
            }
        }
    }

    #[test]
    fn variants_agree_on_random_inputs() {
        // Mostly positive, mixed and all negative values.
        for (lo, hi) in [(-1000, 1000), (-1000, 300), (-1000, -1), (1, 1000)] {
            for seed in 0..30 {
                for len in [1, 2, 3, 7, 8, 9, 63, 64, 65, 1000] {
                    let input = random_i32s(len, seed, lo, hi);

                    for variant in VARIANTS {
                        assert_eq!(
                            (variant.run)(&input),
                            naive(&input),
                            "{} differs on seed {seed}, length {len}, range {lo}..={hi}",
                            variant.name
                        );
                    }
                }
            }
        }
    }
}
