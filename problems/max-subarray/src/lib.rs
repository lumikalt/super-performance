//! Maximum Subarray (LeetCode 53).
//!
//! The largest sum of a non-empty run of neighbouring values.
//! Every step of the write-up adds one function and one entry in `VARIANTS`.

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod avx2;
mod chains;
mod kadane;
mod naive;
mod read;
mod summary;
mod threads;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub use avx2::gather as simd_gather;
pub use chains::*;
pub use kadane::*;
pub use naive::*;
pub use read::*;
pub use threads::*;

/// One way of solving the problem.
pub struct Variant {
    pub name: &'static str,
    pub run: fn(&[i32]) -> i64,
    /// The largest input this variant is benchmarked on, for the slow ones.
    pub max_n: usize,
}

/// The values the problem promises, LeetCode limits them to this much.
/// It lets the SIMD variants add up `i32` lanes for a while without overflowing.
pub const MAX_ABS: i32 = 10_000;

pub const VARIANTS: &[Variant] = &[
    Variant { name: "naive", run: naive, max_n: 1 << 14 },
    Variant { name: "kadane", run: kadane, max_n: usize::MAX },
    Variant { name: "kadane_two_best", run: kadane_two_best, max_n: usize::MAX },
    Variant { name: "chains2", run: chains2, max_n: usize::MAX },
    Variant { name: "chains4", run: chains4, max_n: usize::MAX },
    Variant { name: "chains8", run: chains8, max_n: usize::MAX },
    Variant { name: "chains16", run: chains16, max_n: usize::MAX },
    Variant { name: "chains2_unchecked", run: chains2_unchecked, max_n: usize::MAX },
    Variant { name: "chains4_unchecked", run: chains4_unchecked, max_n: usize::MAX },
    Variant { name: "chains8_unchecked", run: chains8_unchecked, max_n: usize::MAX },
    Variant { name: "chains16_unchecked", run: chains16_unchecked, max_n: usize::MAX },
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    Variant { name: "simd_gather", run: simd_gather, max_n: usize::MAX },
    Variant { name: "threads_equal", run: threads_equal, max_n: usize::MAX },
];

#[cfg(test)]
mod tests {
    use super::*;
    use harness::data::random_i32s;

    #[test]
    fn variants_agree_on_edge_cases() {
        // All negative inputs are where a solution that allows an empty subarray goes wrong.
        let cases: &[&[i32]] = &[
            &[-1],
            &[7],
            &[-3, -1, -2],
            &[-5, -5],
            &[0],
            &[0, 0, 0],
            &[-MAX_ABS, MAX_ABS],
            &[MAX_ABS; 5],
            &[-MAX_ABS; 5],
        ];

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

    #[test]
    fn variants_agree_with_kadane_on_large_inputs() {
        // Around the block size of the SIMD variants, where their loops change shape.
        let block = 8 * 4096;

        for len in [block - 1, block, block + 1, 2 * block, 3 * block + 17, 100_003, 600_007, 1_000_003] {
            for (seed, (lo, hi)) in
                [(-1000, 1000), (-1000, -1), (-10_000, 10_000), (-300, 1000)].into_iter().enumerate()
            {
                let input = random_i32s(len, seed as u64, lo, hi);

                for variant in VARIANTS {
                    if variant.max_n < len {
                        continue;
                    }

                    assert_eq!(
                        (variant.run)(&input),
                        kadane(&input),
                        "{} differs on length {len}, range {lo}..={hi}",
                        variant.name
                    );
                }
            }
        }
    }
}
