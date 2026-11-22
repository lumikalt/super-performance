//! The starting point for a new problem.
//!
//! Copy this crate to `problems/<name>` on a new branch and rename the package.
//! Every optimization adds one function and one entry in `VARIANTS`.
//! The tests and the benchmark pick the new entry up on their own.

/// One way of solving the problem.
pub struct Variant {
    pub name: &'static str,
    pub run: fn(&[i32]) -> i64,
    /// The largest input this variant is benchmarked on, for the slow ones.
    pub max_n: usize,
}

/// The obvious solution, which every other variant is checked against.
pub fn naive(input: &[i32]) -> i64 {
    input.iter().map(|&x| x as i64).sum()
}

pub const VARIANTS: &[Variant] = &[Variant { name: "naive", run: naive, max_n: usize::MAX }];

#[cfg(test)]
mod tests {
    use super::*;
    use harness::data::random_i32s;

    #[test]
    fn every_variant_matches_naive() {
        for seed in 0..50 {
            for len in [1, 2, 3, 7, 8, 9, 63, 64, 65, 1000] {
                let input = random_i32s(len, seed, -1000, 1000);

                for variant in VARIANTS {
                    assert_eq!(
                        (variant.run)(&input),
                        naive(&input),
                        "{} differs on seed {seed}, length {len}",
                        variant.name
                    );
                }
            }
        }
    }
}
