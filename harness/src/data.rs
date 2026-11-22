use crate::rng::SplitMix64;

/// `len` values in `lo..=hi`, the same for the same seed.
pub fn random_i32s(len: usize, seed: u64, lo: i32, hi: i32) -> Vec<i32> {
    let mut rng = SplitMix64::new(seed);

    (0..len).map(|_| rng.range_i32(lo, hi)).collect()
}
