//! The reference for what plain reading costs.

/// Not a solution, the sum of all values, which reads every value once and does nothing else.
/// The benchmark prints it as a reference for what plain reading costs.
/// It is not a hard limit, a variant that reads several streams at once can beat it on big inputs.
pub fn read_sum(input: &[i32]) -> i64 {
    input.iter().map(|&x| x as i64).sum()
}
