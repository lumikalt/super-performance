//! Runs one variant on a fixed input many times, so a profiler or `perf stat` can look at it.
//!
//! usage.
//! Run <variant> <values> <repetitions>

use harness::data::random_i32s;
use max_subarray::VARIANTS;
use std::hint::black_box;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() != 4 {
        eprintln!("usage: run <variant> <values> <repetitions>");
        std::process::exit(2);
    }

    let name = &args[1];
    let n: usize = args[2].parse().expect("values has to be a number");
    let reps: usize = args[3].parse().expect("repetitions has to be a number");
    let variant = VARIANTS.iter().find(|v| v.name == name).expect("unknown variant");
    let input = random_i32s(n, 1, -1000, 1000);

    // One run first, so page faults and cold caches do not count.
    black_box((variant.run)(black_box(&input)));

    let mut sink = 0i64;

    for _ in 0..reps {
        sink ^= (variant.run)(black_box(&input));
    }

    println!("{name}: {n} values, {reps} runs, checksum {sink}");
}
