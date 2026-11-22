//! Runs one variant on a fixed input many times, so a profiler or `perf stat` can look at it.
//!
//! usage.
//! Run <variant> <values> <repetitions> [lowest highest]
//!
//! The values are random between `lowest` and `highest`, which default to -1000 and 1000.
//! The data matters for some variants, a mostly negative range makes kadane's branch hard to predict.

use harness::data::random_i32s;
use max_subarray::VARIANTS;
use std::hint::black_box;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() != 4 && args.len() != 6 {
        eprintln!("usage: run <variant> <values> <repetitions> [lowest highest]");
        std::process::exit(2);
    }

    let name = &args[1];
    let n: usize = args[2].parse().expect("values has to be a number");
    let reps: usize = args[3].parse().expect("repetitions has to be a number");
    let lowest: i32 = args.get(4).map_or(-1000, |s| s.parse().expect("lowest has to be a number"));
    let highest: i32 = args.get(5).map_or(1000, |s| s.parse().expect("highest has to be a number"));
    let variant = VARIANTS.iter().find(|v| v.name == name).expect("unknown variant");
    let input = random_i32s(n, 1, lowest, highest);

    // One run first, so page faults and cold caches do not count.
    black_box((variant.run)(black_box(&input)));

    let mut sink = 0i64;

    for _ in 0..reps {
        sink ^= (variant.run)(black_box(&input));
    }

    println!("{name}: {n} values, {reps} runs, checksum {sink}");
}
