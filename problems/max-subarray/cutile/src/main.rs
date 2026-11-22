//! Maximum Subarray on the GPU with cuTile Rust.
//!
//! Built with cutile 0.5.0 and CUDA 13.4 and run on an RTX 4090.
//!
//! The CUDA version joins four-number summaries with shuffles.
//! cuTile only reduces and scans over single numbers, so the summary of a tile is made from prefix sums.
//! With P the running sum of the tile and E the running sum before each value, which is P minus the value, then the sum of the tile is the last P, the best run that starts at its beginning is the largest P, the best run that ends at its end is the sum minus the smallest E, the best run inside it is the largest of P minus the smallest E so far, a scan with min.
//! A block walks over its tiles and joins them into one summary with the same formulas as the CPU versions.
//! The summaries of the blocks are joined on the host, in order.
//!
//! The input has to be a multiple of tile * blocks values, the tiles are never partly outside the input.
//! A block walks over `n / (tile * blocks)` tiles, the defaults are 1024 values and 1024 blocks.
//! Needs CUDA 13.3 or newer (`tileiras`) and a driver for it, set CUDA_TOOLKIT_PATH if CUDA is not in /usr/local/cuda.
//!
//! cargo run --release -- [--reps N] [--sizes 20,24,26,28] [--tile 1024] [--blocks 1024]

use cuda_async::device_operation::DeviceOp;
use cuda_core::Context;
use cutile::api::{copy_host_vec_to_device, zeros};
use cutile::error::Error;
use cutile::tensor::{IntoPartition, Partition, Tensor, ToHostVec};
use cutile::tile_kernel::TileKernel;
use std::sync::Arc;
use std::time::Instant;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn summarize<const T: i32, const R: i32>(
        sum: &mut Tensor<i64, { [1] }>,
        prefix: &mut Tensor<i64, { [1] }>,
        suffix: &mut Tensor<i64, { [1] }>,
        best: &mut Tensor<i64, { [1] }>,
        input: &Tensor<i32, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let first: i32 = pid.0 * R;
        let part: Partition<i32, { [T] }> = input.partition(shape![T]);

        // The summary of nothing, 2^60 below any sum so that adding two of them cannot overflow.
        let mut s: Tile<i64, { [1] }> = constant(0i64, shape![1]);
        let mut p: Tile<i64, { [1] }> = constant(-1_152_921_504_606_846_976i64, shape![1]);
        let mut u: Tile<i64, { [1] }> = constant(-1_152_921_504_606_846_976i64, shape![1]);
        let mut b: Tile<i64, { [1] }> = constant(-1_152_921_504_606_846_976i64, shape![1]);

        for j in 0i32..R {
            let raw: Tile<i32, { [T] }> = part.load([first + j]);
            let x: Tile<i64, { [T] }> = exti(raw);
            let running: Tile<i64, { [T] }> = scan_sum(x, 0i32, reverse::Forward, 0i64);
            let before: Tile<i64, { [T] }> = running - x;
            let lowest: Tile<i64, { [T] }> =
                scan(before, 0i32, reverse::Forward, 1_152_921_504_606_846_976i64, |acc, v| min(acc, v));

            // A reduction gives a tile of no dimensions, reshaped to one value to be combined with the carried tiles.
            let sum_all: Tile<i64, { [] }> = reduce_sum(x, 0i32);
            let max_running: Tile<i64, { [] }> = reduce_max(running, 0i32);
            let min_before: Tile<i64, { [] }> = reduce_min(before, 0i32);
            let max_inside: Tile<i64, { [] }> = reduce_max(running - lowest, 0i32);

            let t_sum: Tile<i64, { [1] }> = sum_all.reshape(shape![1]);
            let t_prefix: Tile<i64, { [1] }> = max_running.reshape(shape![1]);
            let t_suffix: Tile<i64, { [1] }> = t_sum - min_before.reshape(shape![1]);
            let t_best: Tile<i64, { [1] }> = max_inside.reshape(shape![1]);

            // The summary so far followed by this tile, every formula uses the values from before the update.
            let joined_best: Tile<i64, { [1] }> = max_tile(max_tile(b, t_best), u + t_prefix);
            let joined_prefix: Tile<i64, { [1] }> = max_tile(p, s + t_prefix);
            let joined_suffix: Tile<i64, { [1] }> = max_tile(t_suffix, t_sum + u);
            let joined_sum: Tile<i64, { [1] }> = s + t_sum;

            b = joined_best;
            p = joined_prefix;
            u = joined_suffix;
            s = joined_sum;
        }

        sum.store(s);
        prefix.store(p);
        suffix.store(u);
        best.store(b);
    }
}

use kernels::summarize;

/// What is known about a stretch, enough to join it with its neighbours.
#[derive(Clone, Copy)]
struct Summary {
    sum: i64,
    prefix: i64,
    suffix: i64,
    best: i64,
}

impl Summary {
    fn then(self, next: Self) -> Self {
        Self {
            sum: self.sum + next.sum,
            prefix: self.prefix.max(self.sum + next.prefix),
            suffix: next.suffix.max(next.sum + self.suffix),
            best: self.best.max(next.best).max(self.suffix + next.prefix),
        }
    }
}

fn kadane(values: &[i32]) -> i64 {
    let mut ending_here = values[0] as i64;
    let mut best = ending_here;

    for &x in &values[1..] {
        ending_here = (x as i64).max(ending_here + x as i64);
        best = best.max(ending_here);
    }

    best
}

/// The same generator as the Rust harness, so the inputs are the same.
fn random_values(n: usize, seed: u64, lo: i32, hi: i32) -> Vec<i32> {
    let mut state = seed;
    let span = (hi as i64 - lo as i64 + 1) as u64;

    (0..n)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);

            let mut z = state;

            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (lo as i64 + (z % span) as i64) as i32
        })
        .collect()
}

fn main() -> Result<(), Error> {
    let mut reps = 20;
    let mut exponents = vec![20, 24, 26, 28];
    let mut tile = 1024;
    let mut blocks = 1024;
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--reps" => reps = args.next().expect("--reps needs a number").parse().unwrap(),
            "--sizes" => {
                exponents = args.next().expect("--sizes needs a list").split(',').map(|e| e.parse().unwrap()).collect()
            }
            "--tile" => tile = args.next().expect("--tile needs a number").parse().unwrap(),
            "--blocks" => blocks = args.next().expect("--blocks needs a number").parse().unwrap(),
            other => panic!("unknown option {other}"),
        }
    }

    let device = Context::new(0)?;
    let stream = device.new_stream()?;

    println!("tile {tile} values, {blocks} blocks");
    println!("{:<12} {:>8} {:>22} {:>26}", "values", "size", "kernel GiB/s", "with copy back and join");

    for &(lo, hi) in &[(-1000, 1000), (-1000, -1), (-3000, 1000)] {
        let n = tile * blocks * 4;
        let values = random_values(n, 7, lo, hi);
        let got = run(&stream, &values, 1, tile, blocks)?.1;

        assert_eq!(got, kadane(&values), "cutile and kadane differ on range [{lo}, {hi}]");
    }

    println!("checks: the gpu agrees with kadane on the cpu");

    for exponent in exponents {
        let n = 1usize << exponent;
        let values = random_values(n, 1, -1000, 1000);
        let expected = kadane(&values);
        let (kernel_ms, answer, total_ms) = run(&stream, &values, reps, tile, blocks)?;

        assert_eq!(answer, expected, "cutile and kadane differ on 2^{exponent} values");

        let gib = n as f64 * 4.0 / (1u64 << 30) as f64;
        println!(
            "2^{exponent:<10} {:>5.0} MiB {:>22.1} {:>26.1}",
            gib * 1024.0,
            gib / (kernel_ms / 1e3),
            gib / (total_ms / 1e3)
        );
    }

    Ok(())
}

/// Runs the kernel `reps` times and returns the median kernel time and the median time with the copy back and the join in milliseconds, together with the answer.
fn run(
    stream: &Arc<cuda_core::Stream>,
    values: &[i32],
    reps: usize,
    tile: usize,
    blocks: usize,
) -> Result<(f64, i64, f64), Error> {
    assert_eq!(values.len() % (tile * blocks), 0, "the input has to be a multiple of tile * blocks");

    let per_block = values.len() / (tile * blocks);
    let generics = vec![tile.to_string(), per_block.to_string()];
    let input: Arc<Tensor<i32>> = copy_host_vec_to_device(&Arc::new(values.to_vec())).sync_on(stream)?.into();
    let mut kernel_times = Vec::new();
    let mut total_times = Vec::new();
    let mut answer = 0;

    // The first run compiles the kernel, it is not counted.
    for run in 0..=reps {
        let sum: Partition<Tensor<i64>> = zeros(&[blocks]).sync_on(stream)?.partition([1]);
        let prefix: Partition<Tensor<i64>> = zeros(&[blocks]).sync_on(stream)?.partition([1]);
        let suffix: Partition<Tensor<i64>> = zeros(&[blocks]).sync_on(stream)?.partition([1]);
        let best: Partition<Tensor<i64>> = zeros(&[blocks]).sync_on(stream)?.partition([1]);

        let start = Instant::now();
        let (sum, prefix, suffix, best, _) =
            summarize(sum, prefix, suffix, best, input.clone()).generics(generics.clone()).sync_on(stream)?;
        let kernel = start.elapsed().as_secs_f64() * 1e3;

        let sums = sum.unpartition().to_host_vec().sync_on(stream)?;
        let prefixes = prefix.unpartition().to_host_vec().sync_on(stream)?;
        let suffixes = suffix.unpartition().to_host_vec().sync_on(stream)?;
        let bests = best.unpartition().to_host_vec().sync_on(stream)?;

        let joined = (0..blocks)
            .map(|i| Summary { sum: sums[i], prefix: prefixes[i], suffix: suffixes[i], best: bests[i] })
            .reduce(Summary::then)
            .unwrap();
        let total = start.elapsed().as_secs_f64() * 1e3;

        answer = joined.best;

        if run > 0 {
            kernel_times.push(kernel);
            total_times.push(total);
        }
    }

    let median = |mut times: Vec<f64>| {
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        times[times.len() / 2]
    };

    Ok((median(kernel_times), answer, median(total_times)))
}
