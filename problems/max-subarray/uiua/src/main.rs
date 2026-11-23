//! Maximum Subarray in Uiua, compiled to GPU compute shaders by gpuiua.
//!
//! The Uiua function in `max_subarray.ua` is two scans and a reduction, with no loop in it.
//! gpuiua turns the scans and the reduction into parallel shaders, so nothing here is written for the GPU by hand.
//! It computes in `f32` by default, which holds integers up to 2^24 exactly, and in `f64` with `--f64` on an adapter that supports it.
//! The answer is checked against Kadane's algorithm, so an input whose running sums leave that range fails the check instead of passing quietly.
//!
//! cargo run --release -- [--f64] [--reps N] [--sizes 20,24,26] [--interpreter] [--profile] [--wgsl N]

use gpuiua::{ArgSpec, Gpu, Precision, compile};
use std::time::Instant;
use uiua::{Array, Uiua, Value};

const SOURCE: &str = include_str!("../max_subarray.ua");

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

fn kadane(values: &[i32]) -> i64 {
    let mut ending_here = values[0] as i64;
    let mut best = ending_here;

    for &x in &values[1..] {
        ending_here = (x as i64).max(ending_here + x as i64);
        best = best.max(ending_here);
    }

    best
}

fn to_value(values: &[i32]) -> Value {
    let numbers: Vec<f64> = values.iter().map(|&x| x as f64).collect();

    Array::new([values.len()], &numbers[..]).into()
}

fn scalar(value: &Value) -> f64 {
    match value {
        Value::Num(a) => a.data()[0],
        other => panic!("expected a number, got {}", other.show()),
    }
}

fn median(mut times: Vec<f64>) -> f64 {
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

fn main() -> Result<(), gpuiua::Error> {
    let mut reps = 20;
    let mut exponents = vec![20, 24, 26];
    let mut use_f64 = false;
    let mut interpreter = false;
    let mut profile = false;
    let mut wgsl = None;
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--f64" => use_f64 = true,
            "--interpreter" => interpreter = true,
            "--profile" => profile = true,
            "--wgsl" => wgsl = Some(args.next().expect("--wgsl needs an exponent").parse::<u32>().unwrap()),
            "--reps" => reps = args.next().expect("--reps needs a number").parse().unwrap(),
            "--sizes" => {
                exponents = args.next().expect("--sizes needs a list").split(',').map(|e| e.parse().unwrap()).collect()
            }
            other => panic!("unknown option {other}"),
        }
    }

    // The shaders for one size, one file per stage in `shaders/`.
    if let Some(exponent) = wgsl {
        let precision = if use_f64 { Precision::F64 } else { Precision::F32 };
        let shaders = compile(SOURCE, "MaxSub")?.wgsl_for(precision, &[ArgSpec::shape(&[1usize << exponent])])?;
        let command = format!("cargo run --release -- --wgsl {exponent}{}", if use_f64 { " --f64" } else { "" });

        std::fs::create_dir_all("shaders").expect("the shaders directory");

        // gpuiua separates the stages with a `// stage N` line.
        for stage in shaders.to_string().split("// stage ").skip(1) {
            let (number, code) = stage.split_once('\n').unwrap();
            let path = format!("shaders/stage_{}.wgsl", number.trim());

            std::fs::write(
                &path,
                format!(
                    "// Stage {} of the shaders gpuiua makes from max_subarray.ua for 2^{exponent} values in {precision:?}.\n// {command}\n\n{}\n",
                    number.trim(),
                    code.trim()
                ),
            ).expect("a shader file");
            println!("wrote {path}");
        }

        return Ok(());
    }

    let gpu = if use_f64 { Gpu::new_f64()? } else { Gpu::new()? };
    let kernel = compile(SOURCE, "MaxSub")?;

    println!("adapter: {} ({:?})", gpu.adapter_name(), gpu.precision());

    // The running sums of 2^13 values stay below 2^24, which `f32` holds exactly, whatever the range.
    let check_len = if use_f64 { 1 << 20 } else { 1 << 13 };

    for &(lo, hi) in &[(-1000, 1000), (-1000, -1), (-3000, 1000)] {
        let values = random_values(check_len, 7, lo, hi);
        let got = scalar(&kernel.run(&gpu, &[to_value(&values)])?[0]);

        assert_eq!(got, kadane(&values) as f64, "the gpu and kadane differ on range [{lo}, {hi}]");
    }

    println!("checks: the gpu agrees with kadane on the cpu");
    println!(
        "{:<12} {:>8} {:>22} {:>26} {:>16}",
        "values", "size", "kernel GiB/s", "with upload and read back", "interpreter ms"
    );

    for exponent in exponents {
        let n = 1usize << exponent;
        let values = random_values(n, 1, -1000, 1000);
        let expected = kadane(&values);
        let input = to_value(&values);
        let answer = scalar(&kernel.run(&gpu, std::slice::from_ref(&input))?[0]);

        assert_eq!(answer, expected as f64, "the gpu and kadane differ on 2^{exponent} values");

        // The data is on the device and the shaders are compiled, only the kernels are timed, the mean of 10 back to back runs.
        let kernel_ms: Vec<f64> = (0..reps)
            .map(|_| kernel.time(&gpu, std::slice::from_ref(&input), 10).map(|t| t.as_secs_f64() * 1e3))
            .collect::<Result<_, _>>()?;
        let total_ms: Vec<f64> = (0..reps)
            .map(|_| {
                let start = Instant::now();

                kernel.run(&gpu, std::slice::from_ref(&input)).map(|_| start.elapsed().as_secs_f64() * 1e3)
            })
            .collect::<Result<_, _>>()?;

        let cpu = if interpreter {
            let start = Instant::now();
            let mut uiua = Uiua::with_safe_sys();

            uiua.push(input.clone());
            uiua.run_str(&format!("{SOURCE}\nMaxSub")).unwrap();
            assert_eq!(scalar(&uiua.take_stack()[0]), expected as f64);
            format!("{:.0}", start.elapsed().as_secs_f64() * 1e3)
        } else {
            "-".to_string()
        };

        if profile {
            for (stage, time) in kernel.profile(&gpu, std::slice::from_ref(&input))? {
                println!("    {:>8.3} ms  {stage}", time.as_secs_f64() * 1e3);
            }
        }

        let gib = n as f64 * 4.0 / (1u64 << 30) as f64;

        println!(
            "2^{exponent:<10} {:>5.0} MiB {:>22.1} {:>26.1} {:>16}",
            gib * 1024.0,
            gib / (median(kernel_ms) / 1e3),
            gib / (median(total_ms) / 1e3),
            cpu
        );
    }

    Ok(())
}
