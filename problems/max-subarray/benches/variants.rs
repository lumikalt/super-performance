use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use harness::{data::random_i32s, environment};
use max_subarray::VARIANTS;
use std::time::Duration;

fn variants(c: &mut Criterion) {
    eprintln!("{}", environment::describe());

    let mut group = c.benchmark_group("max-subarray");

    group.sample_size(20);
    group.measurement_time(Duration::from_secs(3));

    // From something that fits in cache to something that does not.
    for log in [10, 14, 20, 26] {
        let n = 1usize << log;
        let input = random_i32s(n, 1, -1000, 1000);

        group.throughput(Throughput::Bytes((n * size_of::<i32>()) as u64));

        for variant in VARIANTS {
            if n > variant.max_n {
                continue;
            }

            group.bench_with_input(BenchmarkId::new(variant.name, n), &input, |b, input| {
                b.iter(|| (variant.run)(black_box(input)))
            });
        }
    }

    group.finish();
}

criterion_group!(benches, variants);
criterion_main!(benches);
