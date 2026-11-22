// Maximum Subarray on the GPU.
//
// The same summary as the CPU versions, four numbers that are enough to join two neighbouring stretches.
// Every block takes a tile of the input and every thread summarizes 16 values of it.
// The summaries are joined in order with warp shuffles and shared memory, one summary per block comes out.
// A second kernel joins the summaries of the blocks.
// The sums are 64 bit, so this works for any `int` input, the SIMD versions on the CPU need the values within +-10^4.
//
// build  nvcc -O3 -std=c++17 -arch=native -o max_subarray max_subarray.cu
// run    ./max_subarray [--reps N] [--sizes 20,24,26,28]
//
// It first checks the GPU against Kadane's algorithm on the CPU, then times the kernels on inputs of 2^k values.
// The timing is of the kernels only, with the input already on the GPU, in GiB/s of input read.
// Two references are printed next to it.
// `read_sum` only adds up all the values, and `copy` is a device to device copy.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#define CHECK(call)                                                                                                    \
    do {                                                                                                               \
        cudaError_t error = (call);                                                                                    \
        if (error != cudaSuccess) {                                                                                    \
            fprintf(stderr, "%s:%d %s failed: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(error));            \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

typedef long long i64;

// What is known about a stretch, and `valid` is 0 for the empty stretch that joins with anything.
struct Summary {
    i64 sum;
    i64 prefix;
    i64 suffix;
    i64 best;
    int valid;
};

__host__ __device__ static inline i64 larger(i64 a, i64 b) { return a > b ? a : b; }

__host__ __device__ static inline Summary of(int x) { return Summary{x, x, x, x, 1}; }

// The next value, this is the Kadane step.
__host__ __device__ static inline void push(Summary &s, int x) {
    s.sum += x;
    s.prefix = larger(s.prefix, s.sum);
    s.suffix = larger((i64)x, s.suffix + x);
    s.best = larger(s.best, s.suffix);
}

// The stretch `a` followed by the stretch `b`, a run can now cross the join.
__host__ __device__ static inline Summary join(const Summary &a, const Summary &b) {
    if (!a.valid) return b;
    if (!b.valid) return a;

    Summary r;
    r.sum = a.sum + b.sum;
    r.prefix = larger(a.prefix, a.sum + b.prefix);
    r.suffix = larger(b.suffix, b.sum + a.suffix);
    r.best = larger(larger(a.best, b.best), a.suffix + b.prefix);
    r.valid = 1;
    return r;
}

__device__ static inline Summary shuffle_down(const Summary &s, int offset) {
    Summary r;
    r.sum = __shfl_down_sync(0xffffffffu, s.sum, offset);
    r.prefix = __shfl_down_sync(0xffffffffu, s.prefix, offset);
    r.suffix = __shfl_down_sync(0xffffffffu, s.suffix, offset);
    r.best = __shfl_down_sync(0xffffffffu, s.best, offset);
    r.valid = __shfl_down_sync(0xffffffffu, s.valid, offset);
    return r;
}

// Joins the summaries of the lanes of a warp in lane order, the result is in lane 0.
// The offsets grow, so a lane always joins two neighbouring stretches, and the join is not commutative.
__device__ static inline Summary join_warp(Summary s, int lanes) {
    for (int offset = 1; offset < lanes; offset <<= 1) s = join(s, shuffle_down(s, offset));
    return s;
}

constexpr int THREADS = 256;
constexpr int PER_THREAD = 16;
constexpr int TILE = THREADS * PER_THREAD;
constexpr int WARPS = THREADS / 32;

// One summary for every tile of the input.
__global__ void summarize(const int *__restrict__ input, size_t n, Summary *tiles) {
    const size_t base = (size_t)blockIdx.x * TILE + (size_t)threadIdx.x * PER_THREAD;
    Summary s = Summary{0, 0, 0, 0, 0};

    if (base + PER_THREAD <= n) {
        // Four 128 bit loads, the tile and the buffer are aligned.
        const int4 *vectors = reinterpret_cast<const int4 *>(input + base);
        int values[PER_THREAD];

        for (int v = 0; v < PER_THREAD / 4; v++) {
            const int4 w = vectors[v];
            values[4 * v + 0] = w.x;
            values[4 * v + 1] = w.y;
            values[4 * v + 2] = w.z;
            values[4 * v + 3] = w.w;
        }

        s = of(values[0]);

        for (int i = 1; i < PER_THREAD; i++) push(s, values[i]);
    } else {
        // The end of the input, a thread with a few values or none.
        for (size_t i = base; i < base + PER_THREAD && i < n; i++) {
            if (!s.valid) s = of(input[i]);
            else push(s, input[i]);
        }
    }

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    __shared__ Summary warps[WARPS];

    s = join_warp(s, 32);

    if (lane == 0) warps[warp] = s;

    __syncthreads();

    if (warp == 0) {
        Summary t = lane < WARPS ? warps[lane] : Summary{0, 0, 0, 0, 0};

        t = join_warp(t, WARPS);

        if (lane == 0) tiles[blockIdx.x] = t;
    }
}

constexpr int FINAL_THREADS = 1024;

// Joins the summaries of the tiles, in order, with one block.
__global__ void combine(const Summary *__restrict__ tiles, size_t count, Summary *result) {
    const size_t per_thread = (count + FINAL_THREADS - 1) / FINAL_THREADS;
    const size_t begin = (size_t)threadIdx.x * per_thread;
    Summary s = Summary{0, 0, 0, 0, 0};

    for (size_t i = begin; i < begin + per_thread && i < count; i++) s = join(s, tiles[i]);

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    __shared__ Summary warps[FINAL_THREADS / 32];

    s = join_warp(s, 32);

    if (lane == 0) warps[warp] = s;

    __syncthreads();

    if (warp == 0) {
        Summary t = warps[lane];

        t = join_warp(t, FINAL_THREADS / 32);

        if (lane == 0) *result = t;
    }
}

// The reference, a plain sum of all values that reads each of them once.
__global__ void read_sum(const int *__restrict__ input, size_t n, unsigned long long *total) {
    const size_t vectors = n / 4;
    const int4 *source = reinterpret_cast<const int4 *>(input);
    i64 sum = 0;

    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < vectors; i += (size_t)gridDim.x * blockDim.x) {
        const int4 w = source[i];
        sum += (i64)w.x + w.y + w.z + w.w;
    }

    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, offset);

    if (threadIdx.x % 32 == 0) atomicAdd(total, (unsigned long long)sum);
}

// The input from the GPU's point of view.
struct Device {
    int *input = nullptr;
    int *copy = nullptr;
    Summary *tiles = nullptr;
    Summary *result = nullptr;
    unsigned long long *total = nullptr;
    size_t capacity = 0;
    int sms = 0;
};

static void allocate(Device &d, size_t n) {
    d.capacity = n;
    CHECK(cudaMalloc(&d.input, n * sizeof(int)));
    CHECK(cudaMalloc(&d.copy, n * sizeof(int)));
    CHECK(cudaMalloc(&d.tiles, ((n + TILE - 1) / TILE) * sizeof(Summary)));
    CHECK(cudaMalloc(&d.result, sizeof(Summary)));
    CHECK(cudaMalloc(&d.total, sizeof(unsigned long long)));
}

static void release(Device &d) {
    cudaFree(d.input);
    cudaFree(d.copy);
    cudaFree(d.tiles);
    cudaFree(d.result);
    cudaFree(d.total);
}

static void launch_max_subarray(const Device &d, size_t n) {
    const size_t blocks = (n + TILE - 1) / TILE;

    summarize<<<(unsigned)blocks, THREADS>>>(d.input, n, d.tiles);
    combine<<<1, FINAL_THREADS>>>(d.tiles, blocks, d.result);
}

static void launch_read_sum(const Device &d, size_t n) {
    CHECK(cudaMemsetAsync(d.total, 0, sizeof(unsigned long long)));
    read_sum<<<d.sms * 16, 256>>>(d.input, n, d.total);
}

static i64 max_subarray_gpu(const Device &d, size_t n) {
    launch_max_subarray(d, n);

    Summary result;
    CHECK(cudaMemcpy(&result, d.result, sizeof(Summary), cudaMemcpyDeviceToHost));
    return result.best;
}

// Kadane's algorithm on the CPU, which the GPU result is checked against.
static i64 kadane(const int *values, size_t n) {
    i64 ending_here = values[0], best = values[0];

    for (size_t i = 1; i < n; i++) {
        ending_here = larger((i64)values[i], ending_here + values[i]);
        best = larger(best, ending_here);
    }

    return best;
}

// The same generator as the Rust harness, so the inputs are the same.
struct SplitMix64 {
    uint64_t state;

    uint64_t next() {
        state += 0x9E3779B97F4A7C15ull;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    int range(int lo, int hi) {
        const uint64_t span = (uint64_t)((i64)hi - (i64)lo + 1);
        return (int)((i64)lo + (i64)(next() % span));
    }
};

static std::vector<int> random_values(size_t n, uint64_t seed, int lo, int hi) {
    SplitMix64 rng{seed};
    std::vector<int> values(n);

    for (size_t i = 0; i < n; i++) values[i] = rng.range(lo, hi);

    return values;
}

static bool check(Device &d, size_t n, uint64_t seed, int lo, int hi) {
    const std::vector<int> values = random_values(n, seed, lo, hi);

    CHECK(cudaMemcpy(d.input, values.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    const i64 expected = kadane(values.data(), n);
    const i64 got = max_subarray_gpu(d, n);

    if (expected != got) {
        printf("FAIL n=%zu seed=%llu range=[%d,%d] expected %lld got %lld\n", n, (unsigned long long)seed, lo, hi, expected,
               got);
        return false;
    }

    return true;
}

static bool run_checks(Device &d) {
    // Around the sizes where the tiles and the threads change shape.
    const size_t sizes[] = {1, 2, 3, 15, 16, 17, 63, 64, 65, 255, 256, 257, 4095, 4096, 4097, 8191, 8192, 8193, 100003, 1048576 + 13};
    const int ranges[][2] = {{-1000, 1000}, {-1000, -1}, {1, 1000}, {-3000, 1000}, {-1000, 3000}, {0, 0}, {-2000000000, 2000000000}};
    int cases = 0;

    for (size_t n : sizes) {
        for (const auto &range : ranges) {
            for (uint64_t seed = 0; seed < 3; seed++) {
                if (!check(d, n, seed, range[0], range[1])) return false;
                cases++;
            }
        }
    }

    printf("checks: %d cases agree with kadane on the cpu\n", cases);
    return true;
}

// Time of `count` runs of a launch in milliseconds, one entry per run.
template <typename Launch> static std::vector<float> time_runs(int count, Launch launch) {
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    launch();
    CHECK(cudaDeviceSynchronize());

    std::vector<float> times;

    for (int run = 0; run < count; run++) {
        CHECK(cudaEventRecord(start));
        launch();
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));

        float ms;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaGetLastError());
    return times;
}

struct Stats {
    double median, best, spread_percent;
};

// Throughput statistics from times in milliseconds, `bytes` is what one run moves.
static Stats throughput(std::vector<float> times, double bytes) {
    std::sort(times.begin(), times.end());

    const double fastest = times.front(), slowest = times.back(), middle = times[times.size() / 2];
    const double gib = 1024.0 * 1024.0 * 1024.0;

    return Stats{bytes / (middle / 1e3) / gib, bytes / (fastest / 1e3) / gib, (slowest - fastest) * 100.0 / middle};
}

int main(int argc, char **argv) {
    int reps = 50;
    std::vector<int> exponents = {20, 24, 26, 28};

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--reps") && i + 1 < argc) {
            reps = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--sizes") && i + 1 < argc) {
            exponents.clear();

            for (char *token = strtok(argv[++i], ","); token; token = strtok(nullptr, ",")) exponents.push_back(atoi(token));
        } else {
            fprintf(stderr, "usage: max_subarray [--reps N] [--sizes 20,24,26,28]\n");
            return 2;
        }
    }

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s, %d SMs, %.1f GiB, compute capability %d.%d\n", prop.name, prop.multiProcessorCount,
           prop.totalGlobalMem / 1073741824.0, prop.major, prop.minor);

    size_t largest = 1;

    for (int e : exponents) largest = std::max(largest, (size_t)1 << e);

    Device d;
    d.sms = prop.multiProcessorCount;
    allocate(d, std::max(largest, (size_t)(1 << 21)));

    if (!run_checks(d)) {
        release(d);
        return 1;
    }

    printf("\n%-12s %-9s %-26s %-26s %-26s\n", "values", "size", "max_subarray GiB/s", "read_sum GiB/s", "copy GiB/s");
    printf("%-12s %-9s %-26s %-26s %-26s\n", "", "", "median (best, spread)", "median (best, spread)", "median (best, spread)");

    for (int e : exponents) {
        const size_t n = (size_t)1 << e;
        const std::vector<int> values = random_values(n, 1, -1000, 1000);

        CHECK(cudaMemcpy(d.input, values.data(), n * sizeof(int), cudaMemcpyHostToDevice));

        if (max_subarray_gpu(d, n) != kadane(values.data(), n)) {
            printf("FAIL on the benchmark input of 2^%d values\n", e);
            release(d);
            return 1;
        }

        const double bytes = (double)n * sizeof(int);
        const Stats gpu = throughput(time_runs(reps, [&] { launch_max_subarray(d, n); }), bytes);
        const Stats reference = throughput(time_runs(reps, [&] { launch_read_sum(d, n); }), bytes);
        const Stats copy = throughput(time_runs(reps, [&] { CHECK(cudaMemcpyAsync(d.copy, d.input, n * sizeof(int), cudaMemcpyDeviceToDevice)); }), 2 * bytes);

        char size[16];
        snprintf(size, sizeof(size), "%.0f MiB", bytes / 1048576.0);
        printf("2^%-10d %-9s %7.1f (%7.1f, %4.1f%%)    %7.1f (%7.1f, %4.1f%%)    %7.1f (%7.1f, %4.1f%%)\n", e, size, gpu.median,
               gpu.best, gpu.spread_percent, reference.median, reference.best, reference.spread_percent, copy.median, copy.best,
               copy.spread_percent);
    }

    release(d);
    return 0;
}
