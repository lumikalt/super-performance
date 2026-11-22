#!/usr/bin/env bash
# Acceptance test for a rented GPU instance, to run before trusting it for benchmarks.
#
# usage: gpu-acceptance.sh [--burn-seconds N] [--no-clock-lock] [--no-kernels] [--out DIR]
#
# It checks
#   - what the environment is and how much root you have
#   - the GPU's identity and state, and whether it is a whole dedicated GPU
#   - whether the profiler may read the hardware counters
#   - whether the clocks can be locked
#   - how repeatable small benchmarks are on it
#   - how clocks, power and temperature behave under a sustained load
#
# Needs bash, nvidia-smi and awk.
# The kernel based parts also need nvcc, and the profiler test needs ncu.
# The clock lock test changes the GPU clocks for a moment and puts them back, skip it with --no-clock-lock.
# Everything is written to a directory, and summary.txt in it is the part to keep or paste.
set -u

burn_seconds=60
try_clock_lock=1
run_kernels=1
out=""

while [ $# -gt 0 ]; do
    case $1 in
        --burn-seconds) burn_seconds=$2; shift 2 ;;
        --no-clock-lock) try_clock_lock=0; shift ;;
        --no-kernels) run_kernels=0; shift ;;
        --out) out=$2; shift 2 ;;
        -h | --help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done

out=${out:-gpu-acceptance-$(hostname)-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$out"
summary=$out/summary.txt
: >"$summary"
n_pass=0 n_warn=0 n_fail=0

say() { printf '%s\n' "$*" | tee -a "$summary"; }
section() { say ""; say "== $1"; }
pass() { n_pass=$((n_pass + 1)); say "$(printf 'PASS  %-20s %s' "$1" "$2")"; }
warn() { n_warn=$((n_warn + 1)); say "$(printf 'WARN  %-20s %s' "$1" "$2")"; }
fail() { n_fail=$((n_fail + 1)); say "$(printf 'FAIL  %-20s %s' "$1" "$2")"; }
info() { say "$(printf 'INFO  %-20s %s' "$1" "$2")"; }

# One field of the first GPU, or N/A.
q() {
    local value
    value=$(nvidia-smi -i 0 --query-gpu="$1" --format=csv,noheader,nounits 2>/dev/null | head -1 | sed 's/^ *//; s/ *$//')
    if [ -z "$value" ] || [[ $value == *"N/A"* ]] || [[ $value == *"not supported"* ]]; then echo "N/A"; else echo "$value"; fi
}

is_number() { [[ ${1:-} =~ ^[0-9]+(\.[0-9]+)?$ ]]; }

# true when version $1 is at least $2
version_at_least() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }

# median, min, max and the spread in percent of the median, of the numbers on stdin
stats() {
    sort -g | awk '{a[NR]=$1} END {n=NR; if (!n) {print "n/a"; exit}
        med=(n%2)?a[(n+1)/2]:(a[n/2]+a[n/2+1])/2
        printf "%.1f %.1f %.1f %.1f", med, a[1], a[n], (a[n]-a[1])*100/med}'
}

say "gpu acceptance test, $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname)"

# ------------------------------------------------------------------ environment
section "environment"

info "user" "$(id -un) (uid $(id -u))"

virt=$(systemd-detect-virt 2>/dev/null || true)
if [ -f /.dockerenv ] || grep -qE 'docker|containerd|kubepods|lxc' /proc/1/cgroup 2>/dev/null; then
    kind="container"
elif [ -n "$virt" ] && [ "$virt" != "none" ]; then
    kind="virtual machine ($virt)"
else
    kind="bare metal or unknown"
fi

info "kind" "$kind"

cap_eff=$(awk '/^CapEff/ {print $2}' /proc/self/status)
sys_admin=$(((16#${cap_eff:-0} >> 21) & 1))

if [ "$(id -u)" = 0 ] && [ "$sys_admin" = 1 ]; then
    pass "root" "uid 0 with CAP_SYS_ADMIN"
elif [ "$(id -u)" = 0 ]; then
    warn "root" "uid 0 but no CAP_SYS_ADMIN, this is root in a container, settings that need admin may fail"
elif sudo -n true 2>/dev/null; then
    pass "root" "not root but sudo works"
else
    warn "root" "no root and no sudo"
fi

cpu_model=$(awk -F: '/model name/ {print $2; exit}' /proc/cpuinfo | sed 's/^ *//')
quota="none"

if [ -r /sys/fs/cgroup/cpu.max ]; then
    read -r max period </sys/fs/cgroup/cpu.max
    [ "$max" != "max" ] && quota=$(awk -v m="$max" -v p="$period" 'BEGIN {printf "%.1f cores", m/p}')
fi

info "cpu" "$cpu_model, $(nproc) visible, cgroup limit $quota"
info "memory" "$(free -g | awk '/^Mem:/ {print $2 " GiB total, " $7 " GiB available"}')"

# ------------------------------------------------------------------------ gpu
section "gpu"

if ! command -v nvidia-smi >/dev/null; then
    fail "nvidia-smi" "not found, there is no usable NVIDIA driver in this environment"
    say ""
    say "result: $n_pass passed, $n_warn warnings, $n_fail failed"
    exit 1
fi

nvidia-smi -q >"$out/nvidia-smi-q.txt" 2>&1
gpu_count=$(nvidia-smi -L | grep -c '^GPU')
name=$(q name)
info "gpu" "$name"

if [ "$gpu_count" = 1 ]; then pass "gpu count" "1 GPU visible"; else warn "gpu count" "$gpu_count GPUs visible, this test only looks at GPU 0"; fi

driver=$(q driver_version)
# the full report has the line in every driver version, the short header changes
cuda=$(grep -m1 'CUDA Version' "$out/nvidia-smi-q.txt" | awk -F: '{print $2}' | awk '{print $1}')
info "driver" "$driver, supports CUDA ${cuda:-unknown}"

if is_number "${driver%%.*}" && version_at_least "$driver" 610.43.02; then
    pass "driver for cuTile" "$driver is at least 610.43.02, which CUDA 13.3 needs"
else
    warn "driver for cuTile" "$driver is older than 610.43.02, cuTile Rust needs CUDA 13.3 and will not run"
fi

mig=$(q mig.mode.current)
if [ "$mig" = "Enabled" ]; then fail "whole gpu" "MIG is enabled, this is a slice of a GPU"; else pass "whole gpu" "no MIG (mode: $mig)"; fi

mode=$(q compute_mode)
case $mode in
    Prohibited) fail "compute mode" "Prohibited, no compute work can run" ;;
    *) pass "compute mode" "$mode" ;;
esac

apps=$(nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | grep -c .)
used=$(q memory.used)

if [ "${apps:-0}" -gt 0 ] || { is_number "$used" && [ "${used%.*}" -gt 500 ]; }; then
    warn "idle gpu" "$apps processes and $used MiB already in use, something else may be using this GPU"
else
    pass "idle gpu" "no other compute processes, $used MiB in use"
fi

memory_total=$(q memory.total)
info "memory" "$memory_total MiB total, ECC mode: $(q ecc.mode.current)"

# a stock RTX 4090 has 24 GB, a bigger one is a modified card with its own power and cooling
if [[ $name == *"RTX 4090"* ]] && is_number "$memory_total" && [ "${memory_total%.*}" -gt 30000 ]; then
    info "variant" "$memory_total MiB is not a stock RTX 4090 (24 GB), numbers will differ from a normal card"
fi
info "persistence" "$(q persistence_mode)"

limit=$(q power.limit)
default_limit=$(q power.default_limit)
max_limit=$(q power.max_limit)

if is_number "$limit" && is_number "$default_limit"; then
    if awk -v l="$limit" -v d="$default_limit" 'BEGIN {exit !(l < 0.95*d)}'; then
        warn "power limit" "$limit W is below the default $default_limit W (max $max_limit W), the card is capped"
    else
        pass "power limit" "$limit W, default $default_limit W, max $max_limit W"
    fi
else
    info "power limit" "not reported"
fi

pcie_idle="gen $(q pcie.link.gen.current) x$(q pcie.link.width.current), max gen $(q pcie.link.gen.max) x$(q pcie.link.width.max)"
info "pcie (idle)" "$pcie_idle, it often drops when idle, the load test below checks it again"

# ---------------------------------------------------------------- profiler access
section "profiler access"

restricted=$(awk '/RmProfilingAdminOnly/ {print $2}' /proc/driver/nvidia/params 2>/dev/null)

case $restricted in
    0) pass "counter policy" "RmProfilingAdminOnly is 0, any user may read the counters" ;;
    1) warn "counter policy" "RmProfilingAdminOnly is 1, only admins may read the counters" ;;
    *) info "counter policy" "cannot read /proc/driver/nvidia/params" ;;
esac

if command -v nvcc >/dev/null; then info "nvcc" "$(nvcc --version | tail -1)"; else warn "nvcc" "not installed, the kernel based tests are skipped"; fi
if command -v ncu >/dev/null; then info "ncu" "$(ncu --version | tail -1)"; else warn "ncu" "not installed, the counter test is skipped"; fi
if command -v nsys >/dev/null; then info "nsys" "$(nsys --version)"; else info "nsys" "not installed"; fi

# ------------------------------------------------------------------- clock lock
section "clock control"

if [ "$try_clock_lock" = 1 ]; then
    max_sm=$(q clocks.max.sm)

    if is_number "$max_sm"; then
        if result=$(nvidia-smi -i 0 -lgc "$max_sm,$max_sm" 2>&1); then
            pass "lock clocks" "nvidia-smi -lgc worked"

            if ! nvidia-smi -i 0 -rgc >/dev/null 2>&1; then
                warn "lock clocks" "the clocks were locked and could not be reset, run nvidia-smi -rgc"
            fi
        else
            warn "lock clocks" "cannot lock clocks: $(echo "$result" | head -1)"
        fi
    else
        info "lock clocks" "max sm clock not reported, skipped"
    fi
else
    info "lock clocks" "skipped with --no-clock-lock"
fi

# ---------------------------------------------------------------------- kernels
if [ "$run_kernels" = 1 ] && command -v nvcc >/dev/null; then
    cat >"$out/probe.cu" <<'CU'
// Small probes of the GPU, one number per line.
//   probe fma N     N timed runs of a compute bound kernel, GFLOPS per run
//   probe copy N    N timed runs of a memory bound kernel, GBps per run
//   probe burn S    the compute kernel for S seconds
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s failed: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

__global__ void fma_kernel(float *out, int iters, float a, float b) {
    float x0 = threadIdx.x, x1 = x0 + 1, x2 = x0 + 2, x3 = x0 + 3, x4 = x0 + 4, x5 = x0 + 5, x6 = x0 + 6, x7 = x0 + 7;
    for (int i = 0; i < iters; i++) {
        x0 = fmaf(x0, a, b); x1 = fmaf(x1, a, b); x2 = fmaf(x2, a, b); x3 = fmaf(x3, a, b);
        x4 = fmaf(x4, a, b); x5 = fmaf(x5, a, b); x6 = fmaf(x6, a, b); x7 = fmaf(x7, a, b);
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}

__global__ void copy_kernel(const float4 *src, float4 *dst, size_t n) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += stride) dst[i] = src[i];
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: probe fma|copy|burn N\n"); return 2; }
    int n = atoi(argv[2]);
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));

    const int iters = 2000000, threads = 256, blocks = prop.multiProcessorCount * 8;
    const size_t count = (size_t)blocks * threads;
    float *out;
    CHECK(cudaMalloc(&out, count * sizeof(float)));
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    const double flops = 2.0 * 8 * (double)iters * (double)count;

    if (!strcmp(argv[1], "fma") || !strcmp(argv[1], "burn")) {
        fma_kernel<<<blocks, threads>>>(out, iters, 1.0000001f, 0.5f);
        CHECK(cudaDeviceSynchronize());
        auto begin = std::chrono::steady_clock::now();

        for (int run = 0; ; run++) {
            if (!strcmp(argv[1], "fma")) { if (run >= n) break; }
            else if (std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count() >= n) break;

            CHECK(cudaEventRecord(start));
            fma_kernel<<<blocks, threads>>>(out, iters, 1.0000001f, 0.5f);
            CHECK(cudaEventRecord(stop));
            CHECK(cudaEventSynchronize(stop));
            float ms;
            CHECK(cudaEventElapsedTime(&ms, start, stop));
            if (!strcmp(argv[1], "fma")) printf("GFLOPS %.1f\n", flops / (ms * 1e6));
        }
    } else if (!strcmp(argv[1], "copy")) {
        const size_t bytes = (size_t)1 << 30, vectors = bytes / sizeof(float4);
        float4 *src, *dst;
        CHECK(cudaMalloc(&src, bytes));
        CHECK(cudaMalloc(&dst, bytes));
        CHECK(cudaMemset(src, 1, bytes));
        copy_kernel<<<blocks, threads>>>(src, dst, vectors);
        CHECK(cudaDeviceSynchronize());

        for (int run = 0; run < n; run++) {
            CHECK(cudaEventRecord(start));
            copy_kernel<<<blocks, threads>>>(src, dst, vectors);
            CHECK(cudaEventRecord(stop));
            CHECK(cudaEventSynchronize(stop));
            float ms;
            CHECK(cudaEventElapsedTime(&ms, start, stop));
            printf("GBps %.1f\n", 2.0 * bytes / (ms * 1e6));
        }
    } else {
        fprintf(stderr, "unknown mode %s\n", argv[1]);
        return 2;
    }

    return 0;
}
CU

    section "kernels"
    cc=$(q compute_cap | tr -d .)
    built=0

    for flags in "-arch=native" "-gencode arch=compute_$cc,code=sm_$cc"; do
        # shellcheck disable=SC2086
        if nvcc -O3 -std=c++14 $flags -o "$out/probe" "$out/probe.cu" >"$out/nvcc.log" 2>&1; then built=1; break; fi
    done

    if [ "$built" = 0 ]; then
        fail "build probes" "nvcc could not build for this GPU (sm_$cc), see $out/nvcc.log, an old toolkit does not know new GPUs"
    else
        pass "build probes" "built with nvcc"

        # the first run is a warm up and is not counted
        "$out/probe" copy 11 | awk '/GBps/ {print $2}' | tail -n +2 >"$out/copy.txt"
        "$out/probe" fma 11 | awk '/GFLOPS/ {print $2}' | tail -n +2 >"$out/fma.txt"

        read -r med min max spread <<<"$(stats <"$out/copy.txt")"

        if [ "${med:-n/a}" = "n/a" ]; then
            fail "memory copy" "the probe printed nothing"
        else
            expected=""
            case $name in
                *"RTX 4090"*) expected=1008 ;;
                *"RTX 5090"*) expected=1792 ;;
            esac

            info "memory copy" "median $med GB/s (min $min, max $max, spread $spread%)"

            if awk -v s="$spread" 'BEGIN {exit !(s > 3)}'; then warn "repeatability" "memory copy spread is $spread%, more than 3%"; else pass "repeatability" "memory copy spread $spread%"; fi

            if [ -n "$expected" ]; then
                if awk -v m="$med" -v e="$expected" 'BEGIN {exit !(m < 0.75*e)}'; then
                    warn "memory speed" "$med GB/s is under 75% of the $expected GB/s this card should reach"
                else
                    pass "memory speed" "$med GB/s against $expected GB/s peak"
                fi
            fi
        fi

        read -r med min max spread <<<"$(stats <"$out/fma.txt")"
        info "fp32 fma" "median $med GFLOP/s (min $min, max $max, spread $spread%)"
        if awk -v s="${spread:-0}" 'BEGIN {exit !(s > 3)}'; then warn "repeatability" "fp32 spread is $spread%, more than 3%"; else pass "repeatability" "fp32 spread $spread%"; fi

        # counter access, the profiler decides
        if command -v ncu >/dev/null; then
            timeout 180 ncu --metrics sm__cycles_elapsed.avg --target-processes all "$out/probe" fma 1 >"$out/ncu.log" 2>&1

            if grep -q 'ERR_NVGPUCTRPERM' "$out/ncu.log"; then
                fail "profiler counters" "ncu is not allowed to read the counters (ERR_NVGPUCTRPERM)"
            elif grep -q 'sm__cycles_elapsed' "$out/ncu.log"; then
                pass "profiler counters" "ncu read a hardware counter"
            else
                warn "profiler counters" "ncu did not give a result, see $out/ncu.log, last line: $(tail -1 "$out/ncu.log")"
            fi
        fi

        # sustained load with a sample every second
        section "sustained load ($burn_seconds s)"

        reasons=""
        prefix=clocks_throttle_reasons
        nvidia-smi -i 0 --query-gpu=clocks_throttle_reasons.active --format=csv,noheader >/dev/null 2>&1 || prefix=clocks_event_reasons

        for reason in sw_power_cap hw_slowdown hw_thermal_slowdown sw_thermal_slowdown hw_power_brake_slowdown; do
            if nvidia-smi -i 0 --query-gpu="$prefix.$reason" --format=csv,noheader >/dev/null 2>&1; then reasons="$reasons,$prefix.$reason"; fi
        done

        fields="clocks.sm,power.draw,temperature.gpu,pcie.link.gen.current,pcie.link.width.current$reasons"
        nvidia-smi -i 0 --query-gpu="$fields" --format=csv,noheader,nounits -l 1 >"$out/samples.csv" 2>/dev/null &
        sampler=$!
        trap 'kill $sampler 2>/dev/null' EXIT

        "$out/probe" burn "$burn_seconds"
        kill "$sampler" 2>/dev/null
        wait "$sampler" 2>/dev/null
        trap - EXIT

        # skip the first seconds, the clocks are still ramping up
        reason_names=$(echo "$reasons" | tr ',' '\n' | sed '/^$/d; s/.*\.//' | paste -sd' ')

        awk -F', *' -v names="$reason_names" 'BEGIN {nr=split(names, rn, " ")}
            NR > 5 {
                n++; c=$1+0; p=$2+0; t=$3+0; g=$4+0; w=$5+0
                if (n==1 || c<cmin) cmin=c; if (c>cmax) cmax=c; csum+=c
                if (p>pmax) pmax=p; psum+=p; if (t>tmax) tmax=t
                if (n==1 || g<gmin) gmin=g; if (n==1 || w<wmin) wmin=w
                for (i=1; i<=nr; i++) if ($(5+i) ~ /^Active/) active[rn[i]]++
            }
            END {
                if (!n) {print "samples=0"; exit}
                printf "samples=%d\nclock_min=%d\nclock_max=%d\nclock_mean=%.0f\npower_mean=%.0f\npower_max=%.0f\ntemp_max=%d\npcie_gen_min=%d\npcie_width_min=%d\n", n, cmin, cmax, csum/n, psum/n, pmax, tmax, gmin, wmin
                for (i=1; i<=nr; i++) printf "reason_%s=%d\n", rn[i], active[rn[i]]+0
            }' "$out/samples.csv" >"$out/load.txt"

        # shellcheck disable=SC1091
        . "$out/load.txt"

        if [ "${samples:-0}" -lt 5 ]; then
            warn "load samples" "only ${samples:-0} samples, the sampler did not work"
        else
            info "load" "$samples samples, sm clock $clock_min to $clock_max MHz (mean $clock_mean), power mean ${power_mean} W (max ${power_max} W), temperature max ${temp_max} C"

            if awk -v lo="$clock_min" -v hi="$clock_max" -v m="$clock_mean" 'BEGIN {exit !((hi-lo)/m > 0.10)}'; then
                warn "clock stability" "the sm clock moved by more than 10% during the load"
            else
                pass "clock stability" "the sm clock stayed within 10%"
            fi

            if [ "${temp_max:-0}" -gt 85 ]; then warn "temperature" "reached $temp_max C"; else pass "temperature" "reached $temp_max C"; fi

            slowdowns=0
            for reason in hw_slowdown hw_thermal_slowdown sw_thermal_slowdown hw_power_brake_slowdown; do
                count=$(eval "echo \${reason_$reason:-0}")
                if [ "$count" -gt 0 ]; then warn "throttling" "$reason was active in $count samples"; slowdowns=1; fi
            done
            [ "$slowdowns" = 0 ] && pass "throttling" "no thermal or hardware slowdown"

            cap=$(eval "echo \${reason_sw_power_cap:-0}")
            if [ "$cap" -gt 0 ]; then
                # a GPU at its limit reports the cap, one far below the limit has a lower limit somewhere else
                if is_number "$limit" && awk -v p="$power_mean" -v l="$limit" 'BEGIN {exit !(p < 0.9*l)}'; then
                    warn "power cap" "the cap was active in $cap of $samples samples at ${power_mean} W, well below the $limit W limit, the real limit is lower"
                else
                    info "power cap" "the cap was active in $cap of $samples samples, normal at full load"
                fi
            fi

            max_width=$(q pcie.link.width.max)
            if is_number "$max_width" && [ "$pcie_width_min" -lt "$max_width" ]; then
                warn "pcie (load)" "the link was x$pcie_width_min under load, the maximum is x$max_width"
            else
                pass "pcie (load)" "gen $pcie_gen_min x$pcie_width_min under load"
            fi
        fi
    fi
elif [ "$run_kernels" = 1 ]; then
    section "kernels"
    info "kernels" "skipped, nvcc is not installed"
fi

section "result"
say "$n_pass passed, $n_warn warnings, $n_fail failed"
say "details are in $out"
[ "$n_fail" = 0 ]
