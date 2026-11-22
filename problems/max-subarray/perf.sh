#!/usr/bin/env bash
# Cycles, instructions and IPC per value for some variants, on one core.
#
# usage: perf.sh <values> <repetitions> <variant>...
#   CORE=0 sets the core to pin to, a fast one on a hybrid chip.
#   RANGE="-3000 1000" sets the lowest and highest value of the random input.
#
# Needs `perf`, for example `nix shell nixpkgs#linuxPackages_latest.perf`.
# Only user-space events are counted, so perf_event_paranoid 2 is enough.
set -eu

if [ $# -lt 3 ]; then
    echo "usage: perf.sh <values> <repetitions> <variant>..." >&2
    exit 2
fi

values=$1
reps=$2
shift 2
core=${CORE:-0}
root=$(cd "$(dirname "$0")/../.." && pwd)

cargo build --release -p max-subarray --example run >/dev/null

# A hybrid chip has separate counters for its fast cores.
if [ -d /sys/devices/cpu_core ]; then
    events=cpu_core/cycles/u,cpu_core/instructions/u
else
    events=cycles:u,instructions:u
fi

for variant in "$@"; do
    out=$(taskset -c "$core" perf stat -x, -e "$events" -- "$root/target/release/examples/run" "$variant" "$values" "$reps" ${RANGE:-} 2>&1 >/dev/null)
    cycles=$(echo "$out" | grep cycles | cut -d, -f1)
    instructions=$(echo "$out" | grep instructions | cut -d, -f1)

    awk -v name="$variant" -v n="$((values * reps))" -v c="$cycles" -v i="$instructions" \
        'BEGIN { printf "%-18s %6.2f cycles/value %6.2f instructions/value  IPC %.2f\n", name, c / n, i / n, i / c }'
done
