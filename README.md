# super-performance

Making algorithm challenges fast, one write-up at a time.
Each problem starts from the obvious solution and gets faster in measured steps.
The write-ups live on the author's website and walk through the steps of one branch.

## How the repository is laid out

`main` is the scaffolding.
It has the Nix shell, the Rust workspace and the shared `harness` crate and no problems.

Every problem lives on a branch of its own named after the problem, such as `max-subarray`.
The branch adds one crate under `problems/` and grows it one optimization at a time.
Each step is a commit and each finished step gets an annotated tag.

```
main                     scaffolding only
max-subarray             problems/max-subarray, one commit per step
  tag max-subarray/00-naive
  tag max-subarray/01-kadane
  ...
```

Tags are named `<problem>/<NN>-<slug>`.
The number keeps them in order and the slug is lowercase words joined by hyphens.
The first line of the tag message is the title of the step.
A write-up is built from these tags, so a step is never edited after tagging.
Fix a mistake with a new step instead.

New work on `main` is merged into the problem branches and never rebased, so the tags stay where they are.

## Running things

Enter the shell with `direnv allow` or `nix develop`.
The `.cargo/config.toml` fixes the code generation target to `x86-64-v3` (AVX2, BMI2 and FMA).

```
cargo test -p max-subarray
cargo bench -p max-subarray
```

Every step keeps the variants of the earlier steps, so one benchmark run at the tip of the branch compares them all in the same session.
Every variant is tested against the naive version.

## Measuring well

A benchmark is only as good as its conditions.

- Close other programs and keep the machine plugged in.
- Use the `performance` governor.
- Pin to one core so the scheduler cannot move the run between fast and slow cores.
  On a hybrid chip look at `/sys/devices/cpu_core/cpus` and `/sys/devices/cpu_atom/cpus` first.
  `taskset -c 0 cargo bench ...` runs on the first logical processor.
- Compare medians and look at the spread, never a single run.
- State the machine, see `harness::environment::describe`.
