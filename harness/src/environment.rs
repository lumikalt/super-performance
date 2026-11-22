use std::fs;

/// One line about the machine, for the top of a benchmark report.
/// It reads Linux files and falls back to "unknown" elsewhere.
pub fn describe() -> String {
    let cpuinfo = fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let model = cpuinfo
        .lines()
        .find(|line| line.starts_with("model name"))
        .and_then(|line| line.split(':').nth(1))
        .map(str::trim)
        .unwrap_or("unknown cpu");
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);
    let governor =
        read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").unwrap_or_else(|| "unknown".to_string());

    // Hybrid chips have fast and efficient cores, a benchmark should say which it ran on.
    let cores = match (read("/sys/devices/cpu_core/cpus"), read("/sys/devices/cpu_atom/cpus")) {
        (Some(p), Some(e)) => format!(", performance cores {p}, efficiency cores {e}"),
        _ => String::new(),
    };

    format!("{model}, {threads} threads{cores}, governor {governor}")
}

fn read(path: &str) -> Option<String> {
    fs::read_to_string(path).ok().map(|text| text.trim().to_string())
}
