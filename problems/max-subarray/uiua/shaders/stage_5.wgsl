// Stage 5 of the shaders gpuiua makes from max_subarray.ua for 2^24 values in F32.
// cargo run --release -- --wgsl 24

@group(0) @binding(0) var<storage, read> in0: array<T>;
@group(0) @binding(1) var<storage, read> in1: array<T>;
@group(0) @binding(2) var<storage, read_write> out0: array<T>;

alias T = f32;
const BIG: T = 3.4028235e38;
// Floored modulo, unlike WGSL's truncated `%`. GPU division isn't always
// correctly rounded, so `floor(b / a)` can be one off for exact multiples;
// the result is brought back into range.
fn umod(b: T, a: T) -> T {
    var r = b - a * floor(b / a);
    if (r * a < T(0.0)) {
        r += a;
    } else if (abs(r) >= abs(a)) {
        r -= a;
    }
    return r;
}

// Division, corrected to be as exact as an IEEE one: GPUs may divide with a
// reciprocal and so miss by a few ulps, which turns `6 / 3` into 1.9999999 and
// `⌊÷` into the wrong integer. One step with the exact remainder fixes it.
fn divx(b: T, a: T) -> T {
    let q = b / a;
    let r = fma(-q, a, b);
    return select(q, q + r / a, (r == r) && (a != T(0.0)) && (abs(q) < T(1.0e30)));
}
var<workgroup> tile: array<T, 2048>;
var<workgroup> sums: array<T, 64>;
const PER: u32 = 32u;

// The operand is 1 x 16777217 x 1, row-major. Workgroup `w` scans block
// `part` (16385 rows) of one (batch, column) pair `j`.
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let t = lid.x;
    let w = wid.x + wid.y * 65535u;
    let j = w / 1024u;
    let part = w % 1024u;
    let jb = j / 1u;
    let jr = j % 1u;
    // `j` is uniform across the workgroup; out-of-range ones just get empty blocks.
    let first = select(16777217u, min(part * 16385u, 16777217u), j < 1u);
    let count = min(first + 16385u, 16777217u) - first;
    var carry: T = BIG;
    if (part > 0u && j < 1u) {
        carry = min(in1[(jb * 1024u + part - 1u) * 1u + jr], carry);
    }
    for (var base = 0u; base < count; base += 2048u) {
        for (var i = 0u; i < PER; i++) {
            let at = i * 64u + t;
            var x: T = BIG;
            if (base + at < count) {
                let idx = (jb * 16777217u + first + base + at) * 1u + jr;
    let v1q: u32 = idx / 1u;
    let v1r: u32 = idx % 1u;
    let v1k: u32 = v1q % 16777217u;
    let v1b: u32 = v1q / 16777217u;
    var v2: T;
    if (v1k < 1u) {
        v2 = 0.0;
    } else {
    let v3: T = in0[((v1b * 16777216u + (v1k - 1u)) * 1u + v1r)];
        v2 = v3;
    }
                x = v2;
            }
            tile[at] = x;
        }
        workgroupBarrier();
        // Each thread scans its own run of the tile.
        var run: T = BIG;
        for (var i = 0u; i < PER; i++) {
            run = min(run, tile[t * PER + i]);
            tile[t * PER + i] = run;
        }
        sums[t] = run;
        workgroupBarrier();
        for (var off = 1u; off < 64u; off = off << 1u) {
            var v = sums[t];
            if (t >= off) {
                v = min(sums[t - off], v);
            }
            workgroupBarrier();
            sums[t] = v;
            workgroupBarrier();
        }
        var before: T = carry;
        if (t > 0u) {
            before = min(carry, sums[t - 1u]);
        }
        for (var i = 0u; i < PER; i++) {
            tile[t * PER + i] = min(before, tile[t * PER + i]);
        }
        let total = min(carry, sums[63u]);
        workgroupBarrier();
        for (var i = 0u; i < PER; i++) {
            let at = i * 64u + t;
            if (base + at < count) {
                let idx = (jb * 16777217u + first + base + at) * 1u + jr;
                out0[idx] = tile[at];
            }
        }
        carry = total;
        workgroupBarrier();
    }
}
