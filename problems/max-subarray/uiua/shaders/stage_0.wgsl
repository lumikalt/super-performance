// Stage 0 of the shaders gpuiua makes from max_subarray.ua for 2^24 values in F32.
// cargo run --release -- --wgsl 24

@group(0) @binding(0) var<storage, read> in0: array<T>;
@group(0) @binding(1) var<storage, read_write> out0: array<T>;

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
var<workgroup> scratch: array<T, 64>;

// The operand is 1 x 16777216 x 1, row-major. Workgroup `w` handles slice
// `part` of the rows for one (batch, column) pair `j`.
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let w = wid.x + wid.y * 65535u;
    let j = w / 1024u;
    let part = w % 1024u;
    let jb = j / 1u;
    let jr = j % 1u;
    var acc: T = 0.0;
    // `j` is the same for the whole workgroup, so the barriers below stay uniform.
    if (j < 1u) {
        for (var k = part * 16384u + lid.x; k < min(part * 16384u + 16384u, 16777216u); k += 64u) {
            let idx = (jb * 16777216u + k) * 1u + jr;
    let v1: T = in0[idx];
            acc = (acc + v1);
        }
    }
    scratch[lid.x] = acc;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) {
        if (lid.x < s) {
            scratch[lid.x] = (scratch[lid.x] + scratch[lid.x + s]);
        }
        workgroupBarrier();
    }
    if (lid.x == 0u && j < 1u) {
        out0[(jb * 1024u + part) * 1u + jr] = scratch[0];
    }
}
