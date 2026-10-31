#![no_std]
#![feature(abi_ptx, asm_experimental_arch)]

use core::arch::{asm, global_asm};

global_asm!(".extern .shared .align 16 .b8 attention_shared[];");

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    unsafe {
        asm!("trap;", options(noreturn));
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Attention {
    q: *const u16,
    k: *const u16,
    v: *const u16,
    out: *mut u16,
    lse: *mut f32,
    rows: u32,
    capacity: u32,
    q_stride: u32,
    q_heads: u32,
    group: u32,
    start: u32,
    position: *const u32,
}

#[inline(always)]
unsafe fn resolve_position(mut p: Attention) -> Attention {
    if !p.position.is_null() {
        unsafe { p.start = core::cmp::min(*p.position, p.start) };
    }
    p
}

#[inline(always)]
fn address(base: u32, row: u32, col: u32, stride: u32) -> u32 {
    base + 2 * (row * stride + (col ^ ((row & 7) << 3)))
}

#[inline(always)]
unsafe fn copy(dst: u32, src: *const u16, valid: bool) {
    unsafe {
        asm!("cp.async.cg.shared.global [{d}], [{s}], 16, {n};",
            d = in(reg32) dst, s = in(reg64) src, n = in(reg32) if valid { 16u32 } else { 0u32 });
    }
}

#[inline(always)]
unsafe fn commit() {
    unsafe {
        asm!("cp.async.commit_group;");
    }
}
#[inline(always)]
unsafe fn wait() {
    unsafe {
        asm!("cp.async.wait_group 0; bar.sync 0;");
    }
}
#[inline(always)]
unsafe fn barrier() {
    unsafe {
        asm!("bar.sync 0;");
    }
}

#[inline(always)]
unsafe fn ld4<const OFFSET: u32>(p: u32) -> [u32; 4] {
    let (a, b, c, d);
    unsafe {
        asm!("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{a},{b},{c},{d}}}, [{p}+{offset}];",
            a=out(reg32) a,b=out(reg32) b,c=out(reg32) c,d=out(reg32) d,p=in(reg32) p,offset=const OFFSET);
    }
    [a, b, c, d]
}

#[inline(always)]
unsafe fn ld4_trans<const OFFSET: u32>(p: u32) -> [u32; 4] {
    let (a, b, c, d);
    unsafe {
        asm!("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{{a},{b},{c},{d}}}, [{p}+{offset}];",
            a=out(reg32) a,b=out(reg32) b,c=out(reg32) c,d=out(reg32) d,p=in(reg32) p,offset=const OFFSET);
    }
    [a, b, c, d]
}

#[inline(always)]
unsafe fn mma(a: [u32; 4], b: [u32; 2], mut c: [f32; 4]) -> [f32; 4] {
    unsafe {
        asm!("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {{{c0},{c1},{c2},{c3}}}, {{{a0},{a1},{a2},{a3}}}, {{{b0},{b1}}}, {{{c0},{c1},{c2},{c3}}};",
            c0=inout(reg32) c[0],c1=inout(reg32) c[1],c2=inout(reg32) c[2],c3=inout(reg32) c[3],
            a0=in(reg32) a[0],a1=in(reg32) a[1],a2=in(reg32) a[2],a3=in(reg32) a[3],b0=in(reg32) b[0],b1=in(reg32) b[1]);
    }
    c
}

#[inline(always)]
unsafe fn maximum(a: f32, b: f32) -> f32 {
    let x;
    unsafe {
        asm!("max.f32 {x}, {a}, {b};",x=out(reg32) x,a=in(reg32) a,b=in(reg32) b);
    }
    x
}
#[inline(always)]
unsafe fn shuffle(a: f32, mask: u32) -> f32 {
    let x: u32;
    unsafe {
        asm!("shfl.sync.bfly.b32 {x}, {a}, {m}, 31, 0xffffffff;", x=out(reg32) x,a=in(reg32) a.to_bits(),m=in(reg32) mask);
    }
    f32::from_bits(x)
}
#[inline(always)]
unsafe fn probability<const D: usize>(a: f32, negative_peak: f32, scale: f32) -> f32 {
    let x;
    unsafe {
        asm!("fma.rn.ftz.f32 {x}, {a}, {scale}, {peak}; ex2.approx.ftz.f32 {x}, {x};",
            x=out(reg32) x,a=in(reg32) a,scale=in(reg32) (core::f32::consts::LOG2_E * scale),peak=in(reg32) negative_peak);
    }
    x
}
#[inline(always)]
unsafe fn logarithm(a: f32) -> f32 {
    let bits = a.to_bits();
    let mut exponent = ((bits >> 23) & 0xff) as i32 - 127;
    let mut mantissa = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
    if mantissa > core::f32::consts::SQRT_2 {
        mantissa *= 0.5;
        exponent += 1;
    }
    let t = (mantissa - 1.0) / (mantissa + 1.0);
    let square = t * t;
    let series = 1.0
        + square
            * (1.0 / 3.0
                + square
                    * (1.0 / 5.0
                        + square * (1.0 / 7.0 + square * (1.0 / 9.0 + square * (1.0 / 11.0)))));
    2.0 * t * series + exponent as f32 * core::f32::consts::LN_2
}
#[inline(always)]
unsafe fn reciprocal(a: f32) -> f32 {
    let x;
    unsafe {
        asm!("rcp.approx.ftz.f32 {x}, {a};",x=out(reg32) x,a=in(reg32) a);
    }
    x
}
#[inline(always)]
unsafe fn pack(a: f32, b: f32) -> u32 {
    let x;
    unsafe {
        asm!("cvt.rn.bf16x2.f32 {x}, {b}, {a};",x=out(reg32) x,a=in(reg32) a,b=in(reg32) b);
    }
    x
}
#[inline(always)]
unsafe fn pack_residual(a: f32, b: f32, high: u32) -> u32 {
    let ah: f32;
    let bh: f32;
    unsafe {
        asm!("{{ .reg .b16 lo, hi; mov.b32 {{lo, hi}}, {high}; cvt.f32.bf16 {ah}, lo; cvt.f32.bf16 {bh}, hi; }}",
            high=in(reg32) high,ah=out(reg32) ah,bh=out(reg32) bh);
        pack(a - ah, b - bh)
    }
}
#[inline(always)]
unsafe fn store_shared4(p: u32, x: [u32; 4]) {
    unsafe {
        asm!("st.shared.v4.b32 [{p}], {{{a},{b},{c},{d}}};",
            p=in(reg32) p,a=in(reg32) x[0],b=in(reg32) x[1],c=in(reg32) x[2],d=in(reg32) x[3]);
    }
}
#[inline(always)]
unsafe fn load_shared(p: u32) -> u32 {
    let x;
    unsafe {
        asm!("ld.shared.u32 {x}, [{p}];",x=out(reg32) x,p=in(reg32) p);
    }
    x
}
#[inline(always)]
unsafe fn store_global4(p: *mut u16, x: [u32; 4]) {
    unsafe {
        asm!("st.global.v4.b32 [{p}], {{{a},{b},{c},{d}}};",
            p=in(reg64) p,a=in(reg32) x[0],b=in(reg32) x[1],c=in(reg32) x[2],d=in(reg32) x[3]);
    }
}
macro_rules! four {
    ($m:ident) => {
        $m!(0);
        $m!(1);
        $m!(2);
        $m!(3);
    };
}
macro_rules! eight {
    ($m:ident) => {
        $m!(0);
        $m!(1);
        $m!(2);
        $m!(3);
        $m!(4);
        $m!(5);
        $m!(6);
        $m!(7);
    };
}
macro_rules! sixteen {
    ($m:ident) => {
        eight!($m);
        $m!(8);
        $m!(9);
        $m!(10);
        $m!(11);
        $m!(12);
        $m!(13);
        $m!(14);
        $m!(15);
    };
}
macro_rules! thirty_two {
    ($m:ident) => {
        sixteen!($m);
        $m!(16);
        $m!(17);
        $m!(18);
        $m!(19);
        $m!(20);
        $m!(21);
        $m!(22);
        $m!(23);
        $m!(24);
        $m!(25);
        $m!(26);
        $m!(27);
        $m!(28);
        $m!(29);
        $m!(30);
        $m!(31);
    };
}

#[inline(always)]
unsafe fn copy_kv<const D: usize, const MASK: bool>(
    dst: [u32; 2],
    src: *const u16,
    row: u32,
    rows: u32,
) {
    unsafe {
        macro_rules! transfer {
            ($i:expr) => {
                asm!("cp.async.cg.shared.global [{dst}+{shared_offset}], [{src}+{global_offset}], 16, {bytes};",
                    dst=in(reg32) dst[$i % 2],src=in(reg64) src,
                    shared_offset=const ($i / 2) * 4096,global_offset=const $i * 2048,
                    bytes=in(reg32) if !MASK || row + $i * (1024 / D as u32) < rows {16u32} else {0u32});
            };
        }
        if D == 256 {
            sixteen!(transfer);
        } else if D == 64 {
            four!(transfer);
        } else {
            eight!(transfer);
        }
    }
}

/// # Safety
/// Launch uses 128 threads, 98304 shared bytes, disjoint BF16 Q/K/V/output and F32 LSE buffers covering the declared extents.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn attention_prefill_d256(p: Attention) {
    unsafe {
        let p = resolve_position(p);
        attention::<256, 32, 0>(p, p.start + p.rows, 0, 0.0625);
    }
}

/// # Safety
/// Launch uses 128 threads, 49152 shared bytes and buffers covering the declared extents.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn attention_prefill_d128(p: Attention) {
    unsafe {
        let p = resolve_position(p);
        attention::<128, 16, 0>(
            p,
            p.start + p.rows,
            0,
            core::f32::consts::FRAC_1_SQRT_2 / 8.0,
        );
    }
}

/// # Safety
/// All keys below `keys` are populated in this sequence; launch uses 128 threads and 98304 shared bytes.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn attention_prefill_d256_bidirectional(p: Attention, keys: u32) {
    unsafe {
        let p = resolve_position(p);
        attention::<256, 32, 1>(p, keys, 0, 0.0625);
    }
}

/// # Safety
/// All keys below `keys` are populated in this sequence; launch uses 128 threads and 49152 shared bytes.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn attention_prefill_d128_bidirectional(p: Attention, keys: u32) {
    unsafe {
        let p = resolve_position(p);
        attention::<128, 16, 1>(p, keys, 0, core::f32::consts::FRAC_1_SQRT_2 / 8.0);
    }
}

/// # Safety
/// All keys below `keys` are populated; launch uses 128 threads and 24576 shared bytes.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn attention_prefill_d64_bidirectional(
    p: Attention,
    keys: u32,
    scale: f32,
) {
    unsafe {
        let p = resolve_position(p);
        attention::<64, 8, 1>(p, keys, 0, scale);
    }
}

/// # Safety
/// All prefix keys are populated; launch uses 128 threads and 49152 shared bytes.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn attention_prefill_d128_prefix(p: Attention, prefix: u32) {
    unsafe {
        let p = resolve_position(p);
        attention::<128, 16, 2>(
            p,
            prefix.max(p.start + p.rows),
            prefix,
            core::f32::consts::FRAC_1_SQRT_2 / 8.0,
        );
    }
}

/// # Safety
/// Launch uses 128 threads and 98304 shared bytes; window is an inclusive left distance.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn attention_prefill_d256_window(
    p: Attention,
    window: u32,
    scale: f32,
) {
    unsafe {
        let p = resolve_position(p);
        attention::<256, 32, 3>(p, p.start + p.rows, window, scale);
    }
}

/// # Safety
/// Window is an inclusive left distance; launch uses 128 threads and 49152 shared bytes.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn attention_prefill_d128_window(p: Attention, window: u32) {
    unsafe {
        let p = resolve_position(p);
        attention::<128, 16, 3>(
            p,
            p.start + p.rows,
            window,
            core::f32::consts::FRAC_1_SQRT_2 / 8.0,
        );
    }
}

#[inline(always)]
unsafe fn attention<const D: usize, const OUTPUT: usize, const MODE: u32>(
    p: Attention,
    end: u32,
    prefix: u32,
    scale: f32,
) {
    unsafe {
        let (tid, tile, head, sm): (u32, u32, u32, u32);
        asm!("mov.u32 {t}, %tid.x; mov.u32 {b}, %ctaid.x; mov.u32 {h}, %ctaid.y; mov.u32 {s}, attention_shared;",
            t=out(reg32) tid,b=out(reg32) tile,h=out(reg32) head,s=out(reg32) sm);
        let lane = tid & 31;
        let warp = tid >> 5;
        let qstart = tile * 64;
        let diagonal = p.start + qstart;
        let sk = sm + (128 * D) as u32;
        let sv = sm + (256 * D) as u32;
        let stop = if MODE == 1 {
            end
        } else if MODE == 2 {
            end.min(prefix.max(diagonal + 64))
        } else {
            end.min(diagonal + 64)
        };
        let interior = if MODE == 1 {
            end
        } else if MODE == 2 {
            end.min(prefix.max(diagonal))
        } else {
            end.min(diagonal)
        };
        let copy_row = tid / (D as u32 / 8);
        let copy_lane = tid % (D as u32 / 8);
        let copy_col = copy_lane * 8;
        let copy_shared = [
            address(0, copy_row, copy_col, D as u32),
            address(0, copy_row + 1024 / D as u32, copy_col, D as u32),
        ];
        let kv = (head / p.group) as usize * p.capacity as usize * D;
        let first = if MODE == 3 {
            diagonal.saturating_sub(prefix) / 64 * 64
        } else {
            0
        };
        let mut kp =
            p.k.wrapping_add(kv + (first + copy_row) as usize * D + copy_col as usize);
        let mut vp =
            p.v.wrapping_add(kv + (first + copy_row) as usize * D + copy_col as usize);
        macro_rules! copy_q {
            ($i:expr) => {{
                let row = copy_row + $i * (1024 / D as u32);
                copy(
                    sm + copy_shared[$i % 2] + ($i / 2) * 4096,
                    p.q.wrapping_add(
                        (qstart + row) as usize * p.q_stride as usize
                            + head as usize * D
                            + copy_col as usize,
                    ),
                    qstart + row < p.rows,
                );
            }};
        }
        if D == 256 {
            sixteen!(copy_q);
        } else if D == 64 {
            four!(copy_q);
        } else {
            eight!(copy_q);
        }
        copy_kv::<D, true>(
            [sk + copy_shared[0], sk + copy_shared[1]],
            kp,
            first + copy_row,
            end,
        );
        commit();
        wait();
        let qr = warp * 16 + (lane & 15);
        let qc = (lane >> 4) * 8;
        let kr = (lane & 7) + (lane >> 4) * 8;
        let kc = ((lane >> 3) & 1) * 8;
        let vr = lane & 15;
        let vc = (lane >> 4) * 8;
        let qb = [
            address(sm, qr, qc, D as u32),
            address(sm, qr, qc + 16, D as u32),
            address(sm, qr, qc + 32, D as u32),
            address(sm, qr, qc + 48, D as u32),
        ];
        let kb = [
            address(sk, kr, kc, D as u32),
            address(sk, kr, kc + 16, D as u32),
            address(sk, kr, kc + 32, D as u32),
            address(sk, kr, kc + 48, D as u32),
        ];
        let vb = [
            address(sv, vr, vc, D as u32),
            address(sv, vr, vc + 16, D as u32),
            address(sv, vr, vc + 32, D as u32),
            address(sv, vr, vc + 48, D as u32),
        ];
        let mut output = [[0.0f32; 4]; OUTPUT];
        let mut peak = [f32::NEG_INFINITY; 2];
        let mut denominator = [0.0f32; 2];
        let r0 = warp * 16 + (lane >> 2);
        let r1 = r0 + 8;
        let mut key = first;
        macro_rules! tile_step {
            ($masked:expr) => {{
                copy_kv::<D, $masked>(
                    [sv + copy_shared[0], sv + copy_shared[1]],
                    vp,
                    key + copy_row,
                    end,
                );
                commit();
                let mut scores = [[0.0f32; 4]; 8];
                macro_rules! qk_step {
                    ($inner:expr) => {{
                        let a = ld4::<{ ($inner / 4) * 128 }>(qb[$inner % 4]);
                        macro_rules! qk_pair {
                            ($pair:expr) => {{
                                let b = if D == 256 {
                                    ld4::<{ ($inner / 4) * 128 + $pair * 8192 }>(kb[$inner % 4])
                                } else if D == 64 {
                                    ld4::<{ $pair * 2048 }>(kb[$inner % 4])
                                } else {
                                    ld4::<{ ($inner / 4) * 128 + $pair * 4096 }>(kb[$inner % 4])
                                };
                                scores[$pair * 2] = mma(a, [b[0], b[1]], scores[$pair * 2]);
                                scores[$pair * 2 + 1] = mma(a, [b[2], b[3]], scores[$pair * 2 + 1]);
                            }};
                        }
                        four!(qk_pair);
                    }};
                }
                if D == 256 {
                    sixteen!(qk_step);
                } else if D == 64 {
                    four!(qk_step);
                } else {
                    eight!(qk_step);
                }
                wait();
                let next_key = key + 64;
                kp = kp.wrapping_add(64 * D);
                if next_key < stop {
                    let destinations = [sk + copy_shared[0], sk + copy_shared[1]];
                    if next_key + 64 <= end {
                        copy_kv::<D, false>(destinations, kp, next_key + copy_row, end);
                    } else {
                        copy_kv::<D, true>(destinations, kp, next_key + copy_row, end);
                    }
                    commit();
                }
                let mut next = peak;
                macro_rules! mask_max {
                    ($n:expr) => {{
                        if $masked || MODE == 3 {
                            let col = key + $n * 8 + (lane & 3) * 2;
                            if col
                                >= if MODE == 1 {
                                    end
                                } else if MODE == 2 {
                                    prefix.max(diagonal + r0 + 1).min(end)
                                } else {
                                    (diagonal + r0 + 1).min(end)
                                }
                            {
                                scores[$n][0] = f32::NEG_INFINITY;
                            }
                            if col + 1
                                >= if MODE == 1 {
                                    end
                                } else if MODE == 2 {
                                    prefix.max(diagonal + r0 + 1).min(end)
                                } else {
                                    (diagonal + r0 + 1).min(end)
                                }
                            {
                                scores[$n][1] = f32::NEG_INFINITY;
                            }
                            if col
                                >= if MODE == 1 {
                                    end
                                } else if MODE == 2 {
                                    prefix.max(diagonal + r1 + 1).min(end)
                                } else {
                                    (diagonal + r1 + 1).min(end)
                                }
                            {
                                scores[$n][2] = f32::NEG_INFINITY;
                            }
                            if col + 1
                                >= if MODE == 1 {
                                    end
                                } else if MODE == 2 {
                                    prefix.max(diagonal + r1 + 1).min(end)
                                } else {
                                    (diagonal + r1 + 1).min(end)
                                }
                            {
                                scores[$n][3] = f32::NEG_INFINITY;
                            }
                        }
                        if MODE == 3 {
                            let col = key + $n * 8 + (lane & 3) * 2;
                            let lo0 = (diagonal + r0).saturating_sub(prefix);
                            let lo1 = (diagonal + r1).saturating_sub(prefix);
                            if col < lo0 {
                                scores[$n][0] = f32::NEG_INFINITY;
                            }
                            if col + 1 < lo0 {
                                scores[$n][1] = f32::NEG_INFINITY;
                            }
                            if col < lo1 {
                                scores[$n][2] = f32::NEG_INFINITY;
                            }
                            if col + 1 < lo1 {
                                scores[$n][3] = f32::NEG_INFINITY;
                            }
                        }
                        next[0] = maximum(next[0], maximum(scores[$n][0], scores[$n][1]));
                        next[1] = maximum(next[1], maximum(scores[$n][2], scores[$n][3]));
                    }};
                }
                eight!(mask_max);
                next[0] = maximum(next[0], shuffle(next[0], 1));
                next[0] = maximum(next[0], shuffle(next[0], 2));
                next[1] = maximum(next[1], shuffle(next[1], 1));
                next[1] = maximum(next[1], shuffle(next[1], 2));
                let mut negative_peak = [
                    -next[0] * (core::f32::consts::LOG2_E * scale),
                    -next[1] * (core::f32::consts::LOG2_E * scale),
                ];
                let mut alpha = [
                    probability::<{ D }>(peak[0] - next[0], 0.0, scale),
                    probability::<{ D }>(peak[1] - next[1], 0.0, scale),
                ];
                if MODE == 3 {
                    for row in 0..2 {
                        if next[row] == f32::NEG_INFINITY {
                            negative_peak[row] = 0.0;
                            alpha[row] = 0.0;
                        }
                    }
                }
                let mut sum = [0.0f32; 2];
                let mut packed = [[0u32; 4]; 4];
                let mut residual = [[0u32; 4]; 4];
                macro_rules! probabilities {
                    ($n:expr) => {{
                        let a = probability::<{ D }>(scores[$n][0], negative_peak[0], scale);
                        let b = probability::<{ D }>(scores[$n][1], negative_peak[0], scale);
                        let c = probability::<{ D }>(scores[$n][2], negative_peak[1], scale);
                        let d = probability::<{ D }>(scores[$n][3], negative_peak[1], scale);
                        sum[0] += a + b;
                        sum[1] += c + d;
                        packed[$n / 2][($n % 2) * 2] = pack(a, b);
                        packed[$n / 2][($n % 2) * 2 + 1] = pack(c, d);
                        if D <= 128 {
                            residual[$n / 2][($n % 2) * 2] =
                                pack_residual(a, b, packed[$n / 2][($n % 2) * 2]);
                            residual[$n / 2][($n % 2) * 2 + 1] =
                                pack_residual(c, d, packed[$n / 2][($n % 2) * 2 + 1]);
                        }
                    }};
                }
                eight!(probabilities);
                denominator[0] = denominator[0] * alpha[0] + sum[0];
                denominator[1] = denominator[1] * alpha[1] + sum[1];
                peak = next;
                macro_rules! rescale {
                    ($n:expr) => {{
                        output[$n][0] *= alpha[0];
                        output[$n][1] *= alpha[0];
                        output[$n][2] *= alpha[1];
                        output[$n][3] *= alpha[1];
                    }};
                }
                if D == 256 {
                    thirty_two!(rescale);
                } else if D == 64 {
                    eight!(rescale);
                } else {
                    sixteen!(rescale);
                }
                macro_rules! pv_step {
                    ($inner:expr) => {{
                        let a = packed[$inner];
                        macro_rules! pv_pair {
                            ($pair:expr) => {{
                                let b = if D == 256 {
                                    ld4_trans::<{ ($pair / 4) * 128 + $inner * 8192 }>(
                                        vb[$pair % 4],
                                    )
                                } else if D == 64 {
                                    ld4_trans::<{ $inner * 2048 }>(vb[$pair % 4])
                                } else {
                                    ld4_trans::<{ ($pair / 4) * 128 + $inner * 4096 }>(
                                        vb[$pair % 4],
                                    )
                                };
                                output[$pair * 2] = mma(a, [b[0], b[1]], output[$pair * 2]);
                                output[$pair * 2 + 1] = mma(a, [b[2], b[3]], output[$pair * 2 + 1]);
                                if D <= 128 {
                                    output[$pair * 2] =
                                        mma(residual[$inner], [b[0], b[1]], output[$pair * 2]);
                                    output[$pair * 2 + 1] =
                                        mma(residual[$inner], [b[2], b[3]], output[$pair * 2 + 1]);
                                }
                            }};
                        }
                        if D == 256 {
                            sixteen!(pv_pair);
                        } else if D == 64 {
                            four!(pv_pair);
                        } else {
                            eight!(pv_pair);
                        }
                    }};
                }
                four!(pv_step);
                wait();
            }};
        }
        while key + 64 <= interior {
            tile_step!(false);
            key += 64;
            vp = vp.wrapping_add(64 * D);
        }
        while key < stop {
            tile_step!(true);
            key += 64;
            vp = vp.wrapping_add(64 * D);
        }
        let (tid, tile, head, sm): (u32, u32, u32, u32);
        asm!("mov.u32 {t}, %tid.x; mov.u32 {b}, %ctaid.x; mov.u32 {h}, %ctaid.y; mov.u32 {s}, attention_shared;",
            t=out(reg32) tid,b=out(reg32) tile,h=out(reg32) head,s=out(reg32) sm);
        let lane = tid & 31;
        let warp = tid >> 5;
        let qstart = tile * 64;
        let copy_row = tid / (D as u32 / 8);
        let copy_lane = tid % (D as u32 / 8);
        let copy_col = copy_lane * 8;
        let r0 = warp * 16 + (lane >> 2);
        let r1 = r0 + 8;
        denominator[0] += shuffle(denominator[0], 1);
        denominator[0] += shuffle(denominator[0], 2);
        denominator[1] += shuffle(denominator[1], 1);
        denominator[1] += shuffle(denominator[1], 2);
        let norm = [reciprocal(denominator[0]), reciprocal(denominator[1])];
        if lane & 3 == 0 {
            if qstart + r0 < p.rows {
                p.lse
                    .add((head * p.rows + qstart + r0) as usize)
                    .write(peak[0] * scale + logarithm(denominator[0]));
            }
            if qstart + r1 < p.rows {
                p.lse
                    .add((head * p.rows + qstart + r1) as usize)
                    .write(peak[1] * scale + logarithm(denominator[1]));
            }
        }
        macro_rules! stage_output {
            ($group:expr) => {{
                let base = sm + warp * (32 * D) as u32 + $group * 1024 + (lane ^ $group) * 16;
                store_shared4(
                    base,
                    [
                        pack(
                            output[$group * 4][0] * norm[0],
                            output[$group * 4][1] * norm[0],
                        ),
                        pack(
                            output[$group * 4 + 1][0] * norm[0],
                            output[$group * 4 + 1][1] * norm[0],
                        ),
                        pack(
                            output[$group * 4 + 2][0] * norm[0],
                            output[$group * 4 + 2][1] * norm[0],
                        ),
                        pack(
                            output[$group * 4 + 3][0] * norm[0],
                            output[$group * 4 + 3][1] * norm[0],
                        ),
                    ],
                );
                store_shared4(
                    base + 512,
                    [
                        pack(
                            output[$group * 4][2] * norm[1],
                            output[$group * 4][3] * norm[1],
                        ),
                        pack(
                            output[$group * 4 + 1][2] * norm[1],
                            output[$group * 4 + 1][3] * norm[1],
                        ),
                        pack(
                            output[$group * 4 + 2][2] * norm[1],
                            output[$group * 4 + 2][3] * norm[1],
                        ),
                        pack(
                            output[$group * 4 + 3][2] * norm[1],
                            output[$group * 4 + 3][3] * norm[1],
                        ),
                    ],
                );
            }};
        }
        if D == 256 {
            eight!(stage_output);
        } else if D == 64 {
            stage_output!(0);
            stage_output!(1);
        } else {
            four!(stage_output);
        }
        barrier();
        macro_rules! write_output {
            ($step:expr) => {{
                let row = copy_row + $step * (1024 / D as u32);
                if qstart + row < p.rows {
                    let base = sm
                        + (row / 16) * (32 * D) as u32
                        + (copy_lane / 4) * 1024
                        + ((row / 8) & 1) * 512
                        + (copy_lane & 3) * 4;
                    let source = ((row & 7) * 4) ^ (copy_lane / 4);
                    let values = [
                        load_shared(base + source * 16),
                        load_shared(base + (source ^ 1) * 16),
                        load_shared(base + (source ^ 2) * 16),
                        load_shared(base + (source ^ 3) * 16),
                    ];
                    let dst = p.out.wrapping_add(
                        ((qstart + row) as usize * p.q_heads as usize + head as usize) * D
                            + copy_col as usize,
                    );
                    store_global4(dst, values);
                }
            }};
        }
        if D == 256 {
            sixteen!(write_output);
        } else if D == 64 {
            four!(write_output);
        } else {
            eight!(write_output);
        }
    }
}

/// # Safety
/// Launch uses 128 threads, 90112 shared bytes and buffers covering the declared extents.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn attention_prefill_d512(p: Attention, scale: f32) {
    unsafe {
        let p = resolve_position(p);
        attention_d512::<false, false>(p, 0, scale, core::ptr::null_mut(), 1, 0)
    }
}

/// # Safety
/// All keys below `keys` are populated in this sequence; launch uses 128 threads and 90112 shared bytes.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn attention_prefill_d512_bidirectional(
    p: Attention,
    keys: u32,
    scale: f32,
) {
    unsafe {
        let p = resolve_position(p);
        attention_d512::<false, true>(p, keys, scale, core::ptr::null_mut(), 1, 0)
    }
}

/// # Safety
/// Partials cover [rows, heads, parts, 514] F32 values; chunk is a multiple of 32.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn attention_partial_d512(
    p: Attention,
    scale: f32,
    partials: *mut f32,
    parts: u32,
    chunk: u32,
) {
    unsafe {
        let p = resolve_position(p);
        attention_d512::<true, false>(p, 0, scale, partials, parts, chunk)
    }
}

#[inline(always)]
unsafe fn attention_d512<const SPLIT: bool, const BIDI: bool>(
    p: Attention,
    keys: u32,
    scale: f32,
    partials: *mut f32,
    parts: u32,
    chunk: u32,
) {
    unsafe {
        let (tid, tile, head, sm): (u32, u32, u32, u32);
        asm!("mov.u32 {t}, %tid.x; mov.u32 {b}, %ctaid.x; mov.u32 {h}, %ctaid.y; mov.u32 {s}, attention_shared;",
            t=out(reg32) tid,b=out(reg32) tile,h=out(reg32) head,s=out(reg32) sm);
        let lane = tid & 31;
        let warp = tid >> 5;
        let qstart = tile * 16;
        // BIDI: every query row sees all keys below the explicit extent.
        let end = if BIDI { keys } else { p.start + p.rows };
        let part: u32;
        asm!("mov.u32 {p}, %ctaid.z;", p=out(reg32) part);
        let begin = if SPLIT { part * chunk } else { 0 };
        let stop = (if BIDI { end } else { end.min(p.start + qstart + 16) })
            .min(if SPLIT { begin + chunk } else { end });
        let sk = sm + 16384;
        let sv = sm + 49152;
        let partial = sm + 81920;
        let kv = (head / p.group) as usize * p.capacity as usize * 512;
        let copy_row = tid / 64;
        let copy_col = (tid % 64) * 8;
        macro_rules! q_copy {
            ($i:expr) => {{
                let row = copy_row + $i * 2;
                copy(
                    address(sm, row, copy_col, 512),
                    p.q.wrapping_add(
                        (qstart + row) as usize * p.q_stride as usize
                            + head as usize * 512
                            + copy_col as usize,
                    ),
                    qstart + row < p.rows,
                );
            }};
        }
        macro_rules! k_copy {
            ($i:expr) => {{
                let row = copy_row + $i * 2;
                copy(
                    address(sk, row, copy_col, 512),
                    p.k.wrapping_add(kv + (begin + row) as usize * 512 + copy_col as usize),
                    begin + row < stop,
                );
            }};
        }
        eight!(q_copy);
        sixteen!(k_copy);
        commit();
        wait();
        let r0 = lane >> 2;
        let r1 = r0 + 8;
        let mut output = [[0.0f32; 4]; 16];
        let mut peak = [f32::NEG_INFINITY; 2];
        let mut denominator = [0.0f32; 2];
        let mut compensation = [0.0f32; 2];
        let mut key = begin;
        while key < stop {
            macro_rules! v_copy {
                ($i:expr) => {{
                    let row = copy_row + $i * 2;
                    copy(
                        address(sv, row, copy_col, 512),
                        p.v.wrapping_add(kv + (key + row) as usize * 512 + copy_col as usize),
                        key + row < end,
                    );
                }};
            }
            sixteen!(v_copy);
            commit();
            let mut scores = [[0.0f32; 4]; 4];
            macro_rules! qk_step {
                ($i:expr) => {{
                    let a = ld4::<0>(address(
                        sm,
                        lane & 15,
                        warp * 128 + $i * 16 + (lane >> 4) * 8,
                        512,
                    ));
                    macro_rules! pair {
                        ($pair:expr) => {{
                            let b = ld4::<0>(address(
                                sk,
                                (lane & 7) + (lane >> 4) * 8 + $pair * 16,
                                warp * 128 + $i * 16 + ((lane >> 3) & 1) * 8,
                                512,
                            ));
                            scores[$pair * 2] = mma(a, [b[0], b[1]], scores[$pair * 2]);
                            scores[$pair * 2 + 1] = mma(a, [b[2], b[3]], scores[$pair * 2 + 1]);
                        }};
                    }
                    pair!(0);
                    pair!(1);
                }};
            }
            eight!(qk_step);
            wait();
            macro_rules! publish {
                ($i:expr) => {{
                    let col = $i * 8 + (lane & 3) * 2;
                    for j in 0..4 {
                        let row = if j < 2 { r0 } else { r1 };
                        let addr = partial + (warp * 512 + row * 32 + col + (j as u32 & 1)) * 4;
                        asm!("st.shared.f32 [{p}], {x};",p=in(reg32) addr,x=in(reg32) scores[$i][j]);
                    }
                }};
            }
            four!(publish);
            barrier();
            macro_rules! reduce {
                ($i:expr) => {{
                    let col = $i * 8 + (lane & 3) * 2;
                    for j in 0..4 {
                        let row = if j < 2 { r0 } else { r1 };
                        let addr = partial + (row * 32 + col + (j as u32 & 1)) * 4;
                        let a = f32::from_bits(load_shared(addr));
                        let b = f32::from_bits(load_shared(addr + 2048));
                        let c = f32::from_bits(load_shared(addr + 4096));
                        let d = f32::from_bits(load_shared(addr + 6144));
                        scores[$i][j] = ((a + b) + c) + d;
                        let bound = if BIDI {
                            end
                        } else {
                            (p.start + qstart + row + 1).min(end)
                        };
                        if key + col + (j as u32 & 1) >= bound {
                            scores[$i][j] = f32::NEG_INFINITY;
                        }
                    }
                }};
            }
            four!(reduce);
            barrier();
            if key + 32 < stop {
                macro_rules! next_copy {
                    ($i:expr) => {{
                        let row = copy_row + $i * 2;
                        copy(
                            address(sk, row, copy_col, 512),
                            p.k.wrapping_add(
                                kv + (key + 32 + row) as usize * 512 + copy_col as usize,
                            ),
                            key + 32 + row < end,
                        );
                    }};
                }
                sixteen!(next_copy);
                commit();
            }
            let mut next = peak;
            macro_rules! maxima {
                ($i:expr) => {{
                    next[0] = maximum(next[0], maximum(scores[$i][0], scores[$i][1]));
                    next[1] = maximum(next[1], maximum(scores[$i][2], scores[$i][3]));
                }};
            }
            four!(maxima);
            for x in &mut next {
                *x = maximum(*x, shuffle(*x, 1));
                *x = maximum(*x, shuffle(*x, 2));
            }
            let mut alpha = [0.0; 2];
            let mut negative_peak = [0.0; 2];
            for i in 0..2 {
                if next[i] != f32::NEG_INFINITY {
                    alpha[i] = probability::<512>(peak[i] - next[i], 0.0, scale);
                    negative_peak[i] = -next[i] * core::f32::consts::LOG2_E * scale;
                }
            }
            let mut sum = [0.0; 2];
            let mut packed = [[0u32; 4]; 2];
            macro_rules! probabilities {
                ($i:expr) => {{
                    let a = probability::<512>(scores[$i][0], negative_peak[0], scale);
                    let b = probability::<512>(scores[$i][1], negative_peak[0], scale);
                    let c = probability::<512>(scores[$i][2], negative_peak[1], scale);
                    let d = probability::<512>(scores[$i][3], negative_peak[1], scale);
                    sum[0] += a + b;
                    sum[1] += c + d;
                    packed[$i / 2][($i % 2) * 2] = pack(a, b);
                    packed[$i / 2][($i % 2) * 2 + 1] = pack(c, d);
                }};
            }
            four!(probabilities);
            for i in 0..2 {
                let scaled = denominator[i] * alpha[i];
                let adjusted = sum[i] - compensation[i] * alpha[i];
                let next = scaled + adjusted;
                compensation[i] = (next - scaled) - adjusted;
                denominator[i] = next;
            }
            peak = next;
            macro_rules! rescale {
                ($i:expr) => {{
                    output[$i][0] *= alpha[0];
                    output[$i][1] *= alpha[0];
                    output[$i][2] *= alpha[1];
                    output[$i][3] *= alpha[1];
                }};
            }
            sixteen!(rescale);
            macro_rules! pv_step {
                ($inner:expr) => {{
                    macro_rules! pair {
                        ($pair:expr) => {{
                            let b = ld4_trans::<0>(address(
                                sv,
                                (lane & 15) + $inner * 16,
                                warp * 128 + $pair * 16 + (lane >> 4) * 8,
                                512,
                            ));
                            output[$pair * 2] =
                                mma(packed[$inner], [b[0], b[1]], output[$pair * 2]);
                            output[$pair * 2 + 1] =
                                mma(packed[$inner], [b[2], b[3]], output[$pair * 2 + 1]);
                        }};
                    }
                    eight!(pair);
                }};
            }
            pv_step!(0);
            pv_step!(1);
            wait();
            key += 32;
        }
        for i in 0..2 {
            denominator[i] += shuffle(denominator[i], 1);
            denominator[i] += shuffle(denominator[i], 2);
            compensation[i] += shuffle(compensation[i], 1);
            compensation[i] += shuffle(compensation[i], 2);
            denominator[i] -= compensation[i];
        }
        if SPLIT {
            macro_rules! write_partial {
                ($i:expr) => {{
                    let col = warp * 128 + $i * 8 + (lane & 3) * 2;
                    for j in 0..4 {
                        let row = qstart + if j < 2 { r0 } else { r1 };
                        if row < p.rows {
                            let offset = ((row as usize * p.q_heads as usize + head as usize)
                                * parts as usize
                                + part as usize)
                                * 514
                                + col as usize
                                + (j & 1);
                            *partials.add(offset) = output[$i][j];
                        }
                    }
                }};
            }
            sixteen!(write_partial);
            if warp == 0 && lane & 3 == 0 {
                for i in 0..2 {
                    let row = qstart + if i == 0 { r0 } else { r1 };
                    if row < p.rows {
                        let offset = ((row as usize * p.q_heads as usize + head as usize)
                            * parts as usize
                            + part as usize)
                            * 514;
                        *partials.add(offset + 512) = peak[i] * scale;
                        *partials.add(offset + 513) = denominator[i];
                    }
                }
            }
            return;
        }
        let inv = [reciprocal(denominator[0]), reciprocal(denominator[1])];
        macro_rules! write {
            ($i:expr) => {{
                let col = warp * 128 + $i * 8 + (lane & 3) * 2;
                if qstart + r0 < p.rows {
                    let dst = p.out.wrapping_add(((qstart + r0) as usize * p.q_heads as usize + head as usize)*512 + col as usize);
                    asm!("st.global.u32 [{p}], {x};",p=in(reg64) dst,x=in(reg32) pack(output[$i][0]*inv[0],output[$i][1]*inv[0]));
                }
                if qstart + r1 < p.rows {
                    let dst = p.out.wrapping_add(((qstart + r1) as usize * p.q_heads as usize + head as usize)*512 + col as usize);
                    asm!("st.global.u32 [{p}], {x};",p=in(reg64) dst,x=in(reg32) pack(output[$i][2]*inv[1],output[$i][3]*inv[1]));
                }
            }};
        }
        sixteen!(write);
        if warp == 0 && lane & 3 == 0 {
            for i in 0..2 {
                let row = qstart + if i == 0 { r0 } else { r1 };
                if row < p.rows {
                    let value = peak[i] * scale + logarithm(denominator[i]);
                    *p.lse.add(head as usize * p.rows as usize + row as usize) = value;
                }
            }
        }
    }
}

/// # Safety
/// Launch uses 128 threads per [row, head], with partials [rows, heads, parts, 514].
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn attention_merge_d512(
    p: Attention,
    partials: *const f32,
    parts: u32,
) {
    unsafe {
        let (tid, row, head): (u32, u32, u32);
        asm!("mov.u32 {t}, %tid.x; mov.u32 {r}, %ctaid.x; mov.u32 {h}, %ctaid.y;",
            t=out(reg32) tid,r=out(reg32) row,h=out(reg32) head);
        let base = (row as usize * p.q_heads as usize + head as usize) * parts as usize * 514;
        let mut peak = f32::NEG_INFINITY;
        for part in 0..parts {
            peak = maximum(peak, *partials.add(base + part as usize * 514 + 512));
        }
        let mut sum = 0.0f32;
        let mut output = [0.0f32; 4];
        for part in 0..parts {
            let source = partials.add(base + part as usize * 514);
            let denominator = *source.add(513);
            if denominator > 0.0 {
                let weight = probability::<512>(*source.add(512) - peak, 0.0, 1.0);
                sum += weight * denominator;
                for (i, value) in output.iter_mut().enumerate() {
                    *value += weight * *source.add(tid as usize * 4 + i);
                }
            }
        }
        let inv = reciprocal(sum);
        let dst = p
            .out
            .add((row as usize * p.q_heads as usize + head as usize) * 512 + tid as usize * 4);
        let a = pack(output[0] * inv, output[1] * inv);
        let b = pack(output[2] * inv, output[3] * inv);
        asm!("st.global.v2.u32 [{p}], {{{a},{b}}};", p=in(reg64) dst,a=in(reg32) a,b=in(reg32) b);
        if tid == 0 {
            *p.lse.add(head as usize * p.rows as usize + row as usize) = peak + logarithm(sum);
        }
    }
}
