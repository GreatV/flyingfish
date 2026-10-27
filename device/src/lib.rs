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
unsafe fn probability(a: f32, negative_peak: f32) -> f32 {
    let x;
    unsafe {
        asm!("fma.rn.ftz.f32 {x}, {a}, {scale}, {peak}; ex2.approx.ftz.f32 {x}, {x};",
            x=out(reg32) x,a=in(reg32) a,scale=in(reg32) (core::f32::consts::LOG2_E/16.0),peak=in(reg32) negative_peak);
    }
    x
}
#[inline(always)]
unsafe fn logarithm(a: f32) -> f32 {
    let x: f32;
    unsafe {
        asm!("lg2.approx.f32 {x}, {a};",x=out(reg32) x,a=in(reg32) a);
    }
    x * core::f32::consts::LN_2
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
unsafe fn copy_kv<const MASK: bool>(dst: [u32; 2], src: *const u16, row: u32, rows: u32) {
    unsafe {
        macro_rules! transfer {
            ($i:expr) => {
                asm!("cp.async.cg.shared.global [{dst}+{shared_offset}], [{src}+{global_offset}], 16, {bytes};",
                    dst=in(reg32) dst[$i % 2],src=in(reg64) src,
                    shared_offset=const ($i / 2) * 4096,global_offset=const $i * 2048,
                    bytes=in(reg32) if !MASK || row + $i * 4 < rows {16u32} else {0u32});
            };
        }
        sixteen!(transfer);
    }
}

/// # Safety
/// Launch uses 128 threads, 98304 shared bytes, disjoint BF16 Q/K/V/output and F32 LSE buffers covering the declared extents.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn attention_prefill_d256(p: Attention) {
    unsafe {
        let (tid, tile, head, sm): (u32, u32, u32, u32);
        asm!("mov.u32 {t}, %tid.x; mov.u32 {b}, %ctaid.x; mov.u32 {h}, %ctaid.y; mov.u32 {s}, attention_shared;",
            t=out(reg32) tid,b=out(reg32) tile,h=out(reg32) head,s=out(reg32) sm);
        let lane = tid & 31;
        let warp = tid >> 5;
        let qstart = tile * 64;
        let end = p.start + p.rows;
        let diagonal = p.start + qstart;
        let sk = sm + 32768;
        let sv = sm + 65536;
        let copy_row = tid >> 5;
        let copy_col = lane * 8;
        let copy_shared = [
            address(0, copy_row, copy_col, 256),
            address(0, copy_row + 4, copy_col, 256),
        ];
        let kv = (head / p.group) as usize * p.capacity as usize * 256;
        let mut kp =
            p.k.wrapping_add(kv + copy_row as usize * 256 + copy_col as usize);
        let mut vp =
            p.v.wrapping_add(kv + copy_row as usize * 256 + copy_col as usize);
        macro_rules! copy_q {
            ($i:expr) => {{
                let row = copy_row + $i * 4;
                copy(
                    sm + copy_shared[$i % 2] + ($i / 2) * 4096,
                    p.q.wrapping_add(
                        (qstart + row) as usize * p.q_stride as usize
                            + head as usize * 256
                            + copy_col as usize,
                    ),
                    qstart + row < p.rows,
                );
            }};
        }
        sixteen!(copy_q);
        copy_kv::<true>(
            [sk + copy_shared[0], sk + copy_shared[1]],
            kp,
            copy_row,
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
            address(sm, qr, qc, 256),
            address(sm, qr, qc + 16, 256),
            address(sm, qr, qc + 32, 256),
            address(sm, qr, qc + 48, 256),
        ];
        let kb = [
            address(sk, kr, kc, 256),
            address(sk, kr, kc + 16, 256),
            address(sk, kr, kc + 32, 256),
            address(sk, kr, kc + 48, 256),
        ];
        let vb = [
            address(sv, vr, vc, 256),
            address(sv, vr, vc + 16, 256),
            address(sv, vr, vc + 32, 256),
            address(sv, vr, vc + 48, 256),
        ];
        let mut output = [[0.0f32; 4]; 32];
        let mut peak = [f32::NEG_INFINITY; 2];
        let mut denominator = [0.0f32; 2];
        let r0 = warp * 16 + (lane >> 2);
        let r1 = r0 + 8;
        let mut key = 0;
        macro_rules! tile_step {
            ($masked:expr) => {{
                copy_kv::<$masked>(
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
                                let b =
                                    ld4::<{ ($inner / 4) * 128 + $pair * 8192 }>(kb[$inner % 4]);
                                scores[$pair * 2] = mma(a, [b[0], b[1]], scores[$pair * 2]);
                                scores[$pair * 2 + 1] = mma(a, [b[2], b[3]], scores[$pair * 2 + 1]);
                            }};
                        }
                        four!(qk_pair);
                    }};
                }
                sixteen!(qk_step);
                wait();
                let next_key = key + 64;
                kp = kp.wrapping_add(64 * 256);
                if next_key < end.min(diagonal + 64) {
                    let destinations = [sk + copy_shared[0], sk + copy_shared[1]];
                    if next_key + 64 <= end {
                        copy_kv::<false>(destinations, kp, next_key + copy_row, end);
                    } else {
                        copy_kv::<true>(destinations, kp, next_key + copy_row, end);
                    }
                    commit();
                }
                let mut next = peak;
                macro_rules! mask_max {
                    ($n:expr) => {{
                        if $masked {
                            let col = key + $n * 8 + (lane & 3) * 2;
                            if col > diagonal + r0 || col >= end {
                                scores[$n][0] = f32::NEG_INFINITY;
                            }
                            if col + 1 > diagonal + r0 || col + 1 >= end {
                                scores[$n][1] = f32::NEG_INFINITY;
                            }
                            if col > diagonal + r1 || col >= end {
                                scores[$n][2] = f32::NEG_INFINITY;
                            }
                            if col + 1 > diagonal + r1 || col + 1 >= end {
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
                let negative_peak = [
                    -next[0] * (core::f32::consts::LOG2_E / 16.0),
                    -next[1] * (core::f32::consts::LOG2_E / 16.0),
                ];
                let alpha = [
                    probability(peak[0], negative_peak[0]),
                    probability(peak[1], negative_peak[1]),
                ];
                let mut sum = [0.0f32; 2];
                let mut packed = [[0u32; 4]; 4];
                macro_rules! probabilities {
                    ($n:expr) => {{
                        let a = probability(scores[$n][0], negative_peak[0]);
                        let b = probability(scores[$n][1], negative_peak[0]);
                        let c = probability(scores[$n][2], negative_peak[1]);
                        let d = probability(scores[$n][3], negative_peak[1]);
                        sum[0] += a + b;
                        sum[1] += c + d;
                        packed[$n / 2][($n % 2) * 2] = pack(a, b);
                        packed[$n / 2][($n % 2) * 2 + 1] = pack(c, d);
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
                thirty_two!(rescale);
                macro_rules! pv_step {
                    ($inner:expr) => {{
                        let a = packed[$inner];
                        macro_rules! pv_pair {
                            ($pair:expr) => {{
                                let b = ld4_trans::<{ ($pair / 4) * 128 + $inner * 8192 }>(
                                    vb[$pair % 4],
                                );
                                output[$pair * 2] = mma(a, [b[0], b[1]], output[$pair * 2]);
                                output[$pair * 2 + 1] = mma(a, [b[2], b[3]], output[$pair * 2 + 1]);
                            }};
                        }
                        sixteen!(pv_pair);
                    }};
                }
                four!(pv_step);
                wait();
            }};
        }
        while key + 64 <= diagonal && key + 64 <= end {
            tile_step!(false);
            key += 64;
            vp = vp.wrapping_add(64 * 256);
        }
        while key < end.min(diagonal + 64) {
            tile_step!(true);
            key += 64;
            vp = vp.wrapping_add(64 * 256);
        }
        let (tid, tile, head, sm): (u32, u32, u32, u32);
        asm!("mov.u32 {t}, %tid.x; mov.u32 {b}, %ctaid.x; mov.u32 {h}, %ctaid.y; mov.u32 {s}, attention_shared;",
            t=out(reg32) tid,b=out(reg32) tile,h=out(reg32) head,s=out(reg32) sm);
        let lane = tid & 31;
        let warp = tid >> 5;
        let qstart = tile * 64;
        let copy_row = tid >> 5;
        let copy_col = lane * 8;
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
                    .write(peak[0] / 16.0 + logarithm(denominator[0]));
            }
            if qstart + r1 < p.rows {
                p.lse
                    .add((head * p.rows + qstart + r1) as usize)
                    .write(peak[1] / 16.0 + logarithm(denominator[1]));
            }
        }
        macro_rules! stage_output {
            ($group:expr) => {{
                let base = sm + warp * 8192 + $group * 1024 + (lane ^ $group) * 16;
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
        eight!(stage_output);
        barrier();
        macro_rules! write_output {
            ($step:expr) => {{
                let row = copy_row + $step * 4;
                if qstart + row < p.rows {
                    let base = sm
                        + (row / 16) * 8192
                        + (lane / 4) * 1024
                        + ((row / 8) & 1) * 512
                        + (lane & 3) * 4;
                    let source = ((row & 7) * 4) ^ (lane / 4);
                    let values = [
                        load_shared(base + source * 16),
                        load_shared(base + (source ^ 1) * 16),
                        load_shared(base + (source ^ 2) * 16),
                        load_shared(base + (source ^ 3) * 16),
                    ];
                    let dst = p.out.wrapping_add(
                        ((qstart + row) as usize * p.q_heads as usize + head as usize) * 256
                            + copy_col as usize,
                    );
                    store_global4(dst, values);
                }
            }};
        }
        sixteen!(write_output);
    }
}
