#include <cuda_bf16.h>

template<int N, int Group>
__device__ __forceinline__ int gdn_index(int row, int col) {
    return row * N + (col ^ ((row & (N / Group - 1)) * Group));
}

extern "C" __global__ void gdn_conv_rows(
    const float* qkv, const float* w, const float* state_in,
    float* state_out, float* out, int tokens, int channels, int kernel)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (c >= channels) return;
    if (t < tokens) {
        float acc = 0.0f;
        for (int j = 0; j < kernel; ++j) {
            const int source = t + j - kernel + 1;
            const float x = source < 0 ? state_in[(source + kernel - 1) * channels + c]
                                      : qkv[(long long)source * channels + c];
            acc += w[c * kernel + j] * x;
        }
        out[(long long)t * channels + c] = acc / (1.0f + expf(-acc));
    }
    if (t == 0) {
        for (int j = 0; j < kernel - 1; ++j) {
            const int source = tokens + j - kernel + 1;
            state_out[j * channels + c] = source < 0
                ? state_in[(source + kernel - 1) * channels + c]
                : qkv[(long long)source * channels + c];
        }
    }
}

template<int C>
__device__ __forceinline__ void gdn_prepare(
    const float* conv, const float* a, const float* b, const float* a_log,
    const float* dt, unsigned short* qk_out, unsigned short* v_out,
    float* control, int tokens, int nk, int nv)
{
    extern __shared__ __align__(16) unsigned char storage[];
    auto* qk = reinterpret_cast<__nv_bfloat16*>(storage);
    float* lower = reinterpret_cast<float*>(qk + 2 * C * 128);
    float* solve = lower + C * C;
    float* mqk = solve + C * C;
    float* g = mqk + C * C;
    float* beta = g + C;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    const int h = blockIdx.x % nv, chunk = blockIdx.x / nv, first = chunk * C;
    const int valid = max(0, min(C, tokens - first));
    const int kh = h / (nv / nk), kd = nk * 128, channels = 2 * kd + nv * 128;
    for (int row = warp; row < C; row += blockDim.x / 32) {
        float q[4]{}, k[4]{};
        float qs = 0.0f, ks = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = lane * 4 + j;
            if (row < valid) {
                const long long at = (long long)(first + row) * channels + kh * 128 + col;
                q[j] = conv[at]; k[j] = conv[at + kd];
            }
            qs += q[j] * q[j]; ks += k[j] * k[j];
        }
        #pragma unroll
        for (int off = 16; off; off >>= 1) {
            qs += __shfl_down_sync(0xffffffff, qs, off);
            ks += __shfl_down_sync(0xffffffff, ks, off);
        }
        const float qn = sqrtf(__shfl_sync(0xffffffff, qs, 0) + 1e-6f);
        const float kn = sqrtf(__shfl_sync(0xffffffff, ks, 0) + 1e-6f);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int at = gdn_index<128,8>(row, lane * 4 + j);
            qk[at] = __float2bfloat16_rn(q[j] / qn);
            qk[C * 128 + at] = __float2bfloat16_rn(k[j] / kn);
        }
    }
    if (tid == 0) {
        float sum = 0.0f;
        for (int row = 0; row < C; ++row) {
            if (row < valid) {
                const long long at = (long long)(first + row) * nv + h;
                const float ap = a[at] + dt[h];
                const float sp = ap > 20.0f ? ap : logf(1.0f + expf(ap));
                sum += -expf(a_log[h]) * sp;
                beta[row] = 1.0f / (1.0f + expf(-b[at]));
                g[row] = sum;
            } else { beta[row] = 0.0f; g[row] = 0.0f; }
        }
    }
    for (int i = tid; i < C * 128; i += blockDim.x) {
        const int row = i / 128, col = i % 128;
        if (row < valid) {
            const long long at = (long long)(first + row) * channels + 2 * kd + h * 128 + col;
            v_out[((long long)(first + row) * nv + h) * 128 + col] =
                __bfloat16_as_ushort(__float2bfloat16_rn(conv[at]));
        }
    }
    __syncthreads();
    constexpr int tiles = (C / 16) * (C / 8);
    const auto* bits = reinterpret_cast<const unsigned short*>(qk);
    for (int tile = warp; tile < 2 * tiles; tile += blockDim.x / 32) {
        const bool query = tile >= tiles;
        const int m = ((tile % tiles) / (C / 8)) * 16;
        const int n = (tile % (C / 8)) * 8;
        const int r = lane / 4, pair = (lane & 3) * 2;
        float d[4]{};
        const unsigned short* av = bits + (query ? 0 : C * 128);
        const unsigned short* bv = bits + C * 128;
        for (int k0 = 0; k0 < 128; k0 += 16) {
            unsigned aa[4], bb[2];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int row = m + r + (j & 1) * 8;
                const int col = k0 + pair + (j / 2) * 8;
                aa[j] = (unsigned)av[gdn_index<128,8>(row,col)] |
                    ((unsigned)av[gdn_index<128,8>(row,col+1)] << 16);
            }
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int col = k0 + pair + j * 8;
                bb[j] = (unsigned)bv[gdn_index<128,8>(n+r,col)] |
                    ((unsigned)bv[gdn_index<128,8>(n+r,col+1)] << 16);
            }
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                : "r"(aa[0]), "r"(aa[1]), "r"(aa[2]), "r"(aa[3]), "r"(bb[0]), "r"(bb[1]));
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int row = m + r + (j / 2) * 8, col = n + pair + (j & 1);
            float value = 0.0f;
            if (row < valid && col < valid && (query ? row >= col : row > col)) {
                value = d[j] * exp2f((g[row] - g[col]) * 1.4426950408889634f);
                if (!query) value *= beta[row];
            }
            (query ? mqk : lower)[row * C + col] = value;
        }
    }
    __syncthreads();
    if (tid < C) {
        for (int row = 0; row < C; ++row) {
            float value = 0.0f;
            if (row < valid && tid < valid) {
                value = row == tid ? beta[tid] : 0.0f;
                for (int j = 0; j < row; ++j)
                    value = fmaf(-lower[row*C+j], solve[j*C+tid], value);
            }
            solve[row*C+tid] = value;
        }
    }
    __syncthreads();
    const long long base = ((long long)chunk * nv + h) * (2 * C * C + 2 * C);
    for (int i = tid; i < C * C; i += blockDim.x) {
        control[base+i] = solve[i]; control[base+C*C+i] = mqk[i];
    }
    if (tid < C) {
        control[base+2*C*C+tid] = tid < valid ? exp2f(g[tid] * 1.4426950408889634f) : 0.0f;
        control[base+2*C*C+C+tid] = tid < valid
            ? exp2f((g[valid-1]-g[tid]) * 1.4426950408889634f) : 0.0f;
    }
    if (h % (nv / nk) == 0) {
        for (int i = tid; i < 2*C*128; i += blockDim.x) {
            const int which = i / (C*128), row = (i/128)%C, col = i%128;
            qk_out[((long long)chunk*nk+kh)*2*C*128+i] =
                bits[which*C*128+gdn_index<128,8>(row,col)];
        }
    }
}

template<int C, int DV>
__device__ __forceinline__ void gdn_recurrence(
    const unsigned short* qk_in, const unsigned short* v, const float* control,
    const float* state_in, float* state_out, float* out,
    int tokens, int chunks, int nk, int nv, float scale)
{
    extern __shared__ __align__(16) unsigned char storage[];
    auto* qk = reinterpret_cast<__nv_bfloat16*>(storage);
    float* matrix = reinterpret_cast<float*>(qk + 2*C*128);
    float* prefix = matrix + C*C;
    float* suffix = prefix + C;
    float* delta = suffix + C;
    constexpr int lanes = 256 / DV, keys = 128 / lanes;
    const int tid = threadIdx.x, lane = tid % lanes, value = tid / lanes;
    const int head = blockIdx.x / (128/DV), start_value = blockIdx.x % (128/DV) * DV;
    const int kh = head / (nv/nk);
    const long long base = ((long long)head*128 + start_value + value)*128;
    float state[keys];
    #pragma unroll
    for (int r = 0; r < keys; ++r) state[r] = state_in[base+lane*keys+r];
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int first = chunk*C, valid = min(C, tokens-first);
        if (valid <= 0) break;
        const long long ctrl = ((long long)chunk*nv+head)*(2*C*C+2*C);
        for (int i = tid; i < 2*C*128; i += blockDim.x) {
            const int which = i/(C*128), row = (i/128)%C, col = i%128;
            qk[which*C*128+gdn_index<128,8>(row,col)] =
                __ushort_as_bfloat16(qk_in[((long long)chunk*nk+kh)*2*C*128+i]);
        }
        for (int i = tid; i < C*C; i += blockDim.x)
            matrix[gdn_index<C,4>(i/C,i%C)] = control[ctrl+i];
        if (tid < C) { prefix[tid]=control[ctrl+2*C*C+tid]; suffix[tid]=control[ctrl+2*C*C+C+tid]; }
        __syncthreads();
        for (int t = 0; t < valid; ++t) {
            float acc = 0.0f;
            #pragma unroll
            for (int r = 0; r < keys; ++r)
                acc = fmaf(state[r], __bfloat162float(qk[C*128+gdn_index<128,8>(t,lane*keys+r)]), acc);
            #pragma unroll
            for (int off = lanes/2; off; off>>=1) acc += __shfl_xor_sync(0xffffffff,acc,off,lanes);
            if (lane == 0) {
                const float vv = __bfloat162float(__ushort_as_bfloat16(v[((long long)(first+t)*nv+head)*128+start_value+value]));
                delta[gdn_index<DV,DV/8>(t,value)] = vv - prefix[t]*acc;
            }
        }
        __syncthreads();
        float transformed[C/lanes];
        #pragma unroll
        for (int i = 0; i < C/lanes; ++i) {
            const int row = lane+i*lanes;
            float acc = 0.0f;
            if (row < valid) {
                for (int j = 0; j <= row; ++j)
                    acc = fmaf(matrix[gdn_index<C,4>(row,j)],delta[gdn_index<DV,DV/8>(j,value)],acc);
            }
            transformed[i] = acc;
        }
        __syncwarp();
        #pragma unroll
        for (int i = 0; i < C/lanes; ++i) {
            const int row = lane+i*lanes;
            if (row < valid) delta[gdn_index<DV,DV/8>(row,value)] = transformed[i];
        }
        __syncthreads();
        for (int i = tid; i < C*C; i += blockDim.x)
            matrix[gdn_index<C,4>(i/C,i%C)] = control[ctrl+C*C+i];
        __syncthreads();
        for (int t = 0; t < valid; ++t) {
            float acc = 0.0f;
            #pragma unroll
            for (int r = 0; r < keys; ++r)
                acc = fmaf(state[r],__bfloat162float(qk[gdn_index<128,8>(t,lane*keys+r)]),acc);
            #pragma unroll
            for (int off = lanes/2; off; off>>=1) acc += __shfl_xor_sync(0xffffffff,acc,off,lanes);
            if (lane == 0) {
                acc *= prefix[t];
                for (int j = 0; j <= t; ++j)
                    acc = fmaf(matrix[gdn_index<C,4>(t,j)],delta[gdn_index<DV,DV/8>(j,value)],acc);
                out[((long long)(first+t)*nv+head)*128+start_value+value] = acc*scale;
            }
        }
        #pragma unroll
        for (int r = 0; r < keys; ++r) {
            float acc = prefix[valid-1]*state[r];
            for (int t = 0; t < valid; ++t)
                acc = fmaf(suffix[t]*delta[gdn_index<DV,DV/8>(t,value)],
                    __bfloat162float(qk[C*128+gdn_index<128,8>(t,lane*keys+r)]),acc);
            state[r] = acc;
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < keys; ++r) state_out[base+lane*keys+r] = state[r];
}

extern "C" __global__ void gdn_norm_rows(
    const float* raw, const float* z, const float* w, float* out,
    int dv, float eps)
{
    const long long base = (long long)blockIdx.x * dv;
    const int tid = threadIdx.x;
    __shared__ float inv;
    if (tid < 32) {
        float sq = 0.0f;
        for (int j = 0; j < dv/32; ++j) {
            const float v = raw[base+tid*(dv/32)+j]; sq += v*v;
        }
        #pragma unroll
        for (int off = 16; off; off>>=1) sq += __shfl_down_sync(0xffffffff,sq,off);
        if (tid == 0) inv = 1.0f/sqrtf(sq/dv+eps);
    }
    __syncthreads();
    for (int i = tid; i < dv; i += blockDim.x) {
        const float zz = z[base+i];
        out[base+i] = raw[base+i]*inv*w[i]*(zz/(1.0f+expf(-zz)));
    }
}

#define GDN_PREP(C) \
extern "C" __global__ void gdn_prepare_c##C(const float* x,const float* a,const float* b,const float* al,const float* dt,unsigned short* qk,unsigned short* v,float* ctrl,int t,int nk,int nv) { gdn_prepare<C>(x,a,b,al,dt,qk,v,ctrl,t,nk,nv); }
#define GDN_REC(C,DV) \
extern "C" __global__ __launch_bounds__(256) void gdn_recurrence_c##C##_dv##DV(const unsigned short* qk,const unsigned short* v,const float* ctrl,const float* si,float* so,float* out,int t,int chunks,int nk,int nv,float scale) { gdn_recurrence<C,DV>(qk,v,ctrl,si,so,out,t,chunks,nk,nv,scale); }
GDN_PREP(16) GDN_PREP(32) GDN_PREP(64)
GDN_REC(16,64) GDN_REC(16,128) GDN_REC(32,64) GDN_REC(32,128) GDN_REC(64,64) GDN_REC(64,128)
#undef GDN_PREP
#undef GDN_REC
