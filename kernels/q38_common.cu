// Q38 ABI v1 common norm/gates. BF16-visible operations round explicitly.
#include "q38_contract.cuh"
#include <cuda_bf16.h>
#include <math.h>
__device__ __forceinline__ float bround(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float sigmoid(float x) {
    if (x>=0.0f) return 1.0f/(1.0f+expf(-x));
    float e=expf(x); return e/(1.0f+e);
}
__device__ __forceinline__ float silu(float x) {return x*sigmoid(x);}
extern "C" __global__ void q38_bias_add(__nv_bfloat16* y,const __nv_bfloat16* bias,uint64_t count,uint32_t width) {
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=uint64_t(gridDim.x)*blockDim.x)
        y[i]=__hadd(y[i],bias[i%width]);
}
extern "C" __global__ void q38_advance(Q38SequenceMeta* meta) {
    q38_meta_guard(meta);
    if(threadIdx.x==0 && blockIdx.x==0) { ++meta->prefix; ++meta->key_count; }
}
extern "C" __global__ void q38_multiply(const __nv_bfloat16* a,const __nv_bfloat16* b,__nv_bfloat16* y,uint64_t count) {
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=uint64_t(gridDim.x)*blockDim.x)
        y[i]=__hmul(a[i],b[i]);
}
__device__ __forceinline__ float block_sum(float v,float* scratch) {
    for(int delta=16;delta;delta/=2) v+=__shfl_down_sync(0xffffffffu,v,delta);
    if(threadIdx.x%32==0) scratch[threadIdx.x/32]=v;
    __syncthreads();
    v=threadIdx.x<8?scratch[threadIdx.x]:0;
    if(threadIdx.x<32) {
        for(int delta=16;delta;delta/=2) v+=__shfl_down_sync(0xffffffffu,v,delta);
        if(threadIdx.x==0) scratch[0]=v;
    }
    __syncthreads(); return scratch[0];
}
// mode: 0 Standard, 1 ZeroCentered, 2 BF16 L2 chain, 3 StandardGated.
// x/y/z strides are in elements; weight stride=1; one CTA per row/head.
extern "C" __global__ void q38_norm(
    const __nv_bfloat16* x,const __nv_bfloat16* w,const __nv_bfloat16* z,__nv_bfloat16* y,
    const Q38SequenceMeta* meta,uint32_t row_base,uint32_t row_capacity,
    uint32_t heads,uint32_t width,
    uint64_t xs0,uint64_t xs1,uint64_t xs2,uint64_t ys0,uint64_t ys1,uint64_t ys2,
    uint64_t zs0,uint64_t zs1,uint64_t zs2,int mode,float eps) {
    q38_meta_guard(meta);
    if(blockDim.x!=256 || blockDim.y!=1 || blockDim.z!=1 || gridDim.y!=1 || gridDim.z!=1 ||
       !heads || !width || !row_capacity || uint64_t(gridDim.x)!=uint64_t(row_capacity)*heads ||
       mode<0 || mode>3 || !x || !y || (mode!=2 && !w) || (mode==3 && !z) || !(eps>0)) {__trap();return;}
    uint64_t row=uint64_t(row_base)+blockIdx.x/heads;
    if(row>=uint64_t(meta->valid_rows)) return;
    uint32_t head=blockIdx.x%heads;
    x+=row*xs0+head*xs1; y+=row*ys0+head*ys1;
    if(mode==3) z+=row*zs0+head*zs1;
    float sum=0;
    for(uint32_t d=threadIdx.x;d<width;d+=256) {
        float a=__bfloat162float(x[uint64_t(d)*xs2]);
        float square=__fmul_rn(a,a);
        sum=__fadd_rn(sum,mode==2?bround(square):square);
    }
    __shared__ float tmp[8];
    sum=block_sum(sum,tmp);
    float inv;
    if(mode==2) inv=bround(rsqrtf(bround(__fadd_rn(bround(sum),eps))));
    else inv=rsqrtf(__fadd_rn(sum/float(width),eps));
    for(uint32_t d=threadIdx.x;d<width;d+=256) {
        float a=__bfloat162float(x[uint64_t(d)*xs2]);
        float result=__fmul_rn(a,inv);
        if(mode==1) result=__fmul_rn(result,__fadd_rn(1.0f,__bfloat162float(w[d])));
        else if(mode!=2) {
            result=bround(__fmul_rn(bround(result),__bfloat162float(w[d])));
            if(mode==3) result=__fmul_rn(result,silu(__bfloat162float(z[uint64_t(d)*zs2])));
        }
        y[uint64_t(d)*ys2]=__float2bfloat16_rn(result);
    }
}
// mode: 0 SiLU, 1 BF16 sigmoid(gate) followed by BF16 multiply.
extern "C" __global__ void q38_gate(
    const __nv_bfloat16* x,const __nv_bfloat16* gate,__nv_bfloat16* y,
    const Q38SequenceMeta* meta,uint32_t row_base,uint32_t row_capacity,
    uint32_t heads,uint32_t width,
    uint64_t xs0,uint64_t xs1,uint64_t xs2,uint64_t gs0,uint64_t gs1,uint64_t gs2,
    uint64_t ys0,uint64_t ys1,uint64_t ys2,int mode) {
    q38_meta_guard(meta);
    if(blockDim.x!=256 || blockDim.y!=1 || blockDim.z!=1 || gridDim.y!=1 || gridDim.z!=1 ||
       !heads || !width || !row_capacity || !x || !y || mode<0 || mode>1 || (mode==1 && !gate)) {__trap();return;}
    uint64_t count=uint64_t(row_capacity)*heads*width;
    for(uint64_t i=uint64_t(blockIdx.x)*256+threadIdx.x;i<count;i+=uint64_t(gridDim.x)*256) {
        uint64_t row=uint64_t(row_base)+i/(uint64_t(heads)*width);
        if(row>=uint64_t(meta->valid_rows)) continue;
        uint32_t h=(i/width)%heads,d=i%width;
        float a=__bfloat162float(x[row*xs0+h*xs1+d*xs2]);
        float out=mode==0?silu(a):__fmul_rn(a,bround(sigmoid(__bfloat162float(gate[row*gs0+h*gs1+d*gs2]))));
        y[row*ys0+h*ys1+d*ys2]=__float2bfloat16_rn(out);
    }
}
extern "C" __global__ void q38_gdn_gates(
    const __nv_bfloat16* a,const __nv_bfloat16* b,const __nv_bfloat16* a_log,const __nv_bfloat16* dt,
    __nv_bfloat16* beta,float* g,const Q38SequenceMeta* meta,uint32_t row_base,uint32_t row_capacity,
    uint64_t as0,uint64_t bs0,uint64_t betas0,uint64_t gs0) {
    q38_meta_guard(meta);
    if(blockDim.x!=256 || blockDim.y!=1 || blockDim.z!=1 || gridDim.y!=1 || gridDim.z!=1 ||
       !a || !b || !a_log || !dt || !beta || !g || !row_capacity) {__trap();return;}
    uint64_t count=uint64_t(row_capacity)*48;
    for(uint64_t i=uint64_t(blockIdx.x)*256+threadIdx.x;i<count;i+=uint64_t(gridDim.x)*256) {
        uint64_t row=uint64_t(row_base)+i/48; if(row>=uint64_t(meta->valid_rows))continue;
        uint32_t h=i%48;
        float t=__fadd_rn(__bfloat162float(a[row*as0+h]),__bfloat162float(dt[h]));
        float soft=t>20.0f?t:log1pf(expf(t));
        beta[row*betas0+h]=__float2bfloat16_rn(sigmoid(__bfloat162float(b[row*bs0+h])));
        g[row*gs0+h]=-expf(__bfloat162float(a_log[h]))*soft;
    }
}

// Stateless LayerNorm: every declared row is valid. No text sequence/causal metadata.
// One CTA per [row,head]; FP32 centered variance and affine, BF16 input/output.
extern "C" __global__ void q38_layer_norm(
    const __nv_bfloat16* x,const __nv_bfloat16* weight,const __nv_bfloat16* bias,__nv_bfloat16* y,
    uint32_t rows,uint32_t heads,uint32_t width,
    uint64_t xs0,uint64_t xs1,uint64_t xs2,uint64_t ys0,uint64_t ys1,uint64_t ys2,float eps) {
    if(blockDim.x!=256 || blockDim.y!=1 || blockDim.z!=1 || gridDim.y!=1 || gridDim.z!=1 ||
       !rows || !heads || !width || uint64_t(gridDim.x)!=uint64_t(rows)*heads ||
       !x || !weight || !bias || !y || !(eps>0)) {__trap();return;}
    uint32_t row=blockIdx.x/heads,head=blockIdx.x%heads;
    x+=uint64_t(row)*xs0+uint64_t(head)*xs1;
    y+=uint64_t(row)*ys0+uint64_t(head)*ys1;
    __shared__ float tmp[8];
    float sum=0;
    for(uint32_t d=threadIdx.x;d<width;d+=256) sum=__fadd_rn(sum,__bfloat162float(x[uint64_t(d)*xs2]));
    float mean=block_sum(sum,tmp)/float(width);
    __syncthreads(); // Every thread has read the mean before reduction scratch is reused.
    float variance=0;
    for(uint32_t d=threadIdx.x;d<width;d+=256) {
        float delta=__fsub_rn(__bfloat162float(x[uint64_t(d)*xs2]),mean);
        variance=__fadd_rn(variance,__fmul_rn(delta,delta));
    }
    float inv=__fdiv_rn(1.0f,__fsqrt_rn(__fadd_rn(block_sum(variance,tmp)/float(width),eps)));
    for(uint32_t d=threadIdx.x;d<width;d+=256) {
        float centered=__fsub_rn(__bfloat162float(x[uint64_t(d)*xs2]),mean);
        float normalized=__fmul_rn(centered,inv);
        float affine=__fadd_rn(__fmul_rn(normalized,__bfloat162float(weight[d])),__bfloat162float(bias[d]));
        y[uint64_t(d)*ys2]=__float2bfloat16_rn(affine);
    }
}
// mode 0: erf GELU. mode 1: tanh approximation. All declared rows are valid.
extern "C" __global__ void q38_gelu(
    const __nv_bfloat16* x,__nv_bfloat16* y,uint32_t rows,uint32_t heads,uint32_t width,
    uint64_t xs0,uint64_t xs1,uint64_t xs2,uint64_t ys0,uint64_t ys1,uint64_t ys2,int mode) {
    if(blockDim.x!=256 || blockDim.y!=1 || blockDim.z!=1 || gridDim.y!=1 || gridDim.z!=1 ||
       !rows || !heads || !width || !x || !y || mode<0 || mode>1) {__trap();return;}
    uint64_t count=uint64_t(rows)*heads*width;
    for(uint64_t i=uint64_t(blockIdx.x)*256+threadIdx.x;i<count;i+=uint64_t(gridDim.x)*256) {
        uint64_t row=i/(uint64_t(heads)*width);uint32_t head=(i/width)%heads,d=i%width;
        float value=__bfloat162float(x[row*xs0+head*xs1+uint64_t(d)*xs2]);
        float cdf;
        if(mode==0) cdf=__fadd_rn(1.0f,erff(__fmul_rn(value,0.7071067811865475244f)));
        else {
            float cubic=__fmul_rn(__fmul_rn(value,value),value);
            float inner=__fmul_rn(0.7978845608028653559f,__fadd_rn(value,__fmul_rn(0.044715f,cubic)));
            cdf=__fadd_rn(1.0f,tanhf(inner));
        }
        y[row*ys0+head*ys1+uint64_t(d)*ys2]=__float2bfloat16_rn(__fmul_rn(__fmul_rn(0.5f,value),cdf));
    }
}
