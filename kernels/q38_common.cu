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
extern "C" __global__ void q38_copy_finite_bf16(const __nv_bfloat16* source,
                                               __nv_bfloat16* destination,
                                               uint64_t count) {
    for (uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
         i<count; i+=uint64_t(gridDim.x)*blockDim.x) {
        __nv_bfloat16 value=source[i];
        if (!isfinite(__bfloat162float(value))) { __trap(); return; }
        destination[i]=value;
    }
}
extern "C" __global__ void q38_bias_add(__nv_bfloat16* y,const __nv_bfloat16* bias,uint64_t count,uint32_t width) {
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=uint64_t(gridDim.x)*blockDim.x)
        y[i]=__hadd(y[i],bias[i%width]);
}
extern "C" __global__ void q38_advance(Q38SequenceMeta* meta) {
    q38_meta_guard(meta);
    if(threadIdx.x==0 && blockIdx.x==0) { ++meta->prefix; ++meta->key_count; }
}
// Fixed-row MTP verification graphs read a changing device position. Keep the
// final physical slot available for the next-row decode metadata.
extern "C" __global__ void q38_batch_meta(const int32_t* position,
                                         Q38SequenceMeta* meta, int32_t rows,
                                         int32_t capacity) {
    if (gridDim.x != 1 || gridDim.y != 1 || gridDim.z != 1 ||
        blockDim.x != 1 || blockDim.y != 1 || blockDim.z != 1) {
        __trap(); return;
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int32_t p = *position;
        if (p < 0 || rows < 1 || rows > 4 || int64_t(p) + rows >= capacity) {
            __trap(); return;
        }
        *meta = {p, rows, p + rows, capacity, 1, {0, 0, 0}};
        q38_meta_guard(meta);
    }
}
extern "C" __global__ void q38_sequence_check(const Q38SequenceMeta* meta,
                                              int32_t prefix, int32_t capacity,
                                              int32_t max_rows, uint32_t* status) {
    if (blockIdx.x == 0 && threadIdx.x == 0 &&
        (meta->prefix != prefix || meta->prefix < 0 || meta->valid_rows < 1 ||
         meta->valid_rows > max_rows || meta->capacity != capacity || meta->causal != 1 ||
         int64_t(meta->prefix) + meta->valid_rows != meta->key_count ||
         meta->key_count > capacity || meta->reserved[0] || meta->reserved[1] || meta->reserved[2]))
        atomicOr(status, 0x80000000u);
}
extern "C" __global__ void q38_stage_inputs(const uint32_t* source, uint32_t* ids,
                                            int32_t* positions, Q38SequenceMeta* meta,
                                            uint32_t rows, int32_t prefix,
                                            int32_t capacity, uint32_t vocab,
                                            uint32_t* status) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row == 0) *meta = {prefix, int32_t(rows), prefix + int32_t(rows), capacity, 1, {0, 0, 0}};
    if (row >= rows) return;
    uint32_t token = source[row];
    if (token >= vocab) { atomicOr(status, 0x40000000u); token = 0; }
    ids[row] = token;
    positions[3 * row] = positions[3 * row + 1] = positions[3 * row + 2] = prefix + int32_t(row) + 1;
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
union Bf16Vec4 {
    uint2 u;
    __nv_bfloat162 h[2];
    __nv_bfloat16 e[4];
};
union Bf16Vec8 {
    uint4 u;
    __nv_bfloat162 h[4];
    __nv_bfloat16 e[8];
};
__device__ __forceinline__ float norm_term(float a,int mode) {
    float square=__fmul_rn(a,a);
    return mode==2?bround(square):square;
}
__device__ __forceinline__ float norm_inv(float sum,uint32_t width,int mode,float eps) {
    if(mode==2) return bround(rsqrtf(bround(__fadd_rn(bround(sum),eps))));
    float scaled=(width==128u)?__fmul_rn(sum,1.0f/128.0f):(width==256u)?__fmul_rn(sum,1.0f/256.0f):(sum/float(width));
    return rsqrtf(__fadd_rn(scaled,eps));
}
__device__ __forceinline__ __nv_bfloat16 norm_scale(float a,float inv,__nv_bfloat16 wd,__nv_bfloat16 zd,int mode) {
    float result=__fmul_rn(a,inv);
    if(mode==1 || mode==4) result=__fmul_rn(result,__fadd_rn(1.0f,__bfloat162float(wd)));
    else if(mode!=2) {
        result=bround(__fmul_rn(bround(result),__bfloat162float(wd)));
        if(mode==3) result=__fmul_rn(result,silu(__bfloat162float(zd)));
    }
    return __float2bfloat16_rn(result);
}
__device__ __forceinline__ void divmod_heads(uint32_t slice,uint32_t heads,uint32_t& row_off,uint32_t& head) {
    if(heads==1u) { row_off=slice; head=0u; }
    else if(heads==4u) { row_off=slice>>2; head=slice&3u; }
    else if(heads==16u) { row_off=slice>>4; head=slice&15u; }
    else if(heads==24u) { row_off=uint32_t((uint64_t(slice>>3)*0x55555556ull)>>32); head=slice-row_off*24u; }
    else if(heads==48u) { row_off=uint32_t((uint64_t(slice>>4)*0x55555556ull)>>32); head=slice-row_off*48u; }
    else { row_off=slice/heads; head=slice-row_off*heads; }
}
__device__ __forceinline__ uint32_t norm_slices_per_block(
    const __nv_bfloat16* x,const __nv_bfloat16* w,const __nv_bfloat16* z,const __nv_bfloat16* y,
    uint32_t row_capacity,uint32_t heads,uint32_t width,
    uint64_t xs0,uint64_t xs1,uint64_t xs2,uint64_t ys0,uint64_t ys1,uint64_t ys2,
    uint64_t zs0,uint64_t zs1,uint64_t zs2,int mode) {
    if(xs2!=1u || ys2!=1u || (mode>=3 && zs2!=1u)) return 1u;
    uintptr_t ptrs=uintptr_t(x)|uintptr_t(y)|uintptr_t(w)|(mode>=3?uintptr_t(z):0u);
    uint64_t strides=xs0|xs1|ys0|ys1|(mode>=3?(zs0|zs1):0u);
    if(width==128u && ((ptrs&7u)==0u) && ((strides&3u)==0u)) return 8u;
    if(width==256u && ((ptrs&15u)==0u) && ((strides&7u)==0u)) return 8u;
    if(width==5120u && heads==1u && row_capacity>16u && ((ptrs&7u)==0u) && (((xs0|ys0|(mode>=3?zs0:0u))&3u)==0u)) return 4u;
    return 1u;
}
__device__ __forceinline__ float block_inv(float v,float* scratch,uint32_t width,int mode,float eps) {
    for(int delta=16;delta;delta/=2) v=__fadd_rn(v,__shfl_down_sync(0xffffffffu,v,delta));
    if(threadIdx.x%32==0) scratch[threadIdx.x/32]=v;
    __syncthreads();
    v=threadIdx.x<8?scratch[threadIdx.x]:0;
    if(threadIdx.x<32) {
        for(int delta=16;delta;delta/=2) v=__fadd_rn(v,__shfl_down_sync(0xffffffffu,v,delta));
        if(threadIdx.x==0) scratch[0]=norm_inv(v,width,mode,eps);
    }
    __syncthreads(); return scratch[0];
}
// mode: 0 Standard, 1 ZeroCentered, 2 BF16 L2 chain, 3 StandardGated, 4 ZeroCentered with in-place x += z.
extern "C" __global__ __launch_bounds__(256, 4) void q38_norm(
    __nv_bfloat16* x,const __nv_bfloat16* w,const __nv_bfloat16* z,__nv_bfloat16* y,
    const Q38SequenceMeta* meta,uint32_t row_base,uint32_t row_capacity,
    uint32_t heads,uint32_t width,
    uint64_t xs0,uint64_t xs1,uint64_t xs2,uint64_t ys0,uint64_t ys1,uint64_t ys2,
    uint64_t zs0,uint64_t zs1,uint64_t zs2,int mode,float eps) {
    q38_meta_guard(meta);
    uint32_t spb=norm_slices_per_block(x,w,z,y,row_capacity,heads,width,xs0,xs1,xs2,ys0,ys1,ys2,zs0,zs1,zs2,mode);
    uint64_t total_slices=uint64_t(row_capacity)*heads;
    uint64_t expected_blocks=(total_slices+spb-1u)/spb;
    if(blockDim.x!=256 || blockDim.y!=1 || blockDim.z!=1 || gridDim.y!=1 || gridDim.z!=1 ||
       !heads || !width || !row_capacity || uint64_t(gridDim.x)!=expected_blocks ||
       mode<0 || mode>4 || !x || !y || (mode!=2 && !w) || (mode>=3 && !z) || !(eps>0)) {__trap();return;}
    if(spb==8u && width==128u) {
        uint32_t warp=threadIdx.x>>5,lane=threadIdx.x&31u;
        uint64_t slice=uint64_t(blockIdx.x)*8u+warp;
        if(slice>=total_slices) return;
        uint32_t row_off,head;
        divmod_heads(uint32_t(slice),heads,row_off,head);
        uint64_t row=uint64_t(row_base)+row_off;
        if(row>=uint64_t(meta->valid_rows)) return;
        __nv_bfloat16* xs=x+row*xs0+head*xs1;
        __nv_bfloat16* ys=y+row*ys0+head*ys1;
        const __nv_bfloat16* zs=(mode>=3)?(z+row*zs0+head*zs1):nullptr;
        Bf16Vec4 vx; vx.u=reinterpret_cast<const uint2*>(xs)[lane];
        if(mode==4) {
            Bf16Vec4 vz; vz.u=reinterpret_cast<const uint2*>(zs)[lane];
            vx.h[0]=__hadd2(vx.h[0],vz.h[0]); vx.h[1]=__hadd2(vx.h[1],vz.h[1]);
            reinterpret_cast<uint2*>(xs)[lane]=vx.u;
        }
        float2 f0=__bfloat1622float2(vx.h[0]),f1=__bfloat1622float2(vx.h[1]);
        float r0=norm_term(f0.x,mode),r1=norm_term(f0.y,mode),r2=norm_term(f1.x,mode),r3=norm_term(f1.y,mode);
        #pragma unroll
        for(int delta=4;delta>=1;delta>>=1) {
            r0=__fadd_rn(r0,__shfl_down_sync(0xffffffffu,r0,delta));
            r1=__fadd_rn(r1,__shfl_down_sync(0xffffffffu,r1,delta));
            r2=__fadd_rn(r2,__shfl_down_sync(0xffffffffu,r2,delta));
            r3=__fadd_rn(r3,__shfl_down_sync(0xffffffffu,r3,delta));
        }
        float ww=__fadd_rn(__fadd_rn(r0,r2),__fadd_rn(r1,r3));
        ww=__fadd_rn(ww,__shfl_down_sync(0xffffffffu,ww,16));
        ww=__fadd_rn(ww,__shfl_down_sync(0xffffffffu,ww,8));
        float inv=(lane==0u)?norm_inv(ww,128u,mode,eps):0.0f;
        inv=__shfl_sync(0xffffffffu,inv,0);
        Bf16Vec4 vw,vz,vy;
        vw.u=(mode!=2)?reinterpret_cast<const uint2*>(w)[lane]:make_uint2(0u,0u);
        vz.u=(mode==3)?reinterpret_cast<const uint2*>(zs)[lane]:make_uint2(0u,0u);
        vy.e[0]=norm_scale(f0.x,inv,vw.e[0],vz.e[0],mode);
        vy.e[1]=norm_scale(f0.y,inv,vw.e[1],vz.e[1],mode);
        vy.e[2]=norm_scale(f1.x,inv,vw.e[2],vz.e[2],mode);
        vy.e[3]=norm_scale(f1.y,inv,vw.e[3],vz.e[3],mode);
        reinterpret_cast<uint2*>(ys)[lane]=vy.u;
        return;
    }
    if(spb==8u && width==256u) {
        uint32_t warp=threadIdx.x>>5,lane=threadIdx.x&31u;
        uint64_t slice=uint64_t(blockIdx.x)*8u+warp;
        if(slice>=total_slices) return;
        uint32_t row_off,head;
        divmod_heads(uint32_t(slice),heads,row_off,head);
        uint64_t row=uint64_t(row_base)+row_off;
        if(row>=uint64_t(meta->valid_rows)) return;
        __nv_bfloat16* xs=x+row*xs0+head*xs1;
        __nv_bfloat16* ys=y+row*ys0+head*ys1;
        const __nv_bfloat16* zs=(mode>=3)?(z+row*zs0+head*zs1):nullptr;
        Bf16Vec8 vx; vx.u=reinterpret_cast<const uint4*>(xs)[lane];
        if(mode==4) {
            Bf16Vec8 vz; vz.u=reinterpret_cast<const uint4*>(zs)[lane];
            #pragma unroll
            for(int k=0;k<4;++k) vx.h[k]=__hadd2(vx.h[k],vz.h[k]);
            reinterpret_cast<uint4*>(xs)[lane]=vx.u;
        }
        float a[8],r[8];
        #pragma unroll
        for(int k=0;k<4;++k) {
            float2 fk=__bfloat1622float2(vx.h[k]);
            a[2*k]=fk.x; a[2*k+1]=fk.y;
            r[2*k]=norm_term(fk.x,mode); r[2*k+1]=norm_term(fk.y,mode);
        }
        #pragma unroll
        for(int delta=2;delta>=1;delta>>=1) {
            #pragma unroll
            for(int k=0;k<8;++k) r[k]=__fadd_rn(r[k],__shfl_down_sync(0xffffffffu,r[k],delta));
        }
        float s4_0=__fadd_rn(r[0],r[4]),s4_1=__fadd_rn(r[1],r[5]),s4_2=__fadd_rn(r[2],r[6]),s4_3=__fadd_rn(r[3],r[7]);
        float ww=__fadd_rn(__fadd_rn(s4_0,s4_2),__fadd_rn(s4_1,s4_3));
        ww=__fadd_rn(ww,__shfl_down_sync(0xffffffffu,ww,16));
        ww=__fadd_rn(ww,__shfl_down_sync(0xffffffffu,ww,8));
        ww=__fadd_rn(ww,__shfl_down_sync(0xffffffffu,ww,4));
        float inv=(lane==0u)?norm_inv(ww,256u,mode,eps):0.0f;
        inv=__shfl_sync(0xffffffffu,inv,0);
        Bf16Vec8 vw,vz,vy;
        vw.u=(mode!=2)?reinterpret_cast<const uint4*>(w)[lane]:make_uint4(0u,0u,0u,0u);
        vz.u=(mode==3)?reinterpret_cast<const uint4*>(zs)[lane]:make_uint4(0u,0u,0u,0u);
        #pragma unroll
        for(int k=0;k<8;++k) vy.e[k]=norm_scale(a[k],inv,vw.e[k],vz.e[k],mode);
        reinterpret_cast<uint4*>(ys)[lane]=vy.u;
        return;
    }
    if(spb==4u && width==5120u) {
        uint64_t block_row_base=uint64_t(row_base)+uint64_t(blockIdx.x)*4u;
        if(block_row_base>=uint64_t(meta->valid_rows)) return;
        uint32_t row_slot=threadIdx.x>>6,tid64=threadIdx.x&63u,hw=tid64>>5,lane=tid64&31u;
        uint32_t slice=blockIdx.x*4u+row_slot;
        uint64_t row=uint64_t(row_base)+slice;
        bool active=(slice<row_capacity) && (row<uint64_t(meta->valid_rows));
        __nv_bfloat16* xs=active?(x+row*xs0):nullptr;
        __nv_bfloat16* ys=active?(y+row*ys0):nullptr;
        const __nv_bfloat16* zs=(active && mode>=3)?(z+row*zs0):nullptr;
        uint2 xv[20];
        float acc[4]={0.0f,0.0f,0.0f,0.0f};
        if(active) {
            #pragma unroll
            for(int c=0;c<20;++c) {
                uint32_t d4=uint32_t(c)*64u+tid64;
                Bf16Vec4 vx; vx.u=reinterpret_cast<const uint2*>(xs)[d4];
                if(mode==4) {
                    Bf16Vec4 vz; vz.u=reinterpret_cast<const uint2*>(zs)[d4];
                    vx.h[0]=__hadd2(vx.h[0],vz.h[0]); vx.h[1]=__hadd2(vx.h[1],vz.h[1]);
                    reinterpret_cast<uint2*>(xs)[d4]=vx.u;
                }
                xv[c]=vx.u;
                float2 f0=__bfloat1622float2(vx.h[0]),f1=__bfloat1622float2(vx.h[1]);
                acc[0]=__fadd_rn(acc[0],norm_term(f0.x,mode));
                acc[1]=__fadd_rn(acc[1],norm_term(f0.y,mode));
                acc[2]=__fadd_rn(acc[2],norm_term(f1.x,mode));
                acc[3]=__fadd_rn(acc[3],norm_term(f1.y,mode));
            }
        }
        #pragma unroll
        for(int delta=4;delta>=1;delta>>=1) {
            acc[0]=__fadd_rn(acc[0],__shfl_down_sync(0xffffffffu,acc[0],delta));
            acc[1]=__fadd_rn(acc[1],__shfl_down_sync(0xffffffffu,acc[1],delta));
            acc[2]=__fadd_rn(acc[2],__shfl_down_sync(0xffffffffu,acc[2],delta));
            acc[3]=__fadd_rn(acc[3],__shfl_down_sync(0xffffffffu,acc[3],delta));
        }
        float ww=__fadd_rn(__fadd_rn(acc[0],acc[2]),__fadd_rn(acc[1],acc[3]));
        __shared__ float smem[4][4];
        if(hw==1u && (lane&7u)==0u) smem[row_slot][lane>>3]=ww;
        __syncthreads();
        if(hw==0u) {
            if((lane&7u)==0u) ww=__fadd_rn(ww,smem[row_slot][lane>>3]);
            ww=__fadd_rn(ww,__shfl_down_sync(0xffffffffu,ww,16));
            ww=__fadd_rn(ww,__shfl_down_sync(0xffffffffu,ww,8));
            if(lane==0u) smem[row_slot][0]=norm_inv(ww,5120u,mode,eps);
        }
        __syncthreads();
        if(active) {
            float inv=smem[row_slot][0];
            #pragma unroll
            for(int c=0;c<20;++c) {
                uint32_t d4=uint32_t(c)*64u+tid64;
                Bf16Vec4 vx,vw,vz,vy;
                vx.u=xv[c];
                vw.u=(mode!=2)?reinterpret_cast<const uint2*>(w)[d4]:make_uint2(0u,0u);
                vz.u=(mode==3)?reinterpret_cast<const uint2*>(zs)[d4]:make_uint2(0u,0u);
                float2 x0=__bfloat1622float2(vx.h[0]),x1=__bfloat1622float2(vx.h[1]);
                vy.e[0]=norm_scale(x0.x,inv,vw.e[0],vz.e[0],mode);
                vy.e[1]=norm_scale(x0.y,inv,vw.e[1],vz.e[1],mode);
                vy.e[2]=norm_scale(x1.x,inv,vw.e[2],vz.e[2],mode);
                vy.e[3]=norm_scale(x1.y,inv,vw.e[3],vz.e[3],mode);
                reinterpret_cast<uint2*>(ys)[d4]=vy.u;
            }
        }
        return;
    }
    uint32_t row_off,head;
    divmod_heads(blockIdx.x,heads,row_off,head);
    uint64_t row=uint64_t(row_base)+row_off;
    if(row>=uint64_t(meta->valid_rows)) return;
    x+=row*xs0+head*xs1; y+=row*ys0+head*ys1;
    if(mode>=3) z+=row*zs0+head*zs1;
    __shared__ float tmp[8];
    if(width==5120u && xs2==1u && ys2==1u && (mode<3 || zs2==1u)) {
        float xv[20],sum=0.0f;
        #pragma unroll
        for(int c=0;c<20;++c) {
            uint32_t d=uint32_t(c)*256u+threadIdx.x;
            __nv_bfloat16 v=x[d];
            if(mode==4) { v=__hadd(v,z[d]); x[d]=v; }
            float a=__bfloat162float(v);
            xv[c]=a;
            sum=__fadd_rn(sum,norm_term(a,mode));
        }
        float inv=block_inv(sum,tmp,5120u,mode,eps);
        if(mode==1 || mode==4) {
            #pragma unroll
            for(int group=0;group<5;++group) {
                __nv_bfloat16 weights[4];
                #pragma unroll
                for(int j=0;j<4;++j) {
                    uint32_t d=uint32_t(4*group+j)*256u+threadIdx.x;
                    weights[j]=w[d];
                }
                #pragma unroll
                for(int j=0;j<4;++j) {
                    int c=4*group+j;
                    uint32_t d=uint32_t(c)*256u+threadIdx.x;
                    y[d]=norm_scale(xv[c],inv,weights[j],__float2bfloat16_rn(0.0f),1);
                }
            }
            return;
        }
        #pragma unroll
        for(int c=0;c<20;++c) {
            uint32_t d=uint32_t(c)*256u+threadIdx.x;
            __nv_bfloat16 wd=(mode!=2)?w[d]:__float2bfloat16_rn(0.0f);
            __nv_bfloat16 zd=(mode==3)?z[d]:__float2bfloat16_rn(0.0f);
            y[d]=norm_scale(xv[c],inv,wd,zd,mode);
        }
        return;
    }
    float sum=0.0f;
    for(uint32_t d=threadIdx.x;d<width;d+=256u) {
        __nv_bfloat16 v=x[uint64_t(d)*xs2];
        if(mode==4) { v=__hadd(v,z[uint64_t(d)*zs2]); x[uint64_t(d)*xs2]=v; }
        float a=__bfloat162float(v);
        sum=__fadd_rn(sum,norm_term(a,mode));
    }
    float inv=block_inv(sum,tmp,width,mode,eps);
    for(uint32_t d=threadIdx.x;d<width;d+=256u) {
        float a=__bfloat162float(x[uint64_t(d)*xs2]);
        __nv_bfloat16 wd=(mode!=2)?w[d]:__float2bfloat16_rn(0.0f);
        __nv_bfloat16 zd=(mode==3)?z[uint64_t(d)*zs2]:__float2bfloat16_rn(0.0f);
        y[uint64_t(d)*ys2]=norm_scale(a,inv,wd,zd,mode);
    }
}
// mode: 0 SiLU, 1 BF16 sigmoid(gate) followed by BF16 multiply,
// 2 BF16 SiLU(x) followed by BF16 multiply by `gate`.
extern "C" __global__ void q38_gate(
    const __nv_bfloat16* x,const __nv_bfloat16* gate,__nv_bfloat16* y,
    const Q38SequenceMeta* meta,uint32_t row_base,uint32_t row_capacity,
    uint32_t heads,uint32_t width,
    uint64_t xs0,uint64_t xs1,uint64_t xs2,uint64_t gs0,uint64_t gs1,uint64_t gs2,
    uint64_t ys0,uint64_t ys1,uint64_t ys2,int mode) {
    q38_meta_guard(meta);
    if(blockDim.x!=256 || blockDim.y!=1 || blockDim.z!=1 || (mode!=2 && gridDim.y!=1) || gridDim.z!=1 ||
       !heads || !width || !row_capacity || !x || !y || mode<0 || mode>2 || (mode!=0 && !gate)) {__trap();return;}
    if(mode==2) {
        const uint32_t vectors=width/8;
        if(heads!=1 || width%8 || xs0!=width || xs1!=width || xs2!=1 ||
           gs0!=width || gs1!=width || gs2!=1 || ys0!=width || ys1!=width || ys2!=1 ||
           gridDim.y!=row_capacity || uint32_t(gridDim.x)!=vectors/256+(vectors%256!=0) ||
           (reinterpret_cast<uintptr_t>(x)|reinterpret_cast<uintptr_t>(gate)|
            reinterpret_cast<uintptr_t>(y))%16) {__trap();return;}
        const uint64_t row=uint64_t(row_base)+blockIdx.y;
        if(row>=uint64_t(meta->valid_rows)) return;
        const uint32_t vector=uint32_t(blockIdx.x)*256+threadIdx.x;
        if(vector>=vectors) return;
        const __nv_bfloat16* xrow=x+row*width;
        const __nv_bfloat16* grow=gate+row*width;
        __nv_bfloat16* yrow=y+row*width;
        const uint4 a=reinterpret_cast<const uint4*>(xrow)[vector];
        const uint4 b=reinterpret_cast<const uint4*>(grow)[vector];
        const __nv_bfloat16* av=reinterpret_cast<const __nv_bfloat16*>(&a);
        const __nv_bfloat16* bv=reinterpret_cast<const __nv_bfloat16*>(&b);
        uint4 result;
        __nv_bfloat16* rv=reinterpret_cast<__nv_bfloat16*>(&result);
        #pragma unroll
        for(int i=0;i<8;++i)
            rv[i]=__hmul(__float2bfloat16_rn(silu(__bfloat162float(av[i]))),bv[i]);
        reinterpret_cast<uint4*>(yrow)[vector]=result;
        return;
    }
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
