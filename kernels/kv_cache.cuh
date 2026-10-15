#pragma once
#include <cuda_bf16.h>
#include <cstdint>

#define KV_SINK_TOKENS 64
#define KV_RECENT_TOKENS 2048
#define KV_GROUP_TOKENS 128
#define KV_TENTATIVE_TOKENS 4

struct KvCacheView {
  unsigned char *k_code, *v_code;
  float *k_stats, *v_stats;
  __nv_bfloat16 *k_ring, *v_ring;
  int* cursor;
  int capacity, ring;
};
static_assert(sizeof(KvCacheView) == 64, "KV cache ABI");

__device__ __forceinline__ int kv_slot(int pos, int ring) {
  return pos < KV_SINK_TOKENS ? pos : KV_SINK_TOKENS + (pos - KV_SINK_TOKENS) % (ring - KV_SINK_TOKENS);
}
__device__ __forceinline__ int kv_sealed(int committed) {
  return max(0, committed - KV_RECENT_TOKENS) / KV_GROUP_TOKENS * KV_GROUP_TOKENS;
}
__device__ __forceinline__ __nv_bfloat16 kv_load(KvCacheView b, bool is_k,
                                                int h, int pos, int d) {
  if (pos < KV_SINK_TOKENS || pos >= b.cursor[1]) {
    const auto* ring = is_k ? b.k_ring : b.v_ring;
    return ring[((size_t)h * b.ring + kv_slot(pos,b.ring)) * 256 + d];
  }
  const unsigned char code = (is_k ? b.k_code : b.v_code)[((size_t)h*b.capacity+pos)*256+d];
  const float* stats = is_k ? b.k_stats : b.v_stats;
  const size_t i = is_k ? ((size_t)h*((b.capacity+(KV_GROUP_TOKENS-1))/KV_GROUP_TOKENS)+pos/KV_GROUP_TOKENS)*256+d
                       : (size_t)h*b.capacity+pos;
  const float x = __fadd_rn(__fmul_rn((float)code,stats[2*i+1]),stats[2*i]);
  return __float2bfloat16_rn(x);
}

__device__ __forceinline__ float kv_source(KvCacheView b, bool is_k, int h,
                                          int pos, int d, const __nv_bfloat16* full) {
  const int slot = full ? pos : kv_slot(pos,b.ring);
  const int cap = full ? b.capacity : b.ring;
  const auto* source = full ? full : (is_k ? b.k_ring : b.v_ring);
  const float x=__bfloat162float(source[((size_t)h*cap+slot)*256+d]);
  if (!isfinite(x)) asm volatile("trap;");
  return x;
}

__device__ __forceinline__ void kv_pack_token(KvCacheView b, bool is_k, int h,
                                             int pos, const __nv_bfloat16* full) {
  float lo=3.402823466e38F,hi=-3.402823466e38F;
  for(int d=0;d<256;++d) {
    const float x=kv_source(b,is_k,h,pos,d,full);
    lo=fminf(lo,x);hi=fmaxf(hi,x);
  }
  const float span=__fsub_rn(hi,lo);
  const float scale=span==0 ? 1.0f : __fdiv_rn(span,255.0f);
  if(!isfinite(scale) || scale<=0) asm volatile("trap;");
  float* stats=is_k ? b.k_stats : b.v_stats;
  const size_t i=(size_t)h*b.capacity+pos;
  stats[2*i]=lo;stats[2*i+1]=scale;
  auto* codes=is_k ? b.k_code : b.v_code;
  for(int d=0;d<256;++d) {
    const float x=kv_source(b,is_k,h,pos,d,full);
    const unsigned char code=(unsigned char)__float2uint_rn(fminf(255.0f,fmaxf(0.0f,__fdiv_rn(__fsub_rn(x,lo),scale))));
    codes[i*256+d]=code;
  }
}

extern "C" __global__ void kv_cache_pack(KvCacheView b, const __nv_bfloat16* full_k,
                                           const __nv_bfloat16* full_v,
                                           const int4* meta, int force) {
  if (!full_k && b.cursor[2] && !force) return;
  const int end=full_k ? meta[0].z : meta[0].x;
  if(end < b.cursor[0] || end > b.capacity) asm volatile("trap;");
  const int start=b.cursor[1]+blockIdx.x*KV_GROUP_TOKENS;
  if(start>=kv_sealed(end)) return;
  const int h=blockIdx.y,d=threadIdx.x;
  {
    float lo=3.402823466e38F,hi=-3.402823466e38F;
    for(int r=0;r<KV_GROUP_TOKENS;++r) {
      const float x=kv_source(b,true,h,start+r,d,full_k);
      lo=fminf(lo,x);hi=fmaxf(hi,x);
    }
    const float span=__fsub_rn(hi,lo),scale=span==0 ? 1.0f : __fdiv_rn(span,255.0f);
    if(!isfinite(scale) || scale<=0) asm volatile("trap;");
    const size_t i=((size_t)h*((b.capacity+(KV_GROUP_TOKENS-1))/KV_GROUP_TOKENS)+start/KV_GROUP_TOKENS)*256+d;
    b.k_stats[2*i]=lo;b.k_stats[2*i+1]=scale;
    for(int r=0;r<KV_GROUP_TOKENS;++r) {
      const float x=kv_source(b,true,h,start+r,d,full_k);
      b.k_code[((size_t)h*b.capacity+start+r)*256+d]=(unsigned char)__float2uint_rn(fminf(255.0f,fmaxf(0.0f,__fdiv_rn(__fsub_rn(x,lo),scale))));
    }
  }
  if(d<KV_GROUP_TOKENS) kv_pack_token(b,false,h,start+d,full_v);
}

extern "C" __global__ void kv_cache_tail(KvCacheView b, const __nv_bfloat16* full_k,
                                           const __nv_bfloat16* full_v,const int4* meta) {
  const int end=meta[0].z,start=kv_sealed(end);
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=4*b.ring*256) return;
  const int d=i%256,slot=(i/256)%b.ring,h=i/(256*b.ring);
  int pos=slot;
  if(slot>=KV_SINK_TOKENS) {
    const int last=end-1;
    if(last<KV_SINK_TOKENS) return;
    pos=last-(kv_slot(last,b.ring)-slot+b.ring-KV_SINK_TOKENS)%(b.ring-KV_SINK_TOKENS);
    if(pos<max(start,KV_SINK_TOKENS)) return;
  } else if(pos>=end) return;
  b.k_ring[i]=full_k[((size_t)h*b.capacity+pos)*256+d];
  b.v_ring[i]=full_v[((size_t)h*b.capacity+pos)*256+d];
}

extern "C" __global__ void kv_cache_cursor(KvCacheView b,const int4* meta,int mode) {
  if(mode==1) {if(b.cursor[0]!=meta[0].x) asm volatile("trap;");b.cursor[2]=1;return;}
  if(mode==0 && b.cursor[2]) return;
  const int end=mode==3 ? meta[0].z : meta[0].x;
  b.cursor[0]=end;b.cursor[1]=kv_sealed(end);b.cursor[2]=0;b.cursor[3]=0;
}

extern "C" __global__ void kv_cache_dequant(KvCacheView b,__nv_bfloat16* k,
                                              __nv_bfloat16* v,const int4* meta) {
  const size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  const int end=meta[0].x;
  if(i>=(size_t)4*end*256) return;
  const int d=i%256,pos=(i/256)%end,h=i/(256*end);
  const size_t dest=((size_t)h*b.capacity+pos)*256+d;
  k[dest]=kv_load(b,true,h,pos,d);v[dest]=kv_load(b,false,h,pos,d);
}
