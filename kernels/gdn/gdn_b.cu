// Experimental G4/W32 FP32 scan. No Engine dispatch or production build integration.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "gdn_b_layout.h"
using namespace gdn_b;

__device__ __forceinline__ bool b_aligned(const void* p) {
    return p && (reinterpret_cast<uintptr_t>(p)&15)==0;
}
__device__ __forceinline__ int b_rows(const Meta* meta, int bound, int leaves) {
    if (!b_aligned(meta) || bound<1 || bound>4096 || leaves<1 || leaves>32 ||
        (leaves&(leaves-1)) || bound>leaves*G*C ||
        (leaves>1 && bound<=leaves/2*G*C)) { __trap(); return 0; }
    const int rows=meta->valid_rows;
    if (rows<1 || rows>bound || meta->prefix<0 || meta->causal!=1 ||
        static_cast<long long>(meta->key_count)!=static_cast<long long>(meta->prefix)+rows ||
        meta->capacity<meta->key_count) { __trap(); return 0; }
    return rows;
}
__device__ __forceinline__ size_t map_at(int head,int leaves,int node,int ab) {
    return ((static_cast<size_t>(head)*(2*leaves-2)+(node-1))*2+ab)*D*D;
}
__device__ __forceinline__ size_t entry_at(int head,int leaves,int group) {
    return (static_cast<size_t>(head)*leaves+group)*D*V;
}

extern "C" __global__ __launch_bounds__(128, 1)
void gdn_b_p1(const float* k,const float* beta,const float* g,float* all_t,
              const Meta* meta,int bound,int leaves) {
    const int rows=b_rows(meta,bound,leaves), tid=threadIdx.x;
    const int head=blockIdx.y, chunk=blockIdx.x, row0=chunk*C;
    if (blockDim.x!=128 || blockDim.y!=1 || blockDim.z!=1 || gridDim.x!=unsigned(leaves*G) ||
        gridDim.y!=H || gridDim.z!=1 || !b_aligned(k) || !b_aligned(beta) ||
        !b_aligned(g) || !b_aligned(all_t)) { __trap(); return; }
    extern __shared__ float sm[];
    float* raw=sm+P1_K; float* atri=sm+P1_A;
    float* be=sm+P1_BETA; float* gs=sm+P1_G;
    for (int i=tid;i<C;i+=128) {
        be[i]=row0+i<rows ? beta[static_cast<size_t>(row0+i)*H+head] : 0.0f;
        gs[i]=row0+i<rows ? g[static_cast<size_t>(row0+i)*H+head] : 0.0f;
    }
    for (int i=tid;i<C*D;i+=128) {
        const int r=i/D,d=i%D;
        raw[d*(C+1)+r]=row0+r<rows ? k[(static_cast<size_t>(row0+r)*H+head)*D+d] : 0.0f;
    }
    __syncthreads();
    if (tid==0) for (int i=1;i<C;++i) gs[i]+=gs[i-1];
    __syncthreads();
    for (int i=tid;i<C*C;i+=128) {
        const int r=i/C,j=i%C; float dot=0.0f;
        if (j<r && be[r]!=0.0f) {
            for (int d=0;d<D;++d) dot=fmaf(raw[d*(C+1)+r],raw[d*(C+1)+j],dot);
            dot=-be[r]*dot*__expf(gs[r]-gs[j]);
        }
        atri[i]=dot;
    }
    __syncthreads(); // raw K is dead; its first C*C floats now hold T.
    float* t=raw;
    for (int i=tid;i<C*C;i+=128) t[i]=(i/C==i%C) ? 1.0f : 0.0f;
    __syncthreads();
    if (tid<32) {
        for (int r=1;r<C;++r) {
            if (tid<r) {
                float acc=atri[r*C+tid];
                for (int m=tid+1;m<r;++m) acc=fmaf(atri[r*C+m],t[m*C+tid],acc);
                t[r*C+tid]=acc;
            }
            __syncwarp();
        }
    }
    __syncthreads();
    const size_t base=(static_cast<size_t>(head)*leaves*G+chunk)*C*C;
    for (int i=tid;i<C*C;i+=128) all_t[base+i]=t[i];
}

extern "C" __global__ __maxnreg__(96)
void gdn_b_leaf(const float* k,const float* v,const float* beta,const float* g,
                const float* all_t,const float* state,float* maps,float* entries,
                const Meta* meta,int bound,int leaves) {
    const int rows=b_rows(meta,bound,leaves), tid=threadIdx.x;
    const int head=blockIdx.y, group=blockIdx.x/4, col=(blockIdx.x%4)*W+tid%32;
    const int warp=tid/32,lane=tid%32,i0=warp*16;
    if (blockDim.x!=256 || blockDim.y!=1 || blockDim.z!=1 || gridDim.x!=unsigned(leaves*4) ||
        gridDim.y!=H || gridDim.z!=1 || !b_aligned(k) || !b_aligned(v) ||
        !b_aligned(beta) || !b_aligned(g) || !b_aligned(all_t) || !b_aligned(state) ||
        (leaves>1 && !b_aligned(maps)) || !b_aligned(entries)) { __trap(); return; }
    extern __shared__ float sm[];
    float* u=sm+LEAF_U; float* n=sm+LEAF_N; float* za=sm+LEAF_T;
    float* zb=sm+LEAF_ZB; float* ra=sm+LEAF_RA; float* rb=sm+LEAF_RB;
    float* be=sm+LEAF_BETA; float* gs=sm+LEAF_G;
    float ar[16],br[16];
    #pragma unroll
    for (int a=0;a<16;++a) {
        ar[a]=(i0+a==col) ? 1.0f : 0.0f; br[a]=0.0f;
        // Snapshot before ANY P3 launch, partitioned by disjoint column CTAs.
        if (group==0) entries[entry_at(head,leaves,0)+(i0+a)*V+col]=
            state[static_cast<size_t>(head)*D*V+(i0+a)*V+col];
    }
    const int end=min((rows+C-1)/C,(group+1)*G);
    for (int chunk=group*G;chunk<end;++chunk) {
        const int row0=chunk*C;
        for (int r=tid;r<C;r+=256) {
            be[r]=row0+r<rows ? beta[static_cast<size_t>(row0+r)*H+head] : 0.0f;
            gs[r]=row0+r<rows ? g[static_cast<size_t>(row0+r)*H+head] : 0.0f;
        }
        for (int x=tid;x<C*D;x+=256) {
            const int r=x/D,d=x%D;
            n[x]=row0+r<rows ? k[(static_cast<size_t>(row0+r)*H+head)*D+d] : 0.0f;
        }
        for (int x=tid;x<C*C;x+=256)
            za[x]=all_t[(static_cast<size_t>(head)*leaves*G+chunk)*C*C+x];
        __syncthreads();
        if (tid==0) for (int r=1;r<C;++r) gs[r]+=gs[r-1];
        __syncthreads();
        // U=Ktail^T*T, stored [T-column,D]. K is still unscaled here.
        for (int x=tid;x<C*D;x+=256) {
            const int c=x/D,d=x%D; float acc=0.0f;
            for (int r=c;r<C;++r) acc=fmaf(n[r*D+d]*__expf(gs[C-1]-gs[r]),za[r*C+c],acc);
            u[x]=acc;
        }
        __syncthreads(); // all U readers have finished T; ZA may replace T.
        for (int x=tid;x<C*D;x+=256) n[x]*=be[x/D]*__expf(gs[x/D]);
        for (int x=tid;x<C*W;x+=256) {
            const int r=x/W,j=(blockIdx.x%4)*W+x%W;
            zb[x]=row0+r<rows ? be[r]*v[(static_cast<size_t>(row0+r)*H+head)*V+j] : 0.0f;
        }
        __syncthreads();
        const float alpha=__expf(gs[C-1]);
        if (chunk==group*G) {
            #pragma unroll
            for (int a=0;a<16;++a) ar[a]*=alpha;
            #pragma unroll 1
            for (int r=0;r<C;++r) {
                const float na=n[r*D+col], bv=zb[r*W+lane];
                #pragma unroll
                for (int a=0;a<16;++a) {
                    const float f=u[r*D+i0+a];
                    ar[a]=fmaf(-f,na,ar[a]); br[a]=fmaf(f,bv,br[a]);
                }
            }
        } else {
            for (int r=0;r<C;++r) {
                float av=0.0f,bv=0.0f;
                #pragma unroll
                for (int a=0;a<16;++a) {
                    av=fmaf(n[r*D+i0+a],ar[a],av);
                    bv=fmaf(n[r*D+i0+a],br[a],bv);
                }
                ra[warp*W+lane]=av; rb[warp*W+lane]=bv;
                __syncthreads();
                if (warp==0) {
                    float asum=0.0f,bsum=0.0f;
                    for (int w=0;w<8;++w) { asum+=ra[w*W+lane]; bsum+=rb[w*W+lane]; }
                    za[r*W+lane]=asum; zb[r*W+lane]-=bsum;
                }
                __syncthreads();
            }
            #pragma unroll
            for (int a=0;a<16;++a) { ar[a]*=alpha; br[a]*=alpha; }
            #pragma unroll 1
            for (int r=0;r<C;++r) {
                const float av=za[r*W+lane], bv=zb[r*W+lane];
                #pragma unroll
                for (int a=0;a<16;++a) {
                    const float f=u[r*D+i0+a];
                    ar[a]=fmaf(-f,av,ar[a]); br[a]=fmaf(f,bv,br[a]);
                }
            }
        }
        __syncthreads(); // all current U/N/Z readers finish before the next chunk load.
    }
    if (leaves>1) {
        const int node=leaves-1+group;
        #pragma unroll
        for (int a=0;a<16;++a) {
            maps[map_at(head,leaves,node,0)+(i0+a)*D+col]=ar[a];
            maps[map_at(head,leaves,node,1)+(i0+a)*D+col]=br[a];
        }
    }
}

extern "C" __global__ __maxnreg__(48)
void gdn_b_up(float* maps,const Meta* meta,int bound,int leaves,int depth) {
    const int rows=b_rows(meta,bound,leaves),tid=threadIdx.x;
    if (depth<1 || depth>4 || (1<<depth)>=leaves || blockDim.x!=256 || blockDim.y!=1 ||
        blockDim.z!=1 || gridDim.x!=unsigned((1<<depth)*16) || gridDim.y!=H || gridDim.z!=1 ||
        !b_aligned(maps)) { __trap(); return; }
    const int head=blockIdx.y,p=blockIdx.x/16,tile=blockIdx.x%16;
    const int node=(1<<depth)-1+p,left=2*node+1,right=left+1;
    const int r0=(tile/4)*32,c0=(tile%4)*32,warp=tid/32,lane=tid%32;
    const int span=leaves>>depth,live=(rows+G*C-1)/(G*C);
    const float* al=maps+map_at(head,leaves,left,0);
    const float* bl=maps+map_at(head,leaves,left,1);
    const float* ar=maps+map_at(head,leaves,right,0);
    const float* br=maps+map_at(head,leaves,right,1);
    float* oa=maps+map_at(head,leaves,node,0); float* ob=maps+map_at(head,leaves,node,1);
    // Entirely inactive -> identity. Inactive right -> exact left copy, no extra rounding.
    if (p*span+span/2>=live) {
        for (int x=tid;x<32*32;x+=256) {
            const int r=r0+x/32,c=c0+x%32;
            oa[r*D+c]=(p*span>=live) ? float(r==c) : al[r*D+c];
            ob[r*D+c]=(p*span>=live) ? 0.0f : bl[r*D+c];
        }
        return;
    }
    extern __shared__ float sm[];
    float* sr=sm; float* sl=sr+1024; float* sb=sl+1024;
    float aa[4],bb[4];
    #pragma unroll
    for (int a=0;a<4;++a) { aa[a]=0.0f; bb[a]=0.0f; }
    for (int k0=0;k0<D;k0+=32) {
        for (int x=tid;x<1024;x+=256) {
            const int r=x/32,c=x%32;
            sr[x]=ar[(r0+r)*D+k0+c]; sl[x]=al[(k0+r)*D+c0+c]; sb[x]=bl[(k0+r)*D+c0+c];
        }
        __syncthreads();
        #pragma unroll
        for (int a=0;a<4;++a) {
            #pragma unroll 4
            for (int t=0;t<32;++t) {
                const float f=sr[(warp*4+a)*32+t];
                aa[a]=fmaf(f,sl[t*32+lane],aa[a]); bb[a]=fmaf(f,sb[t*32+lane],bb[a]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int a=0;a<4;++a) {
        const int at=(r0+warp*4+a)*D+c0+lane;
        oa[at]=aa[a]; ob[at]=bb[a]+br[at];
    }
}

extern "C" __global__ __maxnreg__(48)
void gdn_b_down(const float* maps,float* entries,const Meta* meta,int bound,int leaves,int depth) {
    const int rows=b_rows(meta,bound,leaves),tid=threadIdx.x;
    if (depth<0 || depth>4 || (1<<depth)>=leaves || blockDim.x!=256 || blockDim.y!=1 ||
        blockDim.z!=1 || gridDim.x!=unsigned((1<<depth)*16) || gridDim.y!=H || gridDim.z!=1 ||
        !b_aligned(maps) || !b_aligned(entries)) { __trap(); return; }
    const int p=blockIdx.x/16,tile=blockIdx.x%16,head=blockIdx.y;
    const int span=leaves>>depth,lo=p*span,mid=lo+span/2;
    if (mid>=(rows+G*C-1)/(G*C)) return;
    const int node=(1<<depth)-1+p,left=2*node+1;
    const float* al=maps+map_at(head,leaves,left,0);
    const float* bl=maps+map_at(head,leaves,left,1);
    const float* in=entries+entry_at(head,leaves,lo);
    float* out=entries+entry_at(head,leaves,mid);
    const int r0=(tile/4)*32,c0=(tile%4)*32,warp=tid/32,lane=tid%32;
    extern __shared__ float sm[]; float* sa=sm; float* ss=sm+1024;
    float acc[4];
    #pragma unroll
    for (int a=0;a<4;++a) acc[a]=0.0f;
    for (int k0=0;k0<D;k0+=32) {
        for (int x=tid;x<1024;x+=256) {
            const int r=x/32,c=x%32;
            sa[x]=al[(r0+r)*D+k0+c]; ss[x]=in[(k0+r)*D+c0+c];
        }
        __syncthreads();
        #pragma unroll
        for (int a=0;a<4;++a) {
            #pragma unroll 4
            for (int t=0;t<32;++t) acc[a]=fmaf(sa[(warp*4+a)*32+t],ss[t*32+lane],acc[a]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int a=0;a<4;++a) {
        const int at=(r0+warp*4+a)*V+c0+lane;
        out[at]=acc[a]+bl[at];
    }
}

#include "gdn_b_p3.cuh"

// The binder runs this before conv/history, so an oversized replay cannot
// modify persistent history before the core notices the bucket mismatch.
extern "C" __global__ void gdn_b_guard(const Meta* meta,int bound,int leaves) {
    if (gridDim.x!=1 || gridDim.y!=1 || gridDim.z!=1 || blockDim.x!=1 || blockDim.y!=1 || blockDim.z!=1) {
        __trap(); return;
    }
    (void)b_rows(meta,bound,leaves);
}
