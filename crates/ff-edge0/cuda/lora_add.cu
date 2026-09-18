// y += B·(A·x): the on-the-fly LoRA delta. ax is computed warp-per-row
// with coalesced streaming plus a cross-warp shared reduction — the naive
// one-thread-per-row form is a 16-lane gather that the LSU serializes
// (measured linear in in_dim: 512/2048/8192 -> 15/52/197 us; unrolling
// changed nothing). Summation order differs from the host loop at ULP
// level; acceptance is CPU-vs-GPU token identity, not bitwise equality.
extern "C" __global__ void edge0_lora_add(
    const float* __restrict__ a,   // [rank, in_dim]
    const float* __restrict__ b,   // [out_dim, rank]
    const float* __restrict__ x,   // [in_dim]
    float* __restrict__ y,         // [out_dim]
    int rank,
    int in_dim,
    int out_dim)
{
    __shared__ float red[32][8];
    __shared__ float ax[32];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps = blockDim.x >> 5;
    // Zero first: with rank < warps some red[k][w] slots are never written
    // but all 8 are summed below.
    for (int i = tid; i < 32 * 8; i += blockDim.x) ((float*)red)[i] = 0.0f;
    __syncthreads();
    for (int k = warp; k < rank; k += warps) {
        const float* ar = a + (long long)k * in_dim;
        float acc = 0.0f;
        for (int c = lane; c < in_dim; c += 32) acc += ar[c] * x[c];
        // Reduce the 32 per-lane partials before the leader writes — the
        // original stored lane 0's partial alone (~1/32 of the dot).
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) red[k][warp] = acc;
    }
    __syncthreads();
    for (int k = tid; k < rank; k += blockDim.x) {
        float acc = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; w++) acc += red[k][w];
        ax[k] = acc;
    }
    __syncthreads();
    for (long long row = (long long)blockIdx.x * blockDim.x + tid;
         row < out_dim;
         row += (long long)gridDim.x * blockDim.x)
    {
        const float* br = b + row * rank;
        float acc = 0.0f;
        for (int k = 0; k < rank; k++) acc += br[k] * ax[k];
        y[row] += acc;
    }
}
