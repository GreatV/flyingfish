// Chunkwise P2 split sizes, shared by kernels/gdn/gdn_cw_p2.cu (which static_asserts
// them) and the host planner (build.rs copies them into the generated constants).
// SMEM_BYTES is the s4 arena of gdn/gdn_chunkwise_p1: sA | sT | sV | sVn | sG | sBeta |
// sSc, 64*68*4 + 64*72*2 + 64*72*2 + 64*32*4 + 3*64*4.
#define GDN_CW_P2_SMEM_BYTES  44800
// Workspace bytes of one (chunk, head) slot: W bf16[64,128] 16384 + U fp32[64,128] 32768
// + qk bf16[64,72] 9216 + vn bf16[64,128] 16384 + Sn fp32[128,128] 65536 + the FP32[64]
// vectors Gc, Be, Sc 768.
#define GDN_CW_P2_SLOT_BYTES 141056
