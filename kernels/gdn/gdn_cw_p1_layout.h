// Chunkwise P1 core: dynamic shared-memory bytes per value split s = DV/VW.
//
// The kernel includes this header and static_asserts its own arena size against
// these values; build.rs greps them into the Rust binder, so the launcher and the
// kernel derive the size from ONE definition (the same pattern gdn_b_layout.h uses).
// Protocol §E targets 18/14 KiB for the STATE pass alone; these are the FUSED
// chunk-body arenas, which also carry A/T/qk and the operand staging.
#define GDN_CW_P1_S4_BYTES  40704   // s=4,  VW=32
#define GDN_CW_P1_S8_BYTES  34560   // s=8,  VW=16
#define GDN_CW_P1_S16_BYTES 31488   // s=16, VW=8