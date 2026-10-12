#pragma once
#include <cstddef>
namespace gdn_b {
constexpr int H=48, D=128, V=128, C=32, G=4, W=32;
constexpr int P1_K=0, P1_A=D*(C+1), P1_BETA=P1_A+C*C, P1_G=P1_BETA+C;
constexpr int P1_BYTES=(P1_G+C)*sizeof(float);
constexpr int LEAF_U=0, LEAF_N=C*D, LEAF_T=2*C*D, LEAF_ZB=LEAF_T+C*W;
constexpr int LEAF_RA=LEAF_ZB+C*W, LEAF_RB=LEAF_RA+8*W;
constexpr int LEAF_BETA=LEAF_RB+8*W, LEAF_G=LEAF_BETA+C;
constexpr int LEAF_BYTES=(LEAF_G+C)*sizeof(float);
constexpr int UP_BYTES=3*W*W*sizeof(float), DOWN_BYTES=2*W*W*sizeof(float);
constexpr int P3_BYTES=(C*D+2*C*V+C*C+D+2*C+16*V+C)*sizeof(float);
static_assert(P1_BYTES==21248 && LEAF_BYTES==43264);
static_assert(UP_BYTES==12288 && DOWN_BYTES==8192 && P3_BYTES==62336);
static_assert(P1_BYTES<=101376 && LEAF_BYTES<=101376 && P3_BYTES<=101376);
struct Meta { int prefix,valid_rows,key_count,capacity,causal,r20,r24,r28; };
struct Geometry {
    int bound, leaves, chunks;
    size_t t_floats, map_floats, entry_floats;
    explicit Geometry(int rows) : bound(rows), leaves(1) {
        // Standalone diagnostic upper bound. Production dispatch is intentionally absent.
        if (rows<1 || rows>4096) throw "GDN B diagnostic bound must be 1..4096";
        while (leaves*G*C<rows) leaves*=2;
        chunks=leaves*G;
        t_floats=static_cast<size_t>(H)*chunks*C*C;
        map_floats=static_cast<size_t>(H)*(2*leaves-2)*2*D*D;
        entry_floats=static_cast<size_t>(H)*leaves*D*V;
    }
    size_t bytes() const { return (t_floats+map_floats+entry_floats)*sizeof(float); }
};
}

#define GDN_B_P1_BYTES 21248
static_assert(gdn_b::P1_BYTES == GDN_B_P1_BYTES);
#define GDN_B_LEAF_BYTES 43264
static_assert(gdn_b::LEAF_BYTES == GDN_B_LEAF_BYTES);
#define GDN_B_UP_BYTES 12288
static_assert(gdn_b::UP_BYTES == GDN_B_UP_BYTES);
#define GDN_B_DOWN_BYTES 8192
static_assert(gdn_b::DOWN_BYTES == GDN_B_DOWN_BYTES);
#define GDN_B_P3_BYTES 62336
static_assert(gdn_b::P3_BYTES == GDN_B_P3_BYTES);
