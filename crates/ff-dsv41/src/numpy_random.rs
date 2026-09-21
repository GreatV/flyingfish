//! Bit-exact reproduction of the numpy streams the engram layout depends on.
//!
//! `default_rng(seed).integers(0, bound, size, dtype=int64)` is SeedSequence
//! (numpy/random/bit_generator.pyx, v1.26) feeding PCG64
//! (numpy/random/src/pcg64) through Lemire's bounded sampler
//! (numpy/random/src/distributions/distributions.c). The trained engram
//! multipliers came from exactly this stream, so the reproduction has to be
//! bit-exact; the pinned values in the tests came from real numpy.

const POOL_SIZE: usize = 4;
const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;

fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut value = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    value = value.wrapping_mul(*hash_const);
    value ^= value >> XSHIFT;
    value
}

fn mix(x: u32, y: u32) -> u32 {
    let mut result = MIX_MULT_L
        .wrapping_mul(x)
        .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    result ^= result >> XSHIFT;
    result
}

/// The SeedSequence entropy pool for a single integer seed.
fn seed_pool(seed: u64) -> [u32; POOL_SIZE] {
    let entropy = [seed as u32, (seed >> 32) as u32];
    let mut pool = [0u32; POOL_SIZE];
    let mut hash_const = INIT_A;
    for (index, slot) in pool.iter_mut().enumerate() {
        *slot = hashmix(entropy.get(index).copied().unwrap_or(0), &mut hash_const);
    }
    for source in 0..POOL_SIZE {
        for destination in 0..POOL_SIZE {
            if source != destination {
                let mixed = hashmix(pool[source], &mut hash_const);
                pool[destination] = mix(pool[destination], mixed);
            }
        }
    }
    pool
}

/// The 64-bit seeding words `SeedSequence(seed).generate_state(n, uint64)`.
pub fn seed_sequence_words(seed: u64, words: usize) -> Vec<u64> {
    let pool = seed_pool(seed);
    let mut hash_const = INIT_B;
    let mut flat = Vec::with_capacity(words * 2);
    for index in 0..words * 2 {
        let mut data = pool[index % POOL_SIZE];
        data ^= hash_const;
        hash_const = hash_const.wrapping_mul(MULT_B);
        data = data.wrapping_mul(hash_const);
        data ^= data >> XSHIFT;
        flat.push(data);
    }
    flat.chunks_exact(2)
        .map(|pair| pair[0] as u64 | ((pair[1] as u64) << 32))
        .collect()
}

const PCG_MULTIPLIER: u128 = ((2549297995355413924u128) << 64) | 4865540595714422341u128;

pub struct Pcg64 {
    state: u128,
    increment: u128,
}

impl Pcg64 {
    pub fn new(seed: u64) -> Self {
        let words = seed_sequence_words(seed, 4);
        let initstate = ((words[0] as u128) << 64) | words[1] as u128;
        let initseq = ((words[2] as u128) << 64) | words[3] as u128;
        let mut generator = Self {
            state: 0,
            increment: (initseq << 1) | 1,
        };
        generator.step();
        generator.state = generator.state.wrapping_add(initstate);
        generator.step();
        generator
    }

    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(PCG_MULTIPLIER)
            .wrapping_add(self.increment);
    }

    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let high = (self.state >> 64) as u64;
        let low = self.state as u64;
        (high ^ low).rotate_right((high >> 58) as u32)
    }

    /// One draw from `Generator.integers(0, bound, dtype=int64)`: Lemire's
    /// method on the exclusive upper bound.
    pub fn next_bounded(&mut self, bound: u64) -> u64 {
        let rng = bound - 1;
        if rng == 0 {
            return 0;
        }
        let rng_excl = bound;
        let mut draw = self.next_u64();
        let mut low = (draw as u128).wrapping_mul(rng_excl as u128) as u64;
        if low < rng_excl {
            let threshold = rng_excl.wrapping_neg().wrapping_rem(rng_excl);
            while low < threshold {
                draw = self.next_u64();
                low = (draw as u128).wrapping_mul(rng_excl as u128) as u64;
            }
        }
        (((draw as u128).wrapping_mul(rng_excl as u128)) >> 64) as u64
    }
}

/// The engram hash multipliers for one layer: `default_rng(10007 * layer_id)`
/// drawing `max_ngram_size` values in `[0, bound)`, kept odd.
pub fn engram_multipliers(layer_id: usize, bound: u64, n: usize) -> Vec<u64> {
    let mut generator = Pcg64::new(10007 * layer_id as u64);
    (0..n)
        .map(|_| generator.next_bounded(bound) * 2 + 1)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_words_match_numpy_generate_state() {
        // SeedSequence(10007).generate_state(4, uint64)
        let words = seed_sequence_words(10007, 4);
        assert_eq!(words[0], 0x77d32a348ad70d07);
        assert_eq!(words[1], 0x993ed609b84a4fa5);
        assert_eq!(words[2], 0xd5f972dec8698e49);
        assert_eq!(words[3], 0x7d79ebaebcf148de);
        // SeedSequence(10007 * 14).generate_state(4, uint64)
        let words = seed_sequence_words(10007 * 14, 4);
        assert_eq!(words[0], 0xdb5869feeeccf2be);
        assert_eq!(words[3], 0x9f9e49df66ff1c85);
    }

    #[test]
    fn pcg64_matches_numpy_raw_draws() {
        let mut generator = Pcg64::new(10007);
        assert_eq!(generator.next_u64(), 0xd2c3f84026661e33);
        assert_eq!(generator.next_u64(), 0x0d4fb6969eb57435);
        assert_eq!(generator.next_u64(), 0x62e6e1c58c3bd72b);
    }

    #[test]
    fn bounded_draws_match_numpy_integers() {
        let bound: u64 = 46_539_438_283_891;
        let mut generator = Pcg64::new(10007);
        assert_eq!(generator.next_bounded(bound), 38_316_048_023_122);
        assert_eq!(generator.next_bounded(bound), 2_419_938_046_656);
        assert_eq!(generator.next_bounded(bound), 17_979_836_159_674);
    }

    #[test]
    fn engram_multipliers_match_the_pinned_training_values() {
        let bound = 46_539_438_283_891;
        let one = engram_multipliers(1, bound, 4);
        assert_eq!(
            one,
            [
                76_632_096_046_245,
                4_839_876_093_313,
                35_959_672_319_349,
                73_987_337_458_391
            ]
        );
        let fourteen = engram_multipliers(14, bound, 4);
        assert_eq!(
            fourteen,
            [
                67_716_810_739_261,
                51_510_806_800_915,
                30_921_347_202_721,
                82_619_226_485_591
            ]
        );
    }
}
