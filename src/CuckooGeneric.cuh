#pragma once

#include <cstdint>
#include <cooperative_groups.h>
#include "main.h"
#include "bit_width.h"
#include "hashfunctions.cu"

namespace cg = cooperative_groups;
template <unsigned tile_size>
using Tile = cg::thread_block_tile<tile_size, cg::thread_block>;

// Configuration for CuckooGeneric
struct CuckooConfig {
    //unsigned bucket_size = 32;
    //unsigned key_width = 45;
    unsigned n_rows; // we assume multiples of 32
    unsigned max_loops = 20; // TODO: is this a sensible number?
    unsigned n_hash_functions = 4;
};

// A Cuckoo hash table with support for bucketing and compactification
// 
// Entries (rows) are stored as follows:
// - the most significant bit is 1 iff the row is occupied
// - the next hashid_width bits (see below) indicate the hash function used
// - the rest represents the remainder, starting from the least significant bit.
// 
// If compact is false, the whole hashed key is stored as the remainder.
//
// Partly configured by template arguments, partly by the given CuckooConfig.
// (Template arguments benefit more from compiler optimization but are hard to vary.)
//
// TODO:
// - Currently we assume keys are originally uint64_t. Could be templated.
template <bool compact, typename row_type, unsigned key_width, unsigned bucket_size>
class CuckooGeneric {
    static_assert(bucket_size <= 32 && bucket_size % 2 == 0);
    // TODO: this is the wrong place to check this, see RHASH() in hashfunctions.cu
    static_assert(key_width == 64 || key_width == 50 || key_width == 32 || key_width == 28);

public:
    const CuckooConfig config;

    // Storage space constants in bits (see constructor)
    const unsigned row_width = sizeof(row_type) * 8;
    const unsigned addr_width;
    const unsigned rem_width; // for non-compact, this is key_width
    const unsigned hashid_width;
    const unsigned addr_space = ((uint64_t) 1) << addr_width;
    
    // The table
    row_type *table;
    
    // TODO: these inline functions might be better written on one line or as macros for readability

    __device__ inline uint64_t addr(uint64_t hashed_key) {
        return hashed_key % addr_space;
    }
    
    __device__ inline uint64_t rem(uint64_t hashed_key) {
        if constexpr (compact) return hashed_key / addr_space; else return hashed_key;
    }
    
    __device__ inline uint64_t combine(uint64_t addr, uint64_t rem) {
        if constexpr (compact) return rem * addr_space + addr; else return rem;
    }
    
    __device__ inline bool entry_is_occupied(row_type e) {
        return e >> (row_width - 1);
    }
    
    __device__ inline uint64_t entry_get_rem(row_type e) {
        return e & (1ull << rem_width - 1);
    }
    
    __device__ inline unsigned entry_get_hashid(row_type e) {
        return (e >> (row_width - hashid_width - 1)) & (1ull << hashid_width - 1);
    }
    
    __device__ inline uint64_t hash(unsigned hashid, uint64_t key) {
        return RHASH(key_width, hashid, key);
    }
    
    __device__ inline uint64_t hash_invert(unsigned hashid, uint64_t hashed_key) {
        return RHASH_INVERSE(key_width, hashid, hashed_key);
    }
    
    __device__ inline row_type entry_prepare_for_storage(uint64_t hashed_k, unsigned hashid) {
        row_type e = 1ull << (row_width - 1); // occupied
        e |= ((row_type)hashid) << (row_width - hashid_width - 1); // hash id
        if constexpr (compact) e |= rem(hashed_k); else e |= hashed_k; // remainder
        return e;
    }
    
    // Cooperatively look up a key
    //
    // Gives up if it encounters a bucket that is not full and does not contain k.
    __device__ bool coopLookup(uint64_t k, Tile<bucket_size> tile) {
        const auto rank = tile.thread_rank();
        for (auto i = 0; i < config.n_hash_functions; i++) {
            const auto s = hash(i, k);
            const auto a = addr(s), r = rem(s);
            const auto entry = table[a * bucket_size + rank];
            const bool found = entry_prepare_for_storage(s, i) == entry;
            if (tile.ballot(found)) return true;
            const auto load = __popc(tile.ballot(entry_is_occupied(entry)));
            if (load < bucket_size) return false;
        }
        return false;
    }
    
    // Cooperatively find or put a single key k (as in the article)
    //
    // Returns true if the key was found, false if it has been inserted.
    // If naive is true, the algorithm does not check whether k is already in the table.
    // (This can be used for raw throughput benchmarks.)
    //
    // False negatives and duplicate insertions may occur.
    //
    // TODO: bug in paper: in the XCHG it assumes j is the same but it might not be?
    // so if r' = r it might still be that the hash ids do not match
    //
    // TODO: cuckoor ID must be randomized more to allow for more deduplication
    template <bool naive = false>
    __device__ result coopFindOrPut(uint64_t k, Tile<bucket_size> tile) {
        const auto rank = tile.thread_rank();
        auto x = k;
        auto loop = 0, hashid = 0;
        if constexpr (!naive) if (coopLookup(k, tile)) return FOUND;
        
        while (true) {
            const auto s = hash(hashid, x), r = rem(s), a = addr(s);
            const auto entry = entry_prepare_for_storage(s, hashid);
            assert(entry >> (row_width - 1) == 1);

            const bool occupied = entry_is_occupied(table[a * bucket_size + rank]);
            const auto load = __popc(tile.ballot(occupied));
            if (load == bucket_size) {
                if (loop == config.max_loops) return FAILED;

                // DIVERSION: use hash(hashid, x) instead of hash(0, x),
                // in order to hopefully remove more duplicates.
                // (Awad et al. even use RNG.)
                const auto cuckoor = hash(hashid, x) % bucket_size;
                auto cuckood = entry;
                if (rank == cuckoor) {
                    atomicExch(&table[a * bucket_size + rank], cuckood);
                }
                tile.shfl(cuckood, cuckoor);
                
                // Detect duplicates in bucket (slight diversion from paper)
                // Maybe we should also find duplicates with differing hashes
                if constexpr(!naive) if (entry == cuckood) return FOUND;
                
                hashid = entry_get_hashid(cuckood);
                const auto rc = entry_get_rem(cuckood);
                x = hash_invert(hashid, combine(a, rc));
                hashid = hashid + 1 % config.n_hash_functions;
                loop++;
            } else {
                row_type old;
                if (rank == load) {
                    old = atomicCAS(&table[a * bucket_size + rank], 0, entry);
                }
                old = tile.shfl(old, load);
                if (old == 0) return INSERTED;
            }
        }
    }
    
    CuckooGeneric(CuckooConfig config)
        : config(config)
        , addr_width(bit_width(config.n_rows - 1) - bit_width(bucket_size - 1))
        , rem_width(compact ? key_width - addr_width : key_width)
        , hashid_width(bit_width(config.n_hash_functions))
    {
        // Make sure it fits
        assert(1 + hashid_width + rem_width <= sizeof(row_type) * 8);
        gpuErrchk(cudaMallocManaged(&table, config.n_rows * sizeof(row_type)));
        for (unsigned i = 0; i < config.n_rows; i++) table[i] = 0;
    }
    
    ~CuckooGeneric() {
        gpuErrchk(cudaFree(table));
    }
};

// Lookup
//
// Assumption: the size of batch is the number of threads * bucket_size
template <auto C, typename R, auto W, auto B>
__global__ void lookup(CuckooGeneric<C, R, W, B> *table, uint64_t *batch, bool *results) {
    const auto thb = cg::this_thread_block();
    const auto tile = cg::tiled_partition<B>(thb);
    const auto index = (blockIdx.x * blockDim.x + threadIdx.x) / B;
    results[index] = table->coopLookup(batch[index], tile);
}

// Naive insert (without checking if key is already in the table)
//
// Assumption: the size of batch is the number of threads * bucket_size
template <auto C, typename R, auto W, auto B>
__global__ void insert(CuckooGeneric<C, R, W, B> *table, uint64_t *batch, bool *results) {
    const auto thb = cg::this_thread_block();
    const auto tile = cg::tiled_partition<B>(thb);
    const auto index = (blockIdx.x * blockDim.x + threadIdx.x) / B;
    results[index] = table->coopFindOrPut<true>(batch[index], tile);
}

// findOrPut
//
// Assumption: the size of batch is the number of threads * bucket_size
template <auto C, typename R, auto W, auto B>
__global__ void findOrPut(CuckooGeneric<C, R, W, B> *table, uint64_t *batch, result *results) {
    const auto thb = cg::this_thread_block();
    const auto tile = cg::tiled_partition<B>(thb);
    const auto index = (blockIdx.x * blockDim.x + threadIdx.x) / B;
    results[index] = table->coopFindOrPut(batch[index], tile);
}

// Read everything (for warmup)
template <auto C, typename R, auto W, auto B>
__global__ void readEverything(CuckooGeneric<C, R, W, B> *table) {
    const auto index = blockIdx.x *blockDim.x + threadIdx.x;
    const auto stride = blockDim.x * gridDim.x;

    int val = 0;
    for (auto i = index; i < table->config->n_rows; i += stride) {
        val += table->readIndex(i);
    }
}