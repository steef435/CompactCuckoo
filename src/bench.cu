#include "CuckooGeneric.cuh"
#include "numbergenerators.cu"
#include <algorithm>
#include <chrono>
#include <iostream>

namespace chrono = std::chrono;

template <bool compact, typename row_type, unsigned key_width, unsigned bucket_size>
void bench_cuckoo(const CuckooConfig config, const uint64_cu *batch) {
    const auto BLOCK_SIZE = 512;
    const size_t to_insert = config.n_rows;
    const auto MAX_BLOCKS = to_insert / 512;
    const auto BLOCK_INCS = MAX_BLOCKS / 10;

    using Table = CuckooGeneric<compact, row_type, key_width, bucket_size>;
    Table *table;
    gpuErrchk(cudaMallocManaged(&table, sizeof(*table)));
    new (table) Table(config);

    result *results;
    gpuErrchk(cudaMallocManaged(&results, sizeof(*results) * config.n_rows));
    
    chrono::steady_clock clock;
    for (auto bs = BLOCK_INCS; bs < MAX_BLOCKS; bs += BLOCK_INCS) {
        // Warmup
        clear<<<bs, BLOCK_SIZE>>>(table);
        readEverything<<<bs, BLOCK_SIZE>>>(table);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        auto before = clock.now();
        insertBatch<<<bs, BLOCK_SIZE>>>(table, batch, to_insert, results);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        auto after = clock.now();

        auto duration_ns = chrono::duration_cast<chrono::nanoseconds>(after - before).count();
        std::cout << "\t" << duration_ns;
        if (hasFailed(results, to_insert)) {
            std::cout << "(failed)";
        }
    }
    std::cout<<std::endl;
    
    cudaFree(table);
    cudaFree(results);
}

void bench_insertion() {
    const auto key_width = 32;
    const auto table_sizes_log = {21, 22, 23, 24, 25, 26}; // table sizes in log(rows)

    // Allocate memory, generate random keys
    const auto max_rows = 1ull << *std::max_element(table_sizes_log.begin(), table_sizes_log.end());
    // TODO: I think there might be an issue in generateRandomSet: type parameter to rng is long long INT
    uint64_cu *batch = generateRandomSet(max_rows, (1ull << key_width) - 1);
    
    // Run benchmarks
    for (auto N : table_sizes_log) {
        const CuckooConfig config = {.n_rows = 1ull << N};
        bench_cuckoo<false, uint64_cu, key_width, 16>(config, batch);
        bench_cuckoo<true, uint32_t, key_width, 32>(config, batch);
    }
    
    cudaFree(batch);
}

int main() {
    bench_insertion();
}