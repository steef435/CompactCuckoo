#include <random>
#include <unordered_set>
#include <cuda.h>
#include <curand_kernel.h>

#ifndef MAIN
#define MAIN
#include "main.h"
#endif

#ifndef HASHINCLUDED
#define HASHINCLUDED
#include "hashfunctions.cu"
#endif

std::random_device rd_ng;
std::mt19937_64 e2_ng(rd_ng());

__host__ __device__
bool contains(uint64_cu* arr, uint64_cu val, int index) {
    for (int i = 0; i < index; i++) {
        if (val == arr[i]) {
            return true;
        }
    }
    return false;
}

uint64_cu* generateRandomSet(int size) {
    //Random Number generator
    std::uniform_int_distribution<long long int> dist(0, std::llround(std::pow(2, 58)));

#ifdef GPUCODE
    uint64_cu* res;
    gpuErrchk(cudaMallocManaged(&res, size * sizeof(uint64_cu)));
#else
    uint64_cu* res = new uint64_cu[size];
#endif
    std::unordered_set<uint64_cu> insertedSet;

    int i = 0;
    while (insertedSet.size() != size) {
        uint64_cu rand = dist(e2_ng);
        if (!(insertedSet.find(rand) != insertedSet.end())) {
            insertedSet.insert(rand);
            res[i] = rand;
            i++;
        }
    }
    return res;
}

uint64_cu* generateNormalSet(int size) {
    //Random Number generator
    std::normal_distribution<> dist{std::pow(2, 58)/2 , 1000 };

#ifdef GPUCODE
    uint64_cu* res;
    gpuErrchk(cudaMallocManaged(&res, size * sizeof(uint64_cu)));
#else
    uint64_cu* res = new uint64_cu[size];
#endif
    std::unordered_set<uint64_cu> insertedSet;

    int i = 0;
    while (insertedSet.size() != size) {
        uint64_cu rand = std::round(dist(e2_ng));
        if (!(insertedSet.find(rand) != insertedSet.end())) {
            insertedSet.insert(rand);
            res[i] = rand;
            i++;
        }
    }
    return res;
}

__global__
void setup_kernel(int seed, curandState* state) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__
void firstPassGenSet(curandState* state, uint64_cu* res, int N, int setsize, int begin) {
    int index = threadIdx.x;
    int stride = blockDim.x;

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curandState localState = state[idx];

    int maxVal = setsize + begin < N ? setsize + begin : N;

    for (int i = index + begin; i < maxVal; i += stride) {
        float myrandf = curand_uniform(&localState);
        uint64_cu newval = myrandf * std::llround(std::pow(2, 58));

        res[i] = newval;
    }
    return;
}

__global__
void secondPassGenSet(curandState* state, uint64_cu* res, int N, int setsize, int begin) {
    //printf("Setsize: %i Begin:%i\n", setsize, begin);

    int index = threadIdx.x;
    int stride = blockDim.x;

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curandState localState = state[idx];

    int maxVal = setsize + begin < N ? setsize + begin : N;
    //printf("MaxVal: %i\n", maxVal);

    for (int i = index + begin; i < maxVal; i += stride) {
        //printf("Index: %i\n", i);
        if (contains(res, res[i], i)) {
            while (true) {
                float myrandf = curand_uniform(&localState);
                uint64_cu newval = myrandf * std::llround(std::pow(2, 58));
                //Check if new in table
                if (!contains(res, newval, i)) {
                    res[i] = newval;
                    break;
                }
            }
        }
    }
    return;
}

uint64_cu* generateTestSetParallel(int size, int NUM_THREADS) {

    //Init States
    curandState* states;
    gpuErrchk(cudaMallocManaged(&states, sizeof(curandState) * NUM_THREADS));

    int setsize = 128;
    int split = (int)std::ceil((float)size / (float)setsize);
    split = split == 0 ? 1 : split;

    //Time For Seeding the Randomness
    const auto p1 = std::chrono::system_clock::now();
    setup_kernel << < 1, NUM_THREADS >> > (std::chrono::duration_cast<std::chrono::microseconds>(p1.time_since_epoch()).count(), states);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    //Result array
    uint64_cu* res;
    gpuErrchk(cudaMallocManaged(&res, size * sizeof(uint64_cu)));

    //Fill With Values
    for (int i = 0; i < split; i++) {
        firstPassGenSet << <1, NUM_THREADS >> > (states, res, size, setsize, i * setsize);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }

    //Check for Duplicates
    /*
    for (int i = 0; i < split; i++) {
        secondPassGenSet << <1, NUM_THREADS >> > (states, res, size, setsize, i * setsize);
        printf("New Set %i\n", i);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }*/

    return res;

}

uint64_cu* generateCollidingSet(int size, int N) {

#ifdef GPUCODE
    uint64_cu* res;
    gpuErrchk(cudaMallocManaged(&res, size * sizeof(uint64_cu)));
#else
    uint64_cu* res = new uint64_cu[size];
#endif

    uint64_cu add = 7;

    for (int n = 0; n < (int)size / 2; ++n) {
        uint64_cu num = reformKey(add, n, N);
        uint64_cu nval = RHASH_INVERSE(0, num);
        if (!contains(res, nval, n)) {
            res[n] = nval;
        }
        else {
            //Redo the step
            n--;
        }
    }

    add = 10;

    for (int n = ((int)size / 2); n < size; ++n) {
        uint64_cu num = reformKey(add, n, N);
        uint64_cu nval = RHASH_INVERSE(0, num);
        if (!contains(res, nval, n)) {
            res[n] = nval;
        }
        else {
            //Redo the step
            n--;
        }
    }

    return res;
}

uint64_cu* generateCollisionSet(int N, int AS, int H, int* hs, int percentage, int depth) {
    std::uniform_int_distribution<long long int> dist64(0, std::llround(std::pow(2, 58)));
    std::uniform_int_distribution<long long int> dist16(0, std::llround(std::pow(2, 16)));

    printf("\t\t\t\t\t\t\tgenerateCollisionSet N:%i H:%i perc:%i maxperc:%f\n", N, H, percentage, 100.0 / ((float)H));
    int maxPercentage = std::floor(100.0 / ((float)H));
    if (percentage > maxPercentage) {
        printf("Error: Percentage too Large - 1/H being used instead");
        percentage = maxPercentage;
    }

#ifdef GPUCODE
    uint64_cu* res;
    gpuErrchk(cudaMallocManaged(&res, N * sizeof(uint64_cu)));
#else
    uint64_cu* res = new uint64_cu[N];
#endif
    std::unordered_set<uint64_cu> insertedSet;

    int n = 0;

    if (percentage != 0) {
        for (int h = 0; h < H; h++) {
            //Generate Half the Set First
            int fullSet = (int)(((float)N * (float)percentage) / 100.0);
            int halfSet = std::floor(((float)N * (float)percentage) / (100.0 * ((float)depth + 1)));

            int start = n;
            int maxVal = N < n + halfSet ? N : n + halfSet;
            printf("\t\t\t\t\t\t\t\tGenerate First Set from %i to %i to %i\n", start, start + halfSet, start + fullSet);
            //Generate the First values
            for (int i = n; i < maxVal; i++) {
                uint64_cu rand = dist64(e2_ng);
                if (!(insertedSet.find(rand) != insertedSet.end())) {
                    //printf("\t\t\t\t\t\t\t\t\tInsertingVal1 at %i\n", i);
                    insertedSet.insert(rand);
                    res[i] = rand;
                    n++;
                }
                else {
                    //printf("\t\t\t\t\t\t\t\t\tAlready in Table1\n");
                    i--;
                }
            }

            int hash = hs[h];
            int outerLoop = std::floor((float)(fullSet - halfSet) / (float)depth);
            std::uniform_int_distribution<long long int> index(start, start + halfSet);
            //printf("OuterLoop: %i\n", outerLoop);
            //Generate the Second value set
            for (int i = 0; i < outerLoop; i++) {
                //Select a random index and get the address
                int r = index(e2_ng);
                uint64_cu hashed = RHASH(hash, res[r]);
                uint64_cu add = getAdd(hashed, AS);
                //printf("\t\t\t\t\t\t\t\t\t\tRetrieved Val from %i is %" PRIu64 " with add %" PRIu32 "\n", r, res[r], add);
                for (int j = 0; j < depth; j++) {
                    //Create a new value
                    uint64_cu newVal = reformKey(add, dist16(e2_ng), AS);
                    uint64_cu toInsert = RHASH_INVERSE(hash, newVal);

                    //printf("Trying Insert at %i\n", i);
                    //Check if value exists
                    if (!(insertedSet.find(toInsert) != insertedSet.end())) {
                        //printf("\t\t\t\t\t\t\t\t\tInsertingVal2 at %" PRIu64 "\n", n);
                        insertedSet.insert(toInsert);
                        res[n] = toInsert;
                        n++;
                    }
                    else {
                        //printf("\t\t\t\t\t\t\t\t\tAlready in Table2\n");
                        j--;
                    }
                }
            }
        }
    }

    for (int i = n; i < N; i++) {
        uint64_cu rand = dist64(e2_ng);
        if (!(insertedSet.find(rand) != insertedSet.end())) {
            printf("\t\t\t\t\t\t\t\t\tInsertingVal1 at %i\n", i);
            insertedSet.insert(rand);
            res[i] = rand;
        }
        else {
            printf("\t\t\t\t\t\t\t\t\tAlready in Table\n");
            i--;
        }
    }
    printf("\t\t\t\t\t\t\tgenerateCollisionSet Return\n");
    return res;
}