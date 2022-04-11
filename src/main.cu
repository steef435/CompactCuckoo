#include <iostream>
#include <random>
#include <algorithm>
#include <fstream>
#include <inttypes.h>

#ifndef HASHTABLE
#define HASHTABLE
#include "HashTable.h"
#endif

#ifndef HASHINCLUDED
#define HASHINCLUDED
#include "hashfunctions.cu"
#endif

#include "ClearyCuckoo.cu"
#include "Cleary.cu"

/*
 *
 *  Global Variables
 *
 */

std::random_device rd;
std::mt19937_64 e2(rd());
std::mt19937 g(rd());

/*
 *
 *	Helper Functions
 * 
 */

bool contains(uint64_t* arr, uint64_t val, int index) {
    for (int i = 0; i < index; i++) {
        if (val == arr[i]) {
            return true;
        }
    }
    return false;
}

uint64_t* generateTestSet(int size) {
    //Random Number generator
    std::uniform_int_distribution<long long int> dist(0, std::llround(std::pow(2, 58)));

    uint64_t* res;
    cudaMallocManaged(&res, size * sizeof(uint64_t));

    for (int n = 0; n < size; ++n) {
        uint64_t rand = dist(e2);
        if (!contains(res, rand, n)) {
            res[n] = rand;
        }
        else {
            //Redo the step
            n--;
        }
    }

    return res;
}

/*
 *
 * Main Functions
 *
 */


//TODO Need to make this abstract
__global__
void fillTable(int n, uint64_t* vals, ClearyCuckoo* H)
{
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < n; i += stride) {
        printf("Value %i is %" PRIu64 "\n", i, vals[i]);
        H->debug();
    }
}


void Test(int N) {
	//Create List of Values
	uint64_t* vals;
    vals = generateTestSet(N);

	//Create Table 1
    ClearyCuckoo* cc;
    cudaMallocManaged((void**)&cc, sizeof(ClearyCuckoo));
    new (cc) ClearyCuckoo();
    cc->ClearyCuckooInit(N, 4);


	fillTable << <1, 256 >> > (N, vals, cc);
    cudaDeviceSynchronize();
    cc->print();

	//Create Table 2


    //Destroy Vars
    cudaFree(vals);
}

int main(void)
{
    printf("Starting\n");
	Test(10);
	//Benchmark();

    return 0;
}