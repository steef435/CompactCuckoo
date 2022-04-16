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
void fillClearyCuckoo(int n, uint64_t* vals, ClearyCuckoo* H)
{
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < n; i += stride) {
        printf("Value %i is %" PRIu64 "\n", i, vals[i]);
        H->insert(vals[i]);
        if (i == 9) {
            H->print();
        }
    }
}

//TODO Need to make this abstract
__global__
void fillCleary(int n, uint64_t* vals, Cleary* H)
{
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < n; i += stride) {
        printf("Value %i is %" PRIu64 "\n", i, vals[i]);
        H->insert(vals[i]);
        if (i == 9) {
            H->print();
        }
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

	fillClearyCuckoo << <1, 256 >> > (N, vals, cc);
    cudaDeviceSynchronize();
    printf("Devices Synced\n");
    //cc->print();

	//Create Table 2
    Cleary* c;
    cudaMallocManaged((void**)&c, sizeof(Cleary));
    new (c) Cleary(N);

    fillCleary << <1, 256 >> > (N, vals, c);
    cudaDeviceSynchronize();

    //Destroy Vars
    cudaFree(vals);
    cudaFree(cc);
    cudaFree(c);
}

__device__
void TestEntriesOnDevices(ClearyCuckooEntry<addtype, remtype>* entry1, ClearyCuckooEntry<addtype, remtype>* entry2) {
    entry1->setR(13456);
    entry2->setR(23);

    entry1->print();
    entry2->print();

    //entry1->exchValue(entry2);

    entry1->print();
    entry2->print();

    return;
}

__global__
void TestEntries(ClearyCuckooEntry<addtype, remtype>* entry1, ClearyCuckooEntry<addtype, remtype>* entry2) {
    

    TestEntriesOnDevices(entry1, entry2);
}

int main(void)
{
    printf("Starting\n");

	Test(10);

    return 0;
}