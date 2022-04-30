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

__host__ __device__
uint64_t reformKey(addtype add, remtype rem, int N) {
    rem = rem << N;
    rem += add;
    return rem;
}

uint64_t* generateCollidingSet(int size, int N) {
    uint64_t* res;
    cudaMallocManaged(&res, size * sizeof(uint64_t));

    uint64_t add = 7;

    for (int n = 0; n < (int) size/2; ++n) {
        uint64_t num = reformKey(add, n, N);
        uint64_t nval = RHASH_INVERSE(0, num);
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
        uint64_t num = reformKey(add, n, N);
        uint64_t nval = RHASH_INVERSE(0, num);
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

/*
 *
 * Main Functions
 *
 */

__global__
void fillClearyCuckoo(int N, uint64_t* vals, ClearyCuckoo* H)
{   
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < N; i += stride) {
        H->insert(vals[i]);
        //H->print();
    }
}

__global__
void fillCleary(int N, uint64_t* vals, Cleary* H)
{
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < N; i += stride) {
        H->insert(vals[i]);
    }
}

__global__
void checkClearyCuckoo(int N, uint64_t* vals, ClearyCuckoo* H, bool* res)
{
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < N; i += stride) {
        if (H->lookup(vals[i])) {
            *res = false;
        }
    }
}

__global__
void checkCleary(int N, uint64_t* vals, Cleary* H, bool* res)
{
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < N; i += stride) {
        if (H->lookup(vals[i])) {
            *res = false;
        }
    }
}


void TestFill(int N, int tablesize, uint64_t* vals) {
	//Create Table 1
    ClearyCuckoo* cc;
    cudaMallocManaged((void**)&cc, sizeof(ClearyCuckoo));
    new (cc) ClearyCuckoo(tablesize, 16);

    printf("Filling ClearyCuckoo\n");
	fillClearyCuckoo << <1, 256 >> > (N, vals, cc);
    cudaDeviceSynchronize();
    printf("Devices Synced\n");
    cc->print();

    //Checking 
    bool* res;
    cudaMallocManaged(&res, sizeof(bool));
    *res = true;

    checkClearyCuckoo << <1, 256 >> > (N, vals, cc, res);
    cudaDeviceSynchronize();
    printf("Devices Synced\n");
    if (res) {
        printf("All still in the table\n");
    }
    else {
        printf("Vals Missing\n");
    }

	//Create Table 2
    Cleary* c;
    cudaMallocManaged((void**)&c, sizeof(Cleary));
    new (c) Cleary(tablesize);

    printf("Filling Cleary\n");
    fillCleary << <1, 256 >> > (N, vals, c);
    cudaDeviceSynchronize();
    printf("Devices Synced\n");
    c->print();

    //Checking 
    *res = true;
    checkCleary << <1, 256 >> > (N, vals, c, res);
    cudaDeviceSynchronize();
    printf("Devices Synced\n");
    if (res) {
        printf("All still in the table\n");
    }
    else {
        printf("Vals Missing\n");
    }

    //Destroy Vars
    cudaFree(vals);
    cudaFree(cc);
    cudaFree(c);
}


__global__
void lockTestDevice(ClearyEntry<addtype, remtype>* T){
    addtype left = 1;
    addtype right = 4;

    while (true) {
        printf("\tGetting First Lock\n");
        if (!T[left].lock()) {
            printf("\tFirst Lock Failed\n");
                continue;
        }

        printf("\tLeft");
        T[left].print();

        printf("\tGetting Second Lock\n");
        if (!T[right].lock()) {
            printf("\tSecond Lock Failed\n");
                printf("\tAbort Locking\n");
            T[left].unlock();
            printf("\tUnlocked\n");
                continue;
        }

        printf("\tRight");
        T[left].print();

        printf("\t'Insertion\' Succeeded\n");
        T[left].unlock();
        T[right].unlock();
        printf("\tUnlocked\n");

        printf("\tLeft");
        T[left].print();
        printf("\tRight");
        T[left].print();

        return;
    }

}

void lockTest() {
    int tablesize = 256;
    ClearyEntry<addtype, remtype>* T;
    cudaMallocManaged(&T, tablesize * sizeof(ClearyEntry<addtype, remtype>));

    printf("\tInitializing Entries\n");
    for (int i = 0; i < tablesize; i++) {
        new (&T[i]) ClearyEntry<addtype, remtype>();
    }

    printf("\tStarting Lock Test\n");
    lockTestDevice << <1, 10 >> > (T);
    cudaDeviceSynchronize();

    cudaFree(T);
}


int main(void)
{
    int testSize = 150;
    int addressSize = 8;
    
    printf("==============================================================================================================\n");
    printf("                              BASIC TEST                              \n");
    printf("==============================================================================================================\n");
    uint64_t* testset1 = generateTestSet(testSize);
    TestFill(testSize, addressSize, testset1);
    

    printf("==============================================================================================================\n");
    printf("                            COLLISION TEST                            \n");
    printf("==============================================================================================================\n");
    uint64_t* testset2 = generateCollidingSet(testSize, addressSize);
    TestFill(testSize, addressSize, testset2);

    /*
    printf("==============================================================================================================\n");
    printf("                            FILL AND CHECK TEST                            \n");
    printf("==============================================================================================================\n");
    
    int sampleSize = 50;
    for (int i = 0; i < sampleSize; i++) {
        printf("\n\tTEST %i\n", i);
        uint64_t* testset3 = generateCollidingSet((int)pow(2, addressSize), addressSize);
        TestFillAndCheck(testSize, addressSize, testset3);
    }
    */

    //printf("Lock Test\n");
    //lockTest();

    return 0;
}