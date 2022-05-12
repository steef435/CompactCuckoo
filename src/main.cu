#include <iostream>
#include <random>
#include <algorithm>
#include <fstream>
#include <inttypes.h>
#include <chrono>
#include <vector>

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

template <typename T>
void exportToCSV(std::vector<std::vector<T>>* matrix, std::string name) {
    std::ofstream myfile;
    std::string filename = "../results/benchmark/" + name + ".csv";
    myfile.open(filename);
    if (myfile.is_open()) {
        for (int i = 0; i < matrix->size(); i++) {
            for (int j = 0; j < matrix->at(0).size(); j++) {
                myfile << (*matrix)[i][j] << ",";
            }
            myfile << "\n";
        }
        myfile.close();
    }
    else {
        std::cout << "Failed to open file : \n";
    }
}

template <typename T>
void exportToCSV(std::vector<std::vector<std::vector<T>>>* matrix, std::string name) {
    std::ofstream myfile;
    std::string filename = "../results/benchmark/" + name + ".csv";
    myfile.open(filename);
    if (myfile.is_open()) {
        for (int i = 0; i < matrix->size(); i++) {
            for (int j = 0; j < matrix->at(0).size(); j++) {
                for (int k = 0; k < (matrix->at(0)).at(0).size(); k++) {
                    myfile << i << "," << j << "," << k << "," << ((*matrix)[i][j])[k] << "\n";
                }
            }
        }
        myfile.close();
    }
}


/*
 *
 * Main Functions
 *
 */

__global__
void fillClearyCuckoo(int N, uint64_t* vals, ClearyCuckoo* H, addtype begin=0)
{   
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index+begin; i < N+begin; i += stride) {
        if (!(H->insert(vals[i]))) {
            printf("!------------ Insertion Failure ------------!\n");
            break;
        }
    }
}

__global__
void fillClearyCuckoo(int N, uint64_t* vals, ClearyCuckoo* H, addtype* occupancy)
{
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < N; i += stride) {
        if (!(H->insert(vals[i]))) {
            break;
        }
        atomicAdd(&occupancy[0], 1);
    }
}

__global__
void fillCleary(int N, uint64_t* vals, Cleary* H, addtype begin = 0)
{
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index+begin; i < N+begin; i += stride) {
        if (!(H->insert(vals[i]))) {
            printf("!------------ Insertion Failure ------------!\n");
            break;
        }
    }
}

__global__
void checkClearyCuckoo(int N, uint64_t* vals, ClearyCuckoo* H, bool* res)
{
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < N; i += stride) {
        if (!(H->lookup(vals[i]))) {
            printf("\tSetting Res:Val %" PRIu64 " Missing\n", vals[i]);
            res[0] = false;
        }
    }
}

__global__
void checkCleary(int N, uint64_t* vals, Cleary* H, bool* res)
{
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < N; i += stride) {
        if (!(H->lookup(vals[i]))) {
            printf("\tSetting Res:Val %" PRIu64 " Missing\n", vals[i]);
            res[0] = false;
        }
    }
}


void TestFill(int N, int tablesize, uint64_t* vals) {
    //Init Var
    printf("Making Check Bool\n");
    bool* res;
    cudaMallocManaged((void**)&res, sizeof(bool));
    printf("Assigning Value\n");

	//Create Table 1
    ClearyCuckoo* cc;
    cudaMallocManaged((void**)&cc, sizeof(ClearyCuckoo));
    new (cc) ClearyCuckoo(tablesize, 16);

    printf("Filling ClearyCuckoo\n");
	fillClearyCuckoo << <1, 1 >> > (N, vals, cc);
    cudaDeviceSynchronize();
    printf("Devices Synced\n");
    cc->print();

    //Check Table
    res[0] = true;
    printf("Checking Cleary-Cuckoo\n");
    checkClearyCuckoo << <1, 1 >> > (N, vals, cc, res);
    cudaDeviceSynchronize();
    printf("Devices Synced\n");
    if (res[0]) {
        printf("All still in the table\n");
    }
    else {
        printf("!---------------------Vals Missing---------------------!\n");
    }

	//Create Table 2
    Cleary* c;
    cudaMallocManaged((void**)&c, sizeof(Cleary));
    new (c) Cleary(tablesize);

    printf("Filling Cleary\n");
    fillCleary << <1, 1 >> > (N, vals, c);
    cudaDeviceSynchronize();
    printf("Devices Synced\n");
    c->print();

    //Checking 
    *res = true;
    checkCleary << <1, 1 >> > (N, vals, c, res);
    cudaDeviceSynchronize();
    printf("Devices Synced\n");
    if (res[0]) {
        printf("All still in the table\n");
    }
    else {
        printf("!---------------------Vals Missing---------------------!\n");
    }

    //Destroy Vars
    cudaFree(res);
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

void entryTest() {
    ClearyEntry<addtype, remtype> c = ClearyEntry<addtype, remtype>();
    c.setR(351629921636382);
    c.print();
    printf("Entry After R %" PRIu64 "\n", c.getR());
}

void Test() {
    const int addressSize = 8;
    const int testSize = std::pow(2, addressSize);
    //const int testSize = 5;
    

    //printf("Lock Test\n");
    //lockTest();

    printf("==============================================================================================================\n");
    printf("                              BASIC TEST                              \n");
    printf("==============================================================================================================\n");
    uint64_t* testset1 = generateTestSet(testSize);
    TestFill(testSize, addressSize, testset1);
    cudaFree(testset1);


    printf("==============================================================================================================\n");
    printf("                            COLLISION TEST                            \n");
    printf("==============================================================================================================\n");
    uint64_t* testset2 = generateCollidingSet(testSize, addressSize);
    TestFill(testSize, addressSize, testset2);
    cudaFree(testset2);

    printf("\nTESTING DONE\n");
}


/* ================================================================================================================
 *
 *  Benchmark Methods
 * 
 * ================================================================================================================ 
*/

void BenchmarkFilling(int NUM_TABLES, int INTERVAL, int NUM_SAMPLES, int NUM_THREADS = 48) {

    //Tablesizes
    for (int N = 8; N < NUM_TABLES; N++) {

        int size = std::pow(2, N);
        int setsize = (int)(size / INTERVAL);

        //Table to store the results for this size
        std::vector<std::vector<long long int>>* insTimes_c = new std::vector<std::vector<long long int>>(setsize, std::vector<long long int>(NUM_SAMPLES, 0));
        std::vector<std::vector<long long int>>* insTimes_cc = new std::vector<std::vector<long long int>>(setsize, std::vector<long long int>(NUM_SAMPLES, 0));

        //Number of samples
        for (int S = 0; S < NUM_SAMPLES; S++) {
            uint64_t* vals = generateTestSet(N);

            //Init Cleary Cuckoo
            ClearyCuckoo* cc;
            cudaMallocManaged((void**)&cc, sizeof(ClearyCuckoo));
            new (cc) ClearyCuckoo(N, 16);

            //Init Cleary
            Cleary* c;
            cudaMallocManaged((void**)&c, sizeof(Cleary));
            new (c) Cleary(N);

            //Loop over intervals
            
            for (int j = 0; j < setsize; j++) {
                //Fill the table
                printf("Filling ClearyCuckoo\n");
                //Start the Timer
                std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
                fillClearyCuckoo << <1, NUM_THREADS >> > (setsize, vals, cc, setsize*j);
                cudaDeviceSynchronize();
                //End the timer
                std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

                (*insTimes_cc)[j][S] = std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count();



                //Fill the table
                printf("Filling Cleary\n");
                //Start the Timer
                begin = std::chrono::steady_clock::now();
                fillCleary << <1, NUM_THREADS >> > (setsize, vals, c, setsize * j);
                cudaDeviceSynchronize();
                //End the timer
                end = std::chrono::steady_clock::now();
                (*insTimes_c)[j][S] = std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count();
            }
            printf("Loops Done\n");

            cudaFree(cc);
            cudaFree(c);
        }

        printf("Printing Results\n");
        std::string num = std::to_string(N);
        exportToCSV(insTimes_c, "InsertionC" + num);
        exportToCSV(insTimes_cc, "InsertionCC" + num);

        delete insTimes_c;
        delete insTimes_cc;
    }
}

void BenchmarkMaxOccupancy(int TABLESIZES, int NUM_HASHES, int NUM_LOOPS, int NUM_SAMPLES) {

    printf("=====================================================================\n");
    printf("                   Starting MAX Occupancy Benchmark                  \n");
    printf("=====================================================================\n");

    //MAX_LOOPS
    for (int N = 8; N < 8 + TABLESIZES; N++) {
        //Table to store the results for this size
        std::vector<std::vector<std::vector<addtype>>>* max_cc = new std::vector<std::vector<std::vector<addtype>>>(NUM_HASHES, std::vector<std::vector<addtype>>(NUM_LOOPS, std::vector<addtype>(NUM_SAMPLES, 0)));
        printf("Table Size:%i\n", N);
        int size = std::pow(2, N);
        for (int j = 1; j < NUM_HASHES; j++) {
            printf("\tNum of Hashes:%i\n", j);
            for (int k = 0; k < NUM_LOOPS; k++) {
                printf("\t\tNum of Loops:%i\n", k);
                for (int S = 0; S < NUM_SAMPLES; S++) {
                    //printf("\t\t\tTest %i\n", S);
                    uint64_t* vals = generateTestSet(size);
                    //Init Cleary Cuckoo
                    ClearyCuckoo* cc;
                    cudaMallocManaged((void**)&cc, sizeof(ClearyCuckoo));
                    new (cc) ClearyCuckoo(N, j);

                    //Var to store num of inserted values
                    addtype* occ;
                    cudaMallocManaged(&occ, sizeof(addtype));
                    occ[0] = 0;

                    //Fill the table
                    fillClearyCuckoo << <1, 256 >> > (size, vals, cc, occ);
                    cudaDeviceSynchronize();

                    (*max_cc)[j][k][S] = occ[0];

                    cudaFree(cc);
                    cudaFree(occ);
                }
            }
        }

        printf("Printing Results\n");
        std::string num = std::to_string(N);
        exportToCSV(max_cc, "MaxFillC" + num);

        delete max_cc;
    }
    
}


int main(int argc, char* argv[])
{
    if (argc == 1) {
        printf("No Arguments Passed\n");
    }

    if (strcmp(argv[1], "test") == 0) {
        Test();
    }
    else if (strcmp(argv[1], "benchmax") == 0) {
        if (argc < 6) {
            printf("Not Enough Arguments Passed\n");
            printf("Required: TABLESIZES, NUM_HASHES, NUM_LOOPS, NUM_SAMPLES\n");
        }
        BenchmarkMaxOccupancy(std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]), std::stoi(argv[5]));
    }
    else if (strcmp(argv[1], "benchfill") == 0) {
        if (argc < 6) {
            printf("Not Enough Arguments Passed\n");
            printf("Required: NUM_TABLES, INTERVAL, NUM_SAMPLES, NUM_THREADS\n");
        }
        BenchmarkFilling(std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]), std::stoi(argv[5]));
    }

    return 0;
}