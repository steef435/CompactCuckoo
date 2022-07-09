#include <iostream>
#include <random>
#include <algorithm>
#include <fstream>
#include <inttypes.h>
#include <chrono>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>
#include <thread>

#include "int_cu.h"


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

__host__ __device__
addtype getAdd(keytype key, int AS) {
    hashtype mask = ((hashtype)1 << AS) - 1;
    addtype add = key & mask;
    return add;
}

__host__ __device__
remtype getRem(keytype key, int AS) {
    remtype rem = key >> AS;
    return rem;
}

bool contains(uint64_cu* arr, uint64_cu val, int index) {
    for (int i = 0; i < index; i++) {
        if (val == arr[i]) {
            return true;
        }
    }
    return false;
}

uint64_cu* generateTestSet(int size) {
    //Random Number generator
    std::uniform_int_distribution<long long int> dist(0, std::llround(std::pow(2, 58)));
    
    #ifdef GPUCODE
    uint64_cu* res;
    gpuErrchk(cudaMallocManaged(&res, size * sizeof(uint64_cu)));
    #else
    uint64_cu* res = new uint64_cu[size];
    #endif

    for (int n = 0; n < size; n++) {
        uint64_cu rand = dist(e2);
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
uint64_cu reformKey(addtype add, remtype rem, int N) {
    rem = rem << N;
    rem += add;
    return rem;
}

uint64_cu* generateCollidingSet(int size, int N) {
    
    #ifdef GPUCODE
        uint64_cu* res;
        gpuErrchk(cudaMallocManaged(&res, size * sizeof(uint64_cu)));
    #else
        uint64_cu* res = new uint64_cu[size];
    #endif

    uint64_cu add = 7;

    for (int n = 0; n < (int) size/2; ++n) {
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

template <typename T>
void exportToCSV(std::vector<std::vector<T>>* matrix, std::string name) {
    std::ofstream myfile;
    std::string filename = "results/" + name + ".csv";
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
    std::string filename = "results/" + name + ".csv";
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


//Sources: https://stackoverflow.com/questions/1894886/parsing-a-comma-delimited-stdstring
//         https://stackoverflow.com/questions/11876290/c-fastest-way-to-read-only-last-line-of-text-file
std::vector<std::string>* getLastArgs(std::string filename) {
    std::string line;
    std::ifstream infile;
    infile.open(filename);

    if (infile.is_open())
    {
        char ch;
        infile.seekg(-1, std::ios::end);        // move to location 65
        infile.get(ch);                         // get next char at loc 66
        if (ch == '\n')
        {
            infile.seekg(-2, std::ios::cur);    // move to loc 64 for get() to read loc 65
            infile.seekg(-1, std::ios::cur);    // move to loc 63 to avoid reading loc 65
            infile.get(ch);                     // get the char at loc 64 ('5')
            while (ch != '\n')                   // read each char backward till the next '\n'
            {
                infile.seekg(-2, std::ios::cur);
                infile.get(ch);
            }
            std::string lastLine;
            std::getline(infile, lastLine);
            std::cout << "The last line : " << lastLine << '\n';
            line = lastLine;
        }
        else
            printf("Exception:Check CSV format\n");
            throw std::exception();
    }
    else {
        printf("File failed to open\n");
        return nullptr;
    }

    std::vector<std::string>* vect = new  std::vector<std::string>;
    std::stringstream ss(line);
    std::string field;

    while (getline(ss, field, ',')) {
        vect->push_back(field);
    }

    for (std::size_t i = 0; i < vect->size(); i++){
        std::cout << vect->at(i) << std::endl;
    }

    return vect;
}

/*
 *
 * Main Functions
 *
 */

GPUHEADER_G
void fillClearyCuckoo(int N, uint64_cu* vals, ClearyCuckoo* H, addtype begin=0, int id=0, int s=1)
{
#ifdef GPUCODE
    int index = getThreadID();
    int stride = blockDim.x;
#else
    int index = id;
    int stride = s;
#endif
    
    //printf("\t\t\t\t\t\tStarting Thread %i\n", getThreadID());
    for (int i = index+begin; i < N+begin; i += stride) {
        //printf("\t\t\t\tCC Index:%i\n", i);
        if (!(H->insert(vals[i]))) {
            //printf("\t\t\t\t\t\tStopping Thread %i\n", getThreadID());
            break;
        }
    }
    //printf("\t\t\t\t\t\tStopping Thread %i\n", getThreadID());
}

#ifdef GPUCODE
GPUHEADER_G
void fillClearyCuckoo(int N, uint64_cu* vals, ClearyCuckoo* H, addtype* occupancy, int* failFlag, int id = 0, int s = 1)
{
#ifdef GPUCODE
    int index = getThreadID();
    int stride = blockDim.x;
#else
    int index = id;
    int stride = s;
#endif

    //printf("\t\t\t\t\t\tStarting Thread %i\n", getThreadID());
    for (int i = index; i < N; i += stride) {
        if (failFlag[0]) {
            break;
        }
        if (!(H->insert(vals[i]))) {
            atomicCAS(&(failFlag[0]), 0, 1);
            //printf("\t\t\t\t\t\tStopping Thread %i\n", getThreadID());
            break;
        }
        atomicAdd(&occupancy[0], 1);
    }
    //printf("\t\t\t\t\t\tStopping Thread %i\n", getThreadID());
}
#endif

GPUHEADER_G
void fillCleary(int N, uint64_cu* vals, Cleary* H, addtype begin=0, int id = 0, int s = 1)
{
#ifdef GPUCODE
    int index = getThreadID();
    int stride = blockDim.x;
#else
    int index = id;
    int stride = s;
#endif

    //printf("\t\t\t\t\t\tStarting Thread %i\n", getThreadID());
    for (int i = index+begin; i < N+begin; i += stride) {
        //printf("Inserting %" PRIu64 "\n", vals[i]);
        if (!(H->insert(vals[i]))) {
            //printf("\t\t\t\t\t\tStopping Thread %i\n", getThreadID());
            break;
            //printf("\t\t\t\t\t\tStopping Thread %i\n", index);
        }
        //H->print();
    }
    //printf("\t\t\t\t\t\tStopping Thread %i\n", getThreadID());
}

GPUHEADER_G
void checkClearyCuckoo(int N, uint64_cu* vals, ClearyCuckoo* H, bool* res, int id = 0, int s = 1)
{
#ifdef GPUCODE
    int index = getThreadID();
    int stride = blockDim.x;
#else
    int index = id;
    int stride = s;
#endif

    for (int i = index; i < N; i += stride) {
        if (!(H->lookup(vals[i]))) {
            printf("\tSetting Res:Val %" PRIu64 " Missing\n", vals[i]);
            res[0] = false;
        }
    }
}

GPUHEADER_G
void checkCleary(int N, uint64_cu* vals, Cleary* H, bool* res, int id = 0, int s = 1)
{
#ifdef GPUCODE
    int index = getThreadID();
    int stride = blockDim.x;
#else
    int index = id;
    int stride = s;
#endif

    for (int i = index; i < N; i += stride) {
        if (!(H->lookup(vals[i]))) {
            printf("\tVal %" PRIu64 " Missing\n", vals[i]);
            res[0] = false;
        }
    }
}


bool TestFill(int N, int T, int tablesize, uint64_cu* vals) {
    bool testres = true;

    //Init Var
    #ifdef GPUCODE
    bool* res;
    gpuErrchk(cudaMallocManaged((void**)&res, sizeof(bool)));
    #else
    bool* res = new bool;
    #endif

	//Create Table 1
    #ifdef GPUCODE
    ClearyCuckoo* cc;
    gpuErrchk(cudaMallocManaged((void**)&cc, sizeof(ClearyCuckoo)));
    new (cc) ClearyCuckoo(tablesize, 16);
    #else
    ClearyCuckoo* cc = new ClearyCuckoo(tablesize, 16);
    #endif

    printf("Filling ClearyCuckoo\n");
    #ifdef GPUCODE
        fillClearyCuckoo << <1, 1 >> > (N, vals, cc);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    #else
        int numThreads = std::pow(2, T);
        std::vector<std::thread> vecThread1(numThreads);

        for (int i = 0; i < numThreads; i++) {
            vecThread1.at(i) = std::thread(static_cast<void(*)(int, uint64_cu*, ClearyCuckoo*, addtype, int, int)>(fillClearyCuckoo), N, vals, cc, 0, i, numThreads);
        }

        //Join Threads
        for (int i = 0; i < numThreads; i++) {
            vecThread1.at(i).join();
        }
    #endif
    printf("Devices Synced\n");
    cc->print();

    //Check Table
    res[0] = true;
    printf("Checking Cleary-Cuckoo\n");
    #ifdef GPUCODE
        checkClearyCuckoo << <1, 1 >> > (N, vals, cc, res);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    #else
        checkClearyCuckoo(N, vals, cc, res);
    #endif
    printf("Devices Synced\n");
    if (res[0]) {
        printf("All still in the table\n");
    }
    else {
        //testres = false;
        printf("!---------------------Vals Missing---------------------!\n");
    }

	//Create Table 2
    #ifdef GPUCODE
        Cleary* c;
        gpuErrchk(cudaMallocManaged((void**)&c, sizeof(Cleary)));
        new (c) Cleary(tablesize);
    #else
        Cleary* c = new Cleary(tablesize);
    #endif

    printf("Filling Cleary\n");
    #ifdef GPUCODE
        fillCleary << <1, 1 > >> (N, vals, c);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    #else
        std::vector<std::thread> vecThread2(numThreads);

        for (int i = 0; i < numThreads; i++) {
            vecThread2.at(i) = std::thread(fillCleary, N, vals, c, 0, i, numThreads);
        }

        //Join Threads
        for (int i = 0; i < numThreads; i++) {
            vecThread2.at(i).join();
        }
    #endif
    printf("Devices Synced\n");
    c->print();

    //Checking
    res[0] = true;
    printf("Checking Cleary\n");
    #ifdef GPUCODE
        checkCleary << <1, 1 >> > (N, vals, c, res);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    #else
        checkCleary(N, vals, c, res);
    #endif
    printf("Devices Synced\n");
    if (res[0]) {
        printf("All still in the table\n");
    }
    else {
        testres = false;
        printf("!---------------------Vals Missing---------------------!\n");
    }

    //Destroy Vars
    #ifdef GPUCODE
        gpuErrchk(cudaFree(res));
        gpuErrchk(cudaFree(cc));
        gpuErrchk(cudaFree(c));
    #else
        delete res;
        delete cc;
        delete c;
    #endif

        return testres;
}


GPUHEADER_G
void lockTestDevice(ClearyEntry<addtype, remtype>* T){
    addtype left = 1;
    addtype right = 4;

    while (true) {
        printf("\tGetting First Lock\n");
        if (!T[left].lock(false)) {
            printf("\tFirst Lock Failed\n");
                continue;
        }

        printf("\tLeft");
        T[left].print();

        printf("\tGetting Second Lock\n");
        if (!T[right].lock(false)) {
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
    gpuErrchk(cudaMallocManaged(&T, tablesize * sizeof(ClearyEntry<addtype, remtype>)));

    printf("\tInitializing Entries\n");
    for (int i = 0; i < tablesize; i++) {
        new (&T[i]) ClearyEntry<addtype, remtype>();
    }

    printf("\tStarting Lock Test\n");
#ifdef GPUCODE
    lockTestDevice << <1, 10 >> > (T);
#else

#endif
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk(cudaFree(T));
}

void entryTest() {
    ClearyEntry<addtype, remtype> c{};
    c.setR(351629921636382);
    c.print();
    printf("Entry After R %" PRIu64 "\n", c.getR());
}

void Test(int N, int T, int L) {
    bool res = true;

    const int addressSize = N;
    const int testSize = std::pow(2, addressSize);
    //const int testSize = 5;


    //printf("Lock Test\n");
    //lockTest();

    for (int i = 0; i < L; i++) {

        printf("==============================================================================================================\n");
        printf("                              BASIC TEST                              \n");
        printf("==============================================================================================================\n");
        uint64_cu* testset1 = generateTestSet(testSize);
        if (!TestFill(testSize, T, addressSize, testset1)) {
            res = false;
        }
#ifdef GPUCODE
        gpuErrchk(cudaFree(testset1));
#else
        delete[] testset1;
#endif

        printf("==============================================================================================================\n");
        printf("                            COLLISION TEST                            \n");
        printf("==============================================================================================================\n");
        uint64_cu* testset2 = generateCollidingSet(testSize, addressSize);
        if (!TestFill(testSize, T, addressSize, testset2)) {
            res = false;
        }
#ifdef GPUCODE
        gpuErrchk(cudaFree(testset2));
#else
        delete[] testset2;
#endif

        if (!res) {
            printf("TEST FAILED\n");
            break;
        }
        else {
            printf("TEST PASSED\n");
        }
    }

    if (res) {
        printf("==============================================================================================================\n");
        printf("                                             ALL TESTS PASSED                                                 \n");
        printf("==============================================================================================================\n");
    }
}


/* ================================================================================================================
 *
 *  Benchmark Methods
 *
 * ================================================================================================================
*/

void BenchmarkFilling(int NUM_TABLES, int INTERVAL, int NUM_SAMPLES, int NUM_THREADS, int NUM_LOOPS, int NUM_HASHES, std::vector<std::string>* params = nullptr) {

    const int WARMUP = 2;

    printf("=====================================================================\n");
    printf("                     Starting INSERTION BENCHMARK                    \n");
    printf("=====================================================================\n");

    std::ofstream myfile;
    std::string filename = "results/benchfill.csv";

    if (params) {
        printf("Opening\n");
        myfile.open(filename, std::ios_base::app);
        printf("Maybe\n");
    }
    else {
        myfile.open(filename);
    }

    if (!myfile.is_open()) {
        printf("File Failed to Open\n");

        return;
    }
    printf("File Opened\n");

    if (!params) {
        myfile << "tablesize,numthreads,loops,hashes,samples,type,interval,time\n";
    }

    //Tablesizes
    bool setup = true;
    for (int N = 5; N < 5 + NUM_TABLES; N++) {
        if (params && setup) {
            N = std::stoi(params->at(0));
        }
        printf("Table Size:%i\n", N);

        int size = std::pow(2, N);
        int setsize = (int)(size / INTERVAL);

        //Number of Threads
        for (int T = 0; T < NUM_THREADS; T++) {
            if (params && setup) {
                T = std::stoi(params->at(1));
            }
            printf("\tNumber of Threads:%i\n", T);

            for (int L = 0; L < NUM_LOOPS; L++) {
                if (params && setup) {
                    L = std::stoi(params->at(2));
                }
                printf("\t\tNumber of Loops:%i\n", L);

                for (int H = 1; H < NUM_HASHES; H++) {
                    if (params && setup) {
                        H = std::stoi(params->at(3));
                    }
                    printf("\t\t\tNumber of Hashes:%i\n", H);
                    //Number of samples
                    for (int S = 0; S < NUM_SAMPLES; S++) {
                        //printf("\t\t\t\tSample Number:%i\n", S);
                        if (params && setup) {
                            S = std::stoi(params->at(4));
                        }
                        setup = false;
                        //printf("\t\t\t\t\tGenVals\n");
                        uint64_cu* vals = generateTestSet(size);
                        //printf("\t\t\t\t\tInitCC\n");
                        //Init Cleary Cuckoo

#ifdef GPUCODE
                        ClearyCuckoo* cc;
                        gpuErrchk(cudaMallocManaged((void**)&cc, sizeof(ClearyCuckoo)));
                        new (cc) ClearyCuckoo(N, H);
#else
                        ClearyCuckoo* cc = new ClearyCuckoo(N, H);
#endif

                        cc->setMaxLoops(L);

                        //Loop over intervals
                        //printf("\t\t\t\t\tFilling\n");
                        for (int j = 0; j < INTERVAL + WARMUP; j++) {
                            //Fill the table
                            //printf("\t\t\t\t\t\tFilling ClearyCuckoo\n");
                            //Start the Timer
                            std::chrono::steady_clock::time_point begin;
                            std::chrono::steady_clock::time_point end;

                            begin = std::chrono::steady_clock::now();
                            if (j >= WARMUP) {
                                //printf("\t\tBegin: %i End:%i\n", setsize * j, setsize * (j+1));
#ifdef GPUCODE
                                fillClearyCuckoo << <1, std::pow(2, T) >> > (setsize, vals, cc, setsize * (j - WARMUP));
                                gpuErrchk(cudaPeekAtLastError());
                                gpuErrchk(cudaDeviceSynchronize());
#else
                                int numThreads = std::pow(2, T);
                                std::vector<std::thread> vecThread(numThreads);
                                for (int i = 0; i < numThreads; i++) {
                                    vecThread.at(i) = std::thread(static_cast<void(*)(int, uint64_cu*, ClearyCuckoo*, addtype, int, int)>(fillClearyCuckoo), setsize, vals, cc, setsize * (j - WARMUP), i, numThreads);
                                }

                                //Join Threads
                                for (int i = 0; i < numThreads; i++) {
                                    vecThread.at(i).join();
                                }
#endif
                                //End the timer
                                end = std::chrono::steady_clock::now();

                                myfile << N << "," << std::pow(2, T) << "," << L << "," << H << "," << S << ",cuc," << (j - WARMUP) << "," << (std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count()) / setsize << ",\n";
                            }

                        }
                        //printf("\t\t\t\t\tFreeVals\n");
#ifdef GPUCODE
                        gpuErrchk(cudaFree(cc));
                        gpuErrchk(cudaFree(vals));
#else       
                        delete cc;
                        delete[] vals;
#endif
                    }
                }
            }
        }
    }

    for (int N = 5; N < 5+NUM_TABLES; N++) {
        if (params && setup) {
            N = std::stoi(params->at(0));
        }
        printf("Table Size:%i\n", N);

        int size = std::pow(2, N);
        int setsize = (int)(size / INTERVAL);
        for (int T = 0; T < NUM_THREADS; T++) {
            printf("\tNumber of Threads:%i\n", T);
            for (int S = 0; S < NUM_SAMPLES; S++) {
                printf("\t\t\t\tSample Number:%i\n", S);
                //printf("\t\t\t\t\tGenVals\n");
                uint64_cu* vals = generateTestSet(size);

                //printf("\t\t\t\t\tInitC\n");
                //Init Cleary
                #ifdef GPUCODE
                Cleary* c;
                gpuErrchk(cudaMallocManaged((void**)&c, sizeof(Cleary)));
                new (c) Cleary(N);
                #else
                Cleary* c = new Cleary(N);
                #endif

                //Loop over intervals
                //printf("\t\t\t\t\tFilling\n");
                for (int j = 0; j < INTERVAL + WARMUP; j++) {
                  std::chrono::steady_clock::time_point begin;
                  std::chrono::steady_clock::time_point end;

                    //Fill the table
                    //printf("\t\t\t\t\t\tFilling Cleary\n");
                    //Start the Timer

                    begin = std::chrono::steady_clock::now();
                    if (j >= WARMUP) {
                        #ifdef GPUCODE
                            fillCleary << <1, std::pow(2, T) >> > (setsize, vals, c, setsize* (j - WARMUP));
                            gpuErrchk(cudaPeekAtLastError());
                            gpuErrchk(cudaDeviceSynchronize());
                        #else
                            int numThreads = std::pow(2, T);
                            std::vector<std::thread> vecThread(numThreads);
                            
                            for (int i = 0; i < numThreads; i++) {
                                vecThread.at(i) = std::thread(fillCleary, setsize, vals, c, setsize * (j - WARMUP), i, numThreads);
                            }

                            //Join Threads
                            for (int i = 0; i < numThreads; i++) {
                                vecThread.at(i).join();
                            }
                        #endif
                        //End the timer
                        end = std::chrono::steady_clock::now();
                        myfile << N << "," << std::pow(2, T) << "," << -1 << "," << -1 << "," << S << ",cle," << (j - WARMUP) << "," << (std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count()) / setsize << ",\n";
                    }

                }
                //printf("\t\t\t\t\tFreeVals\n");
                #ifdef GPUCODE
                gpuErrchk(cudaFree(c));
                gpuErrchk(cudaFree(vals));
                #else       
                delete c;
                delete[] vals;
                #endif
            }
        }
    }

    myfile.close();
    printf("\t\t\tBenchmark Done\n");
}

void BenchmarkMaxOccupancy(int TABLESIZES, int NUM_HASHES, int NUM_LOOPS, int NUM_SAMPLES) {

    printf("=====================================================================\n");
    printf("                   Starting MAX Occupancy Benchmark                  \n");
    printf("=====================================================================\n");

    std::ofstream myfile;
    std::string filename = "results/benchmax.csv";
    myfile.open(filename);
    if (!myfile.is_open()) {
        printf("File Failed to Open\n");
        return;
    }
    printf("File Opened");

    myfile << "tablesize,numhashes,numloops,samples,max\n";

    //MAX_LOOPS
    for (int N = 5; N < 5 + TABLESIZES; N++) {
        printf("Table Size:%i\n", N);
        int size = std::pow(2, N);
        for (int j = 1; j < NUM_HASHES; j++) {
            printf("\tNum of Hashes:%i\n", j);
            for (int k = 0; k < NUM_LOOPS; k++) {
                printf("\t\tNum of Loops:%i\n", k);
                for (int S = 0; S < NUM_SAMPLES; S++) {
                    //printf("\t\t'tSample Number:%i\n", S);
                    uint64_cu* vals = generateTestSet(size);

                    int* failFlag;
                    gpuErrchk(cudaMallocManaged(&failFlag, sizeof(int)));
                    failFlag[0] = false;

                    //Init Cleary Cuckoo
                    ClearyCuckoo* cc;
                    gpuErrchk(cudaMallocManaged((void**)&cc, sizeof(ClearyCuckoo)));
                    new (cc) ClearyCuckoo(N, j);
                    cc->setMaxLoops(k);

                    //Var to store num of inserted values
                    addtype* occ;
                    gpuErrchk(cudaMallocManaged(&occ, sizeof(addtype)));
                    occ[0] = 0;

                    //Fill the table
#ifdef GPUCODE
                    fillClearyCuckoo << <1, 256 >> > (size, vals, cc, occ, failFlag);
                    gpuErrchk( cudaPeekAtLastError() );
                    gpuErrchk( cudaDeviceSynchronize() );

                    myfile << N << "," << j << "," << k << "," << S << "," << occ[0] << ",\n";

                    gpuErrchk(cudaFree(failFlag));
                    gpuErrchk(cudaFree(cc));
                    gpuErrchk(cudaFree(occ));
                    gpuErrchk(cudaFree(vals));
#else
                    delete failFlag;
                    delete cc;
                    delete occ;
                    delete[] vals;
#endif
                }
            }
        }
    }

    myfile.close();

    printf("\t\t\tStarting MAX Occupancy Benchmark\n");
}


int main(int argc, char* argv[])
{
    if (argc == 1) {
        printf("No Arguments Passed\n");
    }

    if (strcmp(argv[1], "test") == 0) {
        if (argc < 4) {
            printf("Not Enough Arguments Passed\n");
            printf("Required: TABLESIZE, NUM_THREADS, SAMPLES\n");
            return 0;
        }
        Test(std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]));
    }
    else if (strcmp(argv[1], "benchmax") == 0) {
        if (argc < 6) {
            printf("Not Enough Arguments Passed\n");
            printf("Required: TABLESIZES, NUM_HASHES, NUM_LOOPS, NUM_SAMPLES\n");
            return 0;
        }
        BenchmarkMaxOccupancy(std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]), std::stoi(argv[5]));
    }
    else if (strcmp(argv[1], "benchfill") == 0) {
        if (argc < 6) {
            printf("Not Enough Arguments Passed\n");
            printf("Required: NUM_TABLES, INTERVAL, NUM_SAMPLES, NUM_THREADS, NUM_LOOPS, NUM_HASHES\n");
            return 0;
        }
        else if (strcmp(argv[2], "continue") == 0) {
            printf("Continuing from Last Position\n");
            std::vector<std::string>* lastargs = getLastArgs("results/benchfill.csv");

            BenchmarkFilling(std::stoi(argv[3]), std::stoi(argv[4]), std::stoi(argv[5]), std::stoi(argv[6]), std::stoi(argv[7]), std::stoi(argv[8]), lastargs);
            delete lastargs;
            return 0;
        }

        BenchmarkFilling(std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]), std::stoi(argv[5]), std::stoi(argv[6]), std::stoi(argv[7]));
    }

    else if (strcmp(argv[1], "debug") == 0) {

    }

    return 0;
}
