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
    uint64_cu* res;
    gpuErrchk(cudaMallocManaged(&res, size * sizeof(uint64_cu)));

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
    uint64_cu* res;
    gpuErrchk(cudaMallocManaged(&res, size * sizeof(uint64_cu)));

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

__global__
void fillClearyCuckoo(int N, uint64_cu* vals, ClearyCuckoo* H, addtype begin=0)
{
    int index = threadIdx.x;
    int stride = blockDim.x;
    //printf("\t\t\t\tStarting Thread:%i\n", index + begin);
    for (int i = index+begin; i < N+begin; i += stride) {
        //printf("\t\t\t\tCC Index:%i\n", i);
        if (!(H->insert(vals[i]))) {
            //printf("!------------ Insertion Failure ------------!\n");
            break;
        }
    }
}

__global__
void fillClearyCuckoo(int N, uint64_cu* vals, ClearyCuckoo* H, addtype* occupancy, int* failFlag)
{
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < N; i += stride) {
        if (failFlag[0]) {
            break;
        }
        if (!(H->insert(vals[i]))) {
            atomicCAS(&(failFlag[0]), 0, 1);
            break;
        }
        atomicAdd(&occupancy[0], 1);
    }
}

__global__
void fillCleary(int N, uint64_cu* vals, Cleary* H, addtype begin=0)
{
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index+begin; i < N+begin; i += stride) {
        //printf("Inserting %" PRIu64 "\n", vals[i]);
        if (!(H->insert(vals[i]))) {
            //printf("!------------ Insertion Failure ------------!\n");
            break;
        }
    }
}

__global__
void checkClearyCuckoo(int N, uint64_cu* vals, ClearyCuckoo* H, bool* res)
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
void checkCleary(int N, uint64_cu* vals, Cleary* H, bool* res)
{
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < N; i += stride) {
        if (!(H->lookup(vals[i]))) {
            res[0] = false;
        }
    }
}


void TestFill(int N, int tablesize, uint64_cu* vals) {
    //Init Var
    bool* res;
    gpuErrchk(cudaMallocManaged((void**)&res, sizeof(bool)));

	//Create Table 1
    ClearyCuckoo* cc;
    gpuErrchk(cudaMallocManaged((void**)&cc, sizeof(ClearyCuckoo)));
    new (cc) ClearyCuckoo(tablesize, 16);

    printf("Filling ClearyCuckoo\n");
    fillClearyCuckoo << <1, 1 >> > (N, vals, cc);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    printf("Devices Synced\n");
    cc->print();

    //Check Table
    res[0] = true;
    printf("Checking Cleary-Cuckoo\n");
    checkClearyCuckoo << <1, 1 >> > (N, vals, cc, res);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    printf("Devices Synced\n");
    if (res[0]) {
        printf("All still in the table\n");
    }
    else {
        printf("!---------------------Vals Missing---------------------!\n");
    }

	//Create Table 2
    Cleary* c;
    gpuErrchk(cudaMallocManaged((void**)&c, sizeof(Cleary)));
    new (c) Cleary(tablesize);

    printf("Filling Cleary\n");
    fillCleary << <1, 1 >> > (N, vals, c);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    printf("Devices Synced\n");
    c->print();

    //Checking
    *res = true;
    checkCleary << <1, 1 >> > (N, vals, c, res);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    printf("Devices Synced\n");
    if (res[0]) {
        printf("All still in the table\n");
    }
    else {
        printf("!---------------------Vals Missing---------------------!\n");
    }

    //Destroy Vars
    gpuErrchk(cudaFree(res));
    gpuErrchk(cudaFree(cc));
    gpuErrchk(cudaFree(c));
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
    gpuErrchk(cudaMallocManaged(&T, tablesize * sizeof(ClearyEntry<addtype, remtype>)));

    printf("\tInitializing Entries\n");
    for (int i = 0; i < tablesize; i++) {
        new (&T[i]) ClearyEntry<addtype, remtype>();
    }

    printf("\tStarting Lock Test\n");
    lockTestDevice << <1, 10 >> > (T);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk(cudaFree(T));
}

void entryTest() {
    ClearyEntry<addtype, remtype> c = ClearyEntry<addtype, remtype>();
    c.setR(351629921636382);
    c.print();
    printf("Entry After R %" PRIu64 "\n", c.getR());
}

void Test(int N) {
    const int addressSize = N;
    const int testSize = std::pow(2, addressSize);
    //const int testSize = 5;


    //printf("Lock Test\n");
    //lockTest();

    printf("==============================================================================================================\n");
    printf("                              BASIC TEST                              \n");
    printf("==============================================================================================================\n");
    uint64_cu* testset1 = generateTestSet(testSize);
    TestFill(testSize, addressSize, testset1);
    gpuErrchk(cudaFree(testset1));


    printf("==============================================================================================================\n");
    printf("                            COLLISION TEST                            \n");
    printf("==============================================================================================================\n");
    uint64_cu* testset2 = generateCollidingSet(testSize, addressSize);
    TestFill(testSize, addressSize, testset2);
    gpuErrchk(cudaFree(testset2));

    printf("\nTESTING DONE\n");
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
    for (int N = 5; N < 5+NUM_TABLES; N++) {
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
                        if (params && setup) {
                            S = std::stoi(params->at(4));
                        }
                        setup = false;

                        uint64_cu* vals = generateTestSet(size);

                        //Init Cleary Cuckoo
                        ClearyCuckoo* cc;
                        gpuErrchk(cudaMallocManaged((void**)&cc, sizeof(ClearyCuckoo)));
                        new (cc) ClearyCuckoo(N, H);
                        cc->setMaxLoops(L);

                        //Init Cleary
                        Cleary* c;
                        gpuErrchk(cudaMallocManaged((void**)&c, sizeof(Cleary)));
                        new (c) Cleary(N);

                        //Loop over intervals

                        for (int j = 0; j < INTERVAL + WARMUP; j++) {
                            //Fill the table
                            //printf("Filling ClearyCuckoo\n");
                            //Start the Timer
                            std::chrono::steady_clock::time_point begin;
                            std::chrono::steady_clock::time_point end;

                            if (j >= WARMUP) {
                                //printf("\t\tBegin: %i End:%i\n", setsize * j, setsize * (j+1));
                                fillClearyCuckoo << <1, std::pow(2, T) >> > (setsize, vals, cc, setsize * (j - WARMUP));
                                gpuErrchk( cudaPeekAtLastError() );
                                gpuErrchk( cudaDeviceSynchronize() );
                                //End the timer
                                end = std::chrono::steady_clock::now();

                                myfile << N << "," << std::pow(2, T) << "," << L << "," << H << "," << S << ",cuc," << (j - WARMUP) << "," << (std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count())/setsize << ",\n";
                            }



                            //Fill the table
                            //printf("Filling Cleary\n");
                            //Start the Timer
                            begin = std::chrono::steady_clock::now();
                            if (j >= WARMUP) {
                                fillCleary << <1, std::pow(2, T) >> > (setsize, vals, c, setsize * (j - WARMUP));
                                gpuErrchk( cudaPeekAtLastError() );
                                gpuErrchk( cudaDeviceSynchronize() );
                                //End the timer
                                end = std::chrono::steady_clock::now();

                                myfile << N << "," << std::pow(2, T) << "," << L << "," << H << "," << S << ",cle," << (j - WARMUP) << "," << (std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count()) / setsize << ",\n";
                            }
                        }
                        gpuErrchk(cudaFree(cc));
                        gpuErrchk(cudaFree(c));
                        gpuErrchk(cudaFree(vals));
                    }
                }
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
                    fillClearyCuckoo << <1, 256 >> > (size, vals, cc, occ, failFlag);
                    gpuErrchk( cudaPeekAtLastError() );
                    gpuErrchk( cudaDeviceSynchronize() );

                    myfile << N << "," << j << "," << k << "," << S << "," << occ[0] << ",\n";

                    gpuErrchk(cudaFree(failFlag));
                    gpuErrchk(cudaFree(cc));
                    gpuErrchk(cudaFree(occ));
                    gpuErrchk(cudaFree(vals));
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
        Test(std::stoi(argv[2]));
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
        ClearyEntry<addtype, remtype> c = ClearyEntry<addtype, remtype>();
        c.setA(std::stoi(argv[2]));
        printf("A:%i\n",c.getA());
    }

    return 0;
}
