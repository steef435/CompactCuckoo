#include "numbergeneratorsTest.cu"
#include "ClearyCuckooTest.cu"
#include "SharedQueueTest.cu"
#include <cuda.h>

#ifndef TABLES
#define TABLES
#include "ClearyCuckoo.cu"
#include "Cleary.cu"
#endif

bool TestFill(int N, int T, int tablesize, uint64_cu* vals, bool c_bool, bool cc_bool) {
    bool testres = true;
    printf("TestRes\n");
    //Init Var
#ifdef GPUCODE
    bool* res;
    gpuErrchk(cudaMallocManaged((void**)&res, sizeof(bool)));
#else
    bool* res = new bool;
#endif
    int numThreads = std::pow(2, T);

    //Create Table 1
    printf("Testcc\n");
    if (cc_bool) {
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

        std::vector<std::thread> vecThread1(numThreads);

        for (int i = 0; i < numThreads; i++) {
            vecThread1.at(i) = std::thread(static_cast<void(*)(int, uint64_cu*, ClearyCuckoo*, int*, addtype, int, int)>(fillClearyCuckoo), N, vals, cc, nullptr, 0, i, numThreads);
            //Setting Thread Affinity
            //auto mask = (static_cast<DWORD_PTR>(1) << (i % 4));//core number starts from 0
            //auto ret = SetThreadAffinityMask(vecThread1.at(i).native_handle(), mask);
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

        //Check for duplicates
        std::vector<uint64_cu> ccList = cc->toList();
        std::set<uint64_cu> ccSet(ccList.begin(), ccList.end());

        if (ccSet.size() != ccList.size()) {
            printf("Duplicates Detected\n");
            testres = false;
        }

#ifdef GPUCODE
        gpuErrchk(cudaFree(cc));
#else
        delete cc;
#endif
    }

    printf("Testc\n");
    if (c_bool) {
        //Create Table 2
        printf("CreatingC\n");
#ifdef GPUCODE
        Cleary* c;
        gpuErrchk(cudaMallocManaged((void**)&c, sizeof(Cleary)));
        new (c) Cleary(tablesize);
#else
        Cleary* c = new Cleary(tablesize);
#endif

        printf("Filling Cleary\n");

#ifdef GPUCODE
        fillCleary << <1, 1 >> > (N, vals, c);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
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
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
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

#ifdef GPUCODE
        gpuErrchk(cudaFree(c));
#else
        delete c;
#endif
    }

    //Destroy Vars
#ifdef GPUCODE
    gpuErrchk(cudaFree(res));
#else
    delete res;
#endif

    return testres;
}


GPUHEADER_G
void lockTestDevice(ClearyEntry<addtype, remtype>* T) {
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
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaFree(T));
}

void entryTest() {
    ClearyEntry<addtype, remtype> c{};
    c.setR(351629921636382);
    c.print();
    printf("Entry After R %" PRIu64 "\n", c.getR());
}



void TableTest(int N, int T, int L, bool c, bool cc) {
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
        uint64_cu* testset1 = generateRandomSet(testSize);
        if (!TestFill(testSize, T, addressSize, testset1, c, cc)) {
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
        if (!TestFill(testSize, T, addressSize, testset2, c, cc)) {
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

#ifdef REHASH
    printf("==============================================================================================================\n");
    printf("                                          REHASH TEST                                                         \n");
    
    uint64_cu* testset3 = generateRandomSet(testSize);
    if (!testRehash(N, testset3)) {
        printf("TEST FAILED\n");
        res = false;
    }
    else {
        printf("TEST PASSED\n");
    }
    printf("==============================================================================================================\n");
    #ifdef GPUCODE
    gpuErrchk(cudaFree(testset3));
    #else
    delete[] testset3;
    #endif
#endif

    if (res) {
        printf("==============================================================================================================\n");
        printf("                                             ALL TESTS PASSED                                                 \n");
        printf("==============================================================================================================\n");
    }
    else {
        printf("==============================================================================================================\n");
        printf("                                             TEST(S) FAILED                                                 \n");
        printf("==============================================================================================================\n");
    }
}