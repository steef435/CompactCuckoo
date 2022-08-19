#ifndef TABLES
#define TABLES
#include "ClearyCuckoo.cu"
#include "Cleary.cu"
#endif

bool testRehash(int N, uint64_cu* vals){
    int tablesize = std::pow(2, N);;

#ifdef GPUCODE
    ClearyCuckoo* cc;
    gpuErrchk(cudaMallocManaged((void**)&cc, sizeof(ClearyCuckoo)));
    new (cc) ClearyCuckoo(tablesize, 16);
#else
    ClearyCuckoo* cc = new ClearyCuckoo(N, 2);
#endif

    //Fill an eigth of the table
    fillClearyCuckoo(tablesize / 4, vals, cc);

    cc->print();

    //Rehash
    cc->rehash();

    //Check if all values are still present
    bool res = true;
    checkClearyCuckoo(tablesize / 4, vals, cc, &res);
    return res;
}