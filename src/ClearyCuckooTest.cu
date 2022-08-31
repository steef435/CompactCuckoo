#ifndef TABLES
#define TABLES
#include "ClearyCuckoo.cu"
#include "Cleary.cu"
#endif


#ifdef REHASH
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
#ifdef GPUCODE
    fillClearyCuckoo << <1, 8 >> > (tablesize / 4, vals, cc);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#else
    fillClearyCuckoo(tablesize / 4, vals, cc);
#endif
    

    //Rehash
    cc->rehash();

    //Check if all values are still present
    bool res = true;
#ifdef GPUCODE
    checkClearyCuckoo << <1, 8 >> > (tablesize / 4, vals, cc, &res);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#else
    checkClearyCuckoo(tablesize / 4, vals, cc, &res);
#endif

    
    return res;
}

#endif