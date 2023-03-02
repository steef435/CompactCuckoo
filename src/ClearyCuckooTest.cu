#ifndef TABLES
#define TABLES
#include "ClearyCuckoo.cu"
#include "Cleary.cu"
#include "ClearyCuckooBucketed.cu"
#include "CuckooBucketed.cu"
#include "Cuckoo.cu"
#endif

#ifdef REHASH
GPUHEADER_G
void callRehash(ClearyCuckoo* T) {
    T->rehash();
}


bool testRehash(int N, uint64_cu* vals){
    int tablesize = std::pow(2, N);
    int fillSize = (int) (((float)tablesize) * 0.75);

    printf("Tablesize %i, FillSize %i\n", tablesize, fillSize);

#ifdef GPUCODE
    ClearyCuckoo* cc;
    gpuErrchk(cudaMallocManaged((void**)&cc, sizeof(ClearyCuckoo)));
    new (cc) ClearyCuckoo(N, 4);
#else
    ClearyCuckoo* cc = new ClearyCuckoo(N, 4);
#endif

    //Fill an eigth of the table
#ifdef GPUCODE
    fillClearyCuckoo << <1, 8 >> > (fillSize, vals, cc);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#else
    fillClearyCuckoo(fillSize, vals, cc);
#endif

    printf("Before Rehash\n");
    cc->print();

    //Rehash
    #ifdef GPUCODE
    callRehash<<<1,1>>>(cc);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    #else
    cc->rehash();
    #endif

    printf("After Rehash\n");
    cc->print();

    //Check if all values from vals are still present
    bool res = true;
#ifdef GPUCODE
    checkClearyCuckoo << <1, 8 >> > (fillSize, vals, cc, &res);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#else
    checkClearyCuckoo(fillSize, vals, cc, &res);
#endif

    //Check for duplicates
    std::vector<uint64_cu> ccList = cc->toList();
    std::set<uint64_cu> ccSet(ccList.begin(), ccList.end());

    res = res && (ccSet.size() == ccList.size());

#ifdef GPUCODE
    gpuErrchk(cudaFree(cc));
#else
    delete cc;
#endif

    return res;
}

#endif
