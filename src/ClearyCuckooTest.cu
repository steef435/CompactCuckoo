#ifndef TABLES
#define TABLES
#include "ClearyCuckoo.cu"
#include "Cleary.cu"
#endif

bool testRehash(int N, uint64_cu* vals){
    int tablesize = std::pow(2, N);;

#ifdef GPUCODE
    printf("Alloc Table GPU\n");
    ClearyCuckoo* cc;
    gpuErrchk(cudaMallocManaged((void**)&cc, sizeof(ClearyCuckoo)));
    new (cc) ClearyCuckoo(tablesize, 16);
#else
    printf("Alloc Table CPU\n");
    ClearyCuckoo* cc = new ClearyCuckoo(N, 2);
#endif
    
    printf("Values up to %i:\n", tablesize/8);
    for (int i = 0; i < tablesize / 4; i++) {
        printf("%lli, ", vals[i]);
    }
    printf("\n");

    //Fill an eigth of the table
    printf("Filling Table\n");
    fillClearyCuckoo(tablesize / 4, vals, cc);

    cc->print();

    //Rehash
    printf("Rehash Table\n");
    cc->rehash();

    //Check if all values are still present
    printf("Check Table\n");
    bool res;
    checkClearyCuckoo(tablesize / 4, vals, cc, &res);
    return res;
}