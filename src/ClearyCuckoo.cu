#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <math.h>
#include <iterator>
#include <set>
#include <inttypes.h>
#include <atomic>
#include <random>
#include "SpinBarrier.h"

//For to List
#include <vector>

#include <curand.h>
#include <curand_kernel.h>

#ifndef MAIN
#define MAIN
#include "main.h"
#endif

#ifndef HASHTABLE
#define HASHTABLE
#include "HashTable.h"
#endif

#ifndef HASHINCLUDED
#define HASHINCLUDED
#include "hashfunctions.cu"
#endif

#ifndef SHAREDQUEUE
#define SHAREDQUEUE
#include "SharedQueue.cu"
#endif

#ifndef CCENTRY
#define CCENTRY
#include "ClearyCuckooEntry.cu"
#include "ClearyCuckooEntryCompact.cu"
#endif




class ClearyCuckoo : HashTable{

/*
*
*  Global Variables
*
*/
    private:
        //Constant Vars
        const static int HS = 59;       //HashSize
        int MAXLOOPS = 25;
        int MAXREHASHES = 30;

        const int ENTRYSIZE2 = 64;

        //Vars at Construction
        int AS;                         //AdressSize
        int RS;                         //RemainderSize
        int tablesize;
        const int valSize;
        const int valsPerEntry;
        int numEntries = 0;

        int occupancy = 0;

        //Hash tables
        ClearyCuckooEntryCompact<addtype, remtype>* T;

        int hashcounter = 0;

        //Hash function ID
        int hn;
        int* hashlist;

        //Flags
#ifdef GPUCODE
        int failFlag = 0;
        int occupation = 0;
        int rehashFlag = 0;
#else
        std::atomic<int> failFlag;
        std::atomic<int> occupation;
        std::atomic<int> rehashFlag;
#endif
        SharedQueue<keytype>* rehashQueue;

        //Method to init the hashlsit
        GPUHEADER
        void createHashList(int* list) {
            for (int i = 0; i < hn; i++) {
                list[i] = i;
            }
            return;
        }

        //Method to iterate over hashes (Rehashing)
        GPUHEADER
        void iterateHashList(int* list) {
            //printf("\tUpdating Hashlist\n");
            for (int i = 0; i < hn; i++) {
                list[i] = (list[i]+1+i)%32;
            }
            return;
        }

        //Method to select next hash to use in insertion
        GPUHEADER
        int getNextHash(int* ls, int curr) {
            for (int i = 0; i < hn; i++) {
                if (ls[i] == curr) {
                    if (i + 1 != hn) {
                        return ls[i + 1];
                    }
                    else {
                        return ls[0];
                    }
                }
            }

            //Default return 0 if hash can't be found
            return ls[0];
        }

        //Checks if hash ID is contained in hashlist
        GPUHEADER
        bool containsHash(int* ls, int query) {
            for (int i = 0; i < hn; i++) {
                if (ls[i] == query) {
                    return true;
                }
            }
            return false;
        }

#ifdef GPUCODE
        //Method to set Flags on GPU(Failure/Rehash)
        GPUHEADER_D
        bool setFlag(int* loc, int val, bool strict=true) {
            int val_i = val == 0 ? 1 : 0;

            //In devices, atomically exchange
            uint64_cu res = atomicCAS(loc, val_i, val);
            //Make sure the value hasn't changed in the meantime
            if ( (res != val_i) && strict) {
                return false;
            }
            __threadfence();
            return true;
        }

#else
        GPUHEADER
        //Method to set Flags on CPU (Failure/Rehash)
        bool setFlag(std::atomic<int>* loc, int val, bool strict=true) {
            int val_i = val == 0 ? 1 : 0;
                //printf("%i:\t:Attempting CAS\n", getThreadID());
            if (std::atomic_compare_exchange_strong(loc, &val_i, val)) {
                //printf("%i:\t:Flag Set\n", getThreadID());
                return true;
            }else{
              return false;
            }
        }
#endif



        /**
         * Internal Insertion Loop
         **/
        GPUHEADER_D
        result insertIntoTable(keytype k, ClearyCuckooEntryCompact<addtype, remtype>* T, int* hs, int depth=0){
            printf("InsertIntoTable\n");
            keytype x = k;
            int hash = hs[0];

            //If the key is already inserted don't do anything
            if (lookup(k, T)) {
                return FOUND;
            }
            //Start the iteration
            int c = 0;

            printf("\tEntering Loop\n");
            while(c < MAXLOOPS){
                //Get the add/rem of k
                hashtype hashed1 = RHASH(HFSIZE, hash, x);
                addtype add = getAdd(hashed1, AS);
                remtype rem = getRem(hashed1, AS);

                addtype real_add = (addtype)(add / valsPerEntry);
                addtype subIndex = (addtype)(add % valsPerEntry);

                printf("\tExchVals\n");
                //Exchange Values
                ClearyCuckooEntryCompact<addtype, remtype> entry(rem, hash, true, false, 0);
                T[real_add].tableSwap(&entry, subIndex, 0);

                //Store the old value
                printf("\tStoreOld\n");
                remtype temp = entry.getR(0);
                bool wasoccupied = entry.getO(0);
                int oldhash = entry.getH(0);

                //If the old val was empty return
                if(!wasoccupied){
                    return INSERTED;
                }

                //Otherwise rebuild the original key
                hashtype h_old = reformKey(add, temp, AS);
                x = RHASH_INVERSE(HFSIZE, oldhash, h_old);

                //Hash with the next hash value
                hash = getNextHash(hs, oldhash);

                c++;
            }

        };



        //Lookup internal method
        GPUHEADER
        bool lookup(uint64_cu k, ClearyCuckooEntryCompact<addtype, remtype>* T){
            printf("Lookup\n");
            //Iterate over hash functions and check if found
            for (int i = 0; i < hn; i++) {
                uint64_cu hashed1 = RHASH(HFSIZE, hashlist[i], k);
                addtype add = getAdd(hashed1, AS);
                remtype rem = getRem(hashed1, AS);

                addtype real_add = (addtype)(add / valsPerEntry);
                addtype subIndex = (addtype)(add % valsPerEntry);

                if ( (T[real_add].getR(subIndex) == rem) && T[real_add].getO(subIndex) && (T[real_add].getH(subIndex) == hashlist[i]) ) {
                    return true;
                }
            }
            return false;
        };

        GPUHEADER
        void print(ClearyCuckooEntryCompact<addtype, remtype>* T) {
            printf("----------------------------------------------------------------\n");
            printf("|    i     |     R[i]       | O[i] |        key         |label |\n");
            printf("----------------------------------------------------------------\n");
            printf("Tablesize %i\n", tablesize);
            for (int i = 0; i < tablesize; i++) {

                addtype real_add = (addtype)(i / valsPerEntry);
                addtype subIndex = (addtype)(i % valsPerEntry);

                if (T[real_add].getO(subIndex)) {
                    remtype rem = T[real_add].getR(subIndex);
                    int label = T[real_add].getH(subIndex);
                    hashtype h = reformKey(i, rem, AS);
                    keytype k = RHASH_INVERSE(HFSIZE, label, h);

                    printf("|%-10i|%-16" PRIu64 "|%-6i|%-20" PRIu64 "|%-6i|\n", i, T[real_add].getR(subIndex), T[real_add].getO(subIndex), k, T[real_add].getH(subIndex));
                }
            }
            printf("------------------------------------------------------------\n");
        }


    public:
        /**
         * Constructor
         */
        ClearyCuckoo(): valSize(64), valsPerEntry((int)(ENTRYSIZE2 / valSize)) {}

        ClearyCuckoo(int adressSize) : ClearyCuckoo(adressSize, 4, 64) {}

        ClearyCuckoo(int adressSize, int hashNumber, int value_size):
            valSize(value_size), valsPerEntry( (int) (ENTRYSIZE2 / value_size) ) {
            printf("Constructor\n");
            //Init variables
            AS = adressSize;
            RS = HS-AS;
            tablesize = (int) pow(2,AS);
            numEntries = (int)(tablesize / valsPerEntry);

            int queueSize = std::max(100, (int)(tablesize / 10));

            hn = hashNumber;


            //Allocating Memory for tables
#ifdef GPUCODE
            gpuErrchk(cudaMallocManaged(&T, numEntries * sizeof(ClearyCuckooEntryCompact<addtype,remtype>)));
            gpuErrchk(cudaMallocManaged(&hashlist, hn * sizeof(int)));
            gpuErrchk(cudaMallocManaged((void**)&rehashQueue, sizeof(SharedQueue<int>)));
#else
            T = new ClearyCuckooEntryCompact<addtype, remtype>[tablesize];
            hashlist = new int[hn];
#endif

            //Default MAXLOOPS Value
            //1.82372633e+04 -2.60749645e+02  1.76799265e-02 -1.80594901e+04
            const double A = 18237.2633;
            const double x0 = -260.749645;
            const double k = .0176799265;
            const double off = -18059.4901;

            MAXLOOPS = ceil((A / (1.0 + exp(-k * (((double)AS) - x0)))) + off);

            //Init table entries
            for(int i=0; i<numEntries; i++){
                new (&T[i]) ClearyCuckooEntryCompact<addtype, remtype>(valSize);
            }

            //Create HashList
            createHashList(hashlist);
        }

        /**
         * Destructor
         */
        ~ClearyCuckoo(){
            printf("Destructor\n");
#ifdef GPUCODE
            gpuErrchk(cudaFree(T));
            gpuErrchk(cudaFree(hashlist));

#else
            delete[] T;
            delete[] hashlist;

#endif
        }

        //Public insertion call
        GPUHEADER_D
#ifdef GPUCODE
        result insert(uint64_cu k){
#else
        result insert(uint64_cu k, SpinBarrier* barrier) {
#endif
            printf("Insert\n");

            return insertIntoTable(k, T, hashlist, 0);
        };

        //Method to check for duplicates after insertions
        GPUHEADER
        void removeDuplicates(keytype k) {
            printf("RemoveDuplicates\n");
            //To store whether value was already encountered
            bool found = false;

            for (int i = 0; i < hn; i++) {
                uint64_cu hashed1 = RHASH(HFSIZE, hashlist[i], k);
                addtype add = getAdd(hashed1, AS);
                remtype rem = getRem(hashed1, AS);

                addtype real_add = (addtype)(add / valsPerEntry);
                addtype subIndex = (addtype)(add % valsPerEntry);

                if (T[real_add].getH(subIndex) == hashlist[i] && T[real_add].getR(subIndex) == rem && T[real_add].getO(subIndex)) {
                    //If value was already found
                    if (found) {
                        //Mark as not occupied
                        T[real_add].setO(false, subIndex);
                    }
                    //Mark value as found
                    found = true;
                }
            }
        }

        //Public Lookup call
        GPUHEADER
        bool lookup(uint64_cu k){
            return lookup(k, T);
        };

        //Clear all Table Entries
        GPUHEADER
        void clear(){
            for(int i=0; i<numEntries; i++){
                new (&T[i]) ClearyCuckooEntryCompact<addtype, remtype>(valSize);
            }
        }

        //Get the size of the Table
        GPUHEADER
        int getSize(){
            return tablesize;
        }

        //Return a copy of the hashlist
        GPUHEADER
        int* getHashlistCopy() {
            int* res = new int[hn];
            for (int i = 0; i < hn; i++) {
                res[i] = hashlist[i];
            }
            return res;
        }

        //Transform a vector to a list
        GPUHEADER_H
        std::vector<uint64_cu> toList() {
            std::vector<uint64_cu> list;
            for (int i = 0; i < tablesize; i++) {

                addtype real_add = (addtype)(i / valsPerEntry);
                addtype subIndex = (addtype)(i % valsPerEntry);

                if (T[real_add].getO(subIndex)) {
                    hashtype h_old = reformKey(i, T[real_add].getR(subIndex), AS);
                    keytype x = RHASH_INVERSE(HFSIZE, T[real_add].getH(subIndex), h_old);
                    list.push_back(x);
                }
            }
            return list;
        }

        //Iterate through all entries and do a read
        void readEverything(int N) {
            int j = 0;
            int step = 1;

            if (N < tablesize) {
                step = std::ceil(((float)tablesize) / ((float)N));
            }

            for (int i = 0; i < N; i+=step) {
                int index = i % tablesize;
                j += T[(addtype) (index/valsPerEntry)].getR((index % valsPerEntry));
            }

            if (j != 0) {
                printf("Not all Zero\n");
            }
        }


        //Public print call
        GPUHEADER
        void print(){
            printf("Hashlist:");
            for (int i = 0; i < hn; i++) {
                printf("%i,", hashlist[i]);
            }
            printf("\n");
            print(T);
        }

        //Method used for debugging
        GPUHEADER
        void debug(uint64_cu i) {

        }

        //Set the number of rehashes allowed
        void setMaxRehashes(int x){
            MAXREHASHES = x;
        }

        //Set the number of loops allowed
        void setMaxLoops(int x){
            MAXLOOPS = x;
        }

        //Get the number of hashes
        int getHashNum() {
            return hn;
        }

};

//Method to fill ClearyCuckoo table
GPUHEADER_G
#ifdef GPUCODE
void fillClearyCuckoo(int N, uint64_cu* vals, ClearyCuckoo* H, int* failFlag=nullptr, addtype begin = 0, int id = 0, int s = 1)
#else
void fillClearyCuckoo(int N, uint64_cu* vals, ClearyCuckoo* H, SpinBarrier* barrier, int* failFlag = nullptr, addtype begin = 0, int id = 0, int s = 1)
#endif
{
#ifdef GPUCODE
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x;
#else
    int index = id;
    int stride = s;
#endif
    //printf("Thread %i Starting\n", getThreadID());
    for (int i = index + begin; i < N + begin; i += stride) {
        //printf("\t\t\t\t\t\t\t%i\n", i);
#ifdef GPUCODE
        if (H->insert(vals[i]) == FAILED) {
#else
        if (!(H->insert(vals[i], barrier))) {
#endif
            if (failFlag != nullptr) {
                (*failFlag) = true;
            }
            break;
        }
    }
    //printf("Insertions %i Over\n", getThreadID());
#ifdef DUPCHECK
#ifndef GPUCODE
    barrier->signalThreadStop();
#endif
#endif
}


#ifdef GPUCODE
//Method to fill ClearyCuckoo table with a failCheck on every insertion
GPUHEADER_G
void fillClearyCuckoo(int N, uint64_cu* vals, ClearyCuckoo* H, addtype* occupancy, int* failFlag, int id = 0, int s = 1)
{
#ifdef GPUCODE
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x;
#else
    int index = id;
    int stride = s;
#endif

    for (int i = index; i < N; i += stride) {
        if (failFlag[0]) {
            break;
        }
#ifdef GPUCODE
        if (H->insert(vals[i]) == FAILED) {
#else
        if (H->insert(vals[i]) == FAILED) {
#endif
            atomicCAS(&(failFlag[0]), 0, 1);
            break;
        }
        atomicAdd(&occupancy[0], 1);
    }
}
#endif

#ifndef GPUCODE
//Method to fill ClearyCuckoo table with a failCheck on every insertion
GPUHEADER_G
void fillClearyCuckoo(int N, uint64_cu* vals, ClearyCuckoo* H, SpinBarrier* barrier, std::atomic<addtype>* occupancy, std::atomic<bool>* failFlag, int id = 0, int s = 1)
{
#ifdef GPUCODE
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x;
#else
    int index = id;
    int stride = s;
#endif

    for (int i = index; i < N; i += stride) {
        if ((*failFlag).load()) {
            break;
        }
#ifdef GPUCODE
        if (H->insert(vals[i], stride) == FAILED) {
#else
        if (H->insert(vals[i], stride, barrier) == FAILED) {
#endif
            (*failFlag).store(true);
            break;
        }
        (*occupancy).fetch_add(1);
    }
#ifdef DUPCHECK
#ifndef GPUCODE
    barrier->signalThreadStop();
#endif
#endif
}
#endif

//Method to check whether a ClearyCuckoo table contains a set of values
GPUHEADER_G
void checkClearyCuckoo(int N, uint64_cu* vals, ClearyCuckoo* H, bool* res, int id = 0, int s = 1)
{
#ifdef GPUCODE
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x;
#else
    int index = id;
    int stride = s;
#endif

    for (int i = index; i < N; i += stride) {
        if (!(H->lookup(vals[i]))) {
            res[0] = false;
        }
    }
}

//Method to do lookups in a ClearyCuckoo table on an array of values
GPUHEADER_G
void lookupClearyCuckoo(int N, int start, int end, uint64_cu* vals, ClearyCuckoo* H, int id = 0, int s = 1) {
#ifdef GPUCODE
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x;
#else
    int index = id;
    int stride = s;
#endif

    for (int i = index; i < N; i += stride) {
        H->lookup(vals[(i + start) % end]);
    }
}

//Method to fill ClearyCuckoo table
GPUHEADER_G
void dupCheckClearyCuckoo(int N, uint64_cu* vals, ClearyCuckoo* H, addtype begin = 0)

{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x;

    //printf("Thread %i Starting\n", getThreadID());
    for (int i = index + begin; i < N + begin; i += stride) {
        H->removeDuplicates(vals[i]);
    }
    //printf("Insertions %i Over\n", getThreadID());
}