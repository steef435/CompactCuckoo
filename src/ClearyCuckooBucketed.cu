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


class ClearyCuckooBucketed: HashTable{

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

        //Vars at Construction
        int RS;                         //RemainderSize
        int AS;                         //AdressSize
        int B;                          //NumBuckets
        int Bs;                         //BucketSize
        
        int tablesize;
        int occupancy = 0;

        //Hash tables
        ClearyCuckooEntry<addtype, remtype>* T;

        int hashcounter = 0;

        //Hash function ID
        int hn;
        int* hashlist;

        //Bucket Variables
        int* bucketIndex;

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
            ////printf("\tUpdating Hashlist\n");
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
                ////printf("%i:\t:Attempting CAS\n", getThreadID());
            if (std::atomic_compare_exchange_strong(loc, &val_i, val)) {
                ////printf("%i:\t:Flag Set\n", getThreadID());
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
        bool insertIntoTable(keytype k, ClearyCuckooEntry<addtype, remtype>* T, int* hs, int depth=0){
            keytype x = k;
            int hash = hs[0];

            //If the key is already inserted don't do anything
            if (lookup(k, T)) {
                return false;
            }
            //Start the iteration
            int c = 0;

            while (c < MAXLOOPS) {
                //Get the add/rem of k
                hashtype hashed1 = RHASH(hash, x);
                addtype add = getAdd(hashed1, AS);
                remtype rem = getRem(hashed1, AS);

                //printf("\t\thashed1 %" PRIu64 "\n", hashed1);
                //printf("\t\tadd %" PRIu32 "\n", add);
                //printf("\t\trem %" PRIu64 "\n\n", rem);

                //Exchange Values
                ClearyCuckooEntry<addtype, remtype> entry(rem, hash, true, false);
                T[ (add*Bs) + (bucketIndex[add])].exchValue(&entry);
                //Iterate Buvketindex
                bucketIndex[add] = (bucketIndex[add] + 1) % Bs;


                //Store the old value
                remtype temp = entry.getR();
                bool wasoccupied = entry.getO();
                int oldhash = entry.getH();
                //printf("\t\t\told: rem:%" PRIu64 " Occ:%i hash:%i \n", temp, wasoccupied, oldhash);

                //If the old val was empty return
                if (!wasoccupied) {
                    return true;
                }

                //Otherwise rebuild the original key
                hashtype h_old = reformKey(add, temp, AS);
                x = RHASH_INVERSE(oldhash, h_old);

                //Hash with the next hash value
                hash = getNextHash(hs, oldhash);

                c++;
            }

#ifdef REHASH
            //If loops fail call rehash
            rehashQueue->push(x);
            if (depth > 0) { return false; }
            //If MAXLOOPS is reached rehash the whole table
            if (!rehash()) {
                //If rehash fails, return
                return false;
            }
            return true;
#else
            return false;
#endif
        };


        //Method to check for duplicates after insertions
        GPUHEADER
        void removeDuplicates(keytype k) {
            //To store whether value was already encountered
            bool found = false;

            for (int i = 0; i < hn; i++) {
                uint64_cu hashed1 = RHASH(hashlist[i], k);
                addtype add = getAdd(hashed1, AS);
                remtype rem = getRem(hashed1, AS);
                for (int j = 0; j < Bs; j++) {

                    if (T[add*Bs +j].getH() == hashlist[i] && T[add*Bs + j].getR() == rem && T[add*Bs +j].getO()) {
                        //If value was already found
                        if (found) {
                            //Mark as not occupied
                            T[add*Bs + j].setO(false);
                        }
                        //Mark value as found
                        found = true;
                    }
                }
            }
        }

        //Lookup internal method
        GPUHEADER
        bool lookup(uint64_cu k, ClearyCuckooEntry<addtype, remtype>* T){

        };

        GPUHEADER
        void print(ClearyCuckooEntry<addtype, remtype>* T) {
            printf("----------------------------------------------------------------\n");
            printf("|    i     |     R[i]       | O[i] |        key         |label |\n");
            printf("----------------------------------------------------------------\n");
            printf("Tablesize %i\n", tablesize);
            
            for (int i = 0; i < B; i++) {
                printf("----------------------------------------------------------------\n");
                printf("|                   Bucket %i                                   \n", i);
                printf("----------------------------------------------------------------\n");
                for (int j = 0; j < Bs; j++) {
                    remtype rem = T[i*Bs + j].getR();
                    int label = T[i*Bs + j].getH();
                    hashtype h = reformKey(i, rem, AS);
                    keytype k = RHASH_INVERSE(label, h);

                    printf("|%-10i|%-16" PRIu64 "|%-6i|%-20" PRIu64 "|%-6i|\n", j, T[i*Bs + j].getR(), T[i*Bs + j].getO(), k, T[i*Bs + j].getH());
                }
            }

            printf("------------------------------------------------------------\n");
        }


    public:
        /**
         * Constructor
         */
        ClearyCuckooBucketed() {}

        ClearyCuckooBucketed(int addressSize, int bucketSize,  int hashNumber){
            //printf("Constructor\n");
            //Init variables
            AS = addressSize;
            B = (int)pow(2, AS);
            RS = HS-AS;
            Bs = bucketSize;
            tablesize = B*Bs;
            
            int queueSize = std::max(100, (int)(tablesize / 10));

            hn = hashNumber;

            //Allocating Memory for tables
            //printf("\tAlloc Mem\n");
#ifdef GPUCODE
            gpuErrchk(cudaMallocManaged(&T, tablesize * sizeof(ClearyCuckooEntry<addtype, remtype>)));
            gpuErrchk(cudaMallocManaged(&hashlist, hn * sizeof(int)));
            gpuErrchk(cudaMallocManaged((void**)&rehashQueue, sizeof(SharedQueue<int>)));
            gpuErrchk(cudaMallocManaged(&bucketIndex, Bs * sizeof(int)));
#else
            T = new ClearyCuckooEntry<addtype, remtype>[numBuckets*Bs + bucketSize];
            hashlist = new int[hn];
            bucketIndex = new int[Bs];
#endif
            //printf("\tInit Entries\n");
            //Init table entries
            for(int i=0; i<B; i++){
                for (int j = 0; j < Bs; j++) {
                    //printf("\t\tEntry %i %i\n",i, j);
                    new (&T[i*Bs + j]) ClearyCuckooEntry<addtype, remtype>();
                }
                bucketIndex[i] = 0;
            }
            //printf("\tCreate Hashlist\n");
            //Create HashList
            createHashList(hashlist);
            //printf("\tInit Complete\n");
        }

        /**
         * Destructor
         */
        ~ClearyCuckooBucketed(){
            //printf("Destructor\n");
            #ifdef GPUCODE
            gpuErrchk(cudaFree(T));
            gpuErrchk(cudaFree(hashlist));
            gpuErrchk(cudaFree(bucketIndex));

            #else
            delete[] T;
            delete[] hashlist;
            delete[] bucketIndex;
            #endif
        }

        //Public insertion call
        GPUHEADER_D
#ifdef GPUCODE
            bool insert(uint64_cu k, int numThreads) {
#else
            bool insert(uint64_cu k, int numThreads, SpinBarrier * barrier) {
#endif
            //printf("Insert %" PRIu64 "\n", k);
#ifdef REHASH
#ifdef GPUCODE
            
            //Need to check if rehash or Fail has occurred
            if (failFlag) {
                return false;
            }
            int count = 0;
            while (rehashFlag) {
                if (count > 10000) {
                    count = 0;
                }
                if (failFlag) {
                    return false;
                }
                count++;
            }
#else
            if (failFlag.load()) {
                return false;
            }
            while (rehashFlag.load()) {
                if (failFlag.load()) {
                    return false;
                }
            }
#endif
#endif
            //Stores success/failure of rehash
            bool finalRes = false;
            if (insertIntoTable(k, T, hashlist, 0)) {
                //Reset the Hash Counter
#ifdef REHASH
                hashcounter = 0;
#endif

                finalRes = true;
            }
#ifdef REHASH
            //If insert failed, set failFlag
            if (!finalRes) {
                while (!setFlag(&failFlag, 1, false)) {}
            }
#endif

            //Duplicate Check Phase
#ifdef DUPCHECK
#ifdef GPUCODE
            __syncthreads();
#else
            barrier->Wait();
#endif
            //Do duplicate Check if insertion was successful
            if (finalRes) {
                removeDuplicates(k);
            }

#ifdef GPUCODE
            __syncthreads();
#else
            barrier->Wait();
#endif
#endif

            return finalRes;
        };

#ifdef REHASH
        GPUHEADER_D
        bool rehash(){

        }
#endif

        //Public Lookup call
        GPUHEADER
        bool lookup(uint64_cu k){
            //printf("Lookup\n");
            //Iterate over hash functions and check if found
            for (int i = 0; i < hn; i++) {
                uint64_cu hashed1 = RHASH(hashlist[i], k);
                addtype add = getAdd(hashed1, AS);
                remtype rem = getRem(hashed1, AS);
                for (int j = 0; j < Bs; j++) {
                    if (T[add * Bs + j].getR() == rem && T[add*Bs + j].getO()) {
                        //printf("\t\t Lookup Success\n");
                        return true;
                    }
                }
            }
            //printf("\t\t Lookup Failed\n");
            return false;
        };

        //Clear all Table Entries
        GPUHEADER
        void clear(){
            for (int i = 0; i < B; i++) {
                for (int j = 0; j < Bs; j++) {
                    new (&T[i*Bs + j]) ClearyCuckooEntry<addtype, remtype>();
                }
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
                for (int j = 0; j < tablesize; j++) {
                    if (T[i * Bs + j].getO()) {
                        hashtype h_old = reformKey(i, T[i * Bs + j].getR(), AS);
                        keytype x = RHASH_INVERSE(T[i * Bs + j].getH(), h_old);
                        list.push_back(x);
                    }
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
                for (int k = 0; k < Bs; k++) {
                    j += T[(i * Bs + j) % tablesize].getR();
                }
            }

            if (j != 0) {
                //printf("Not all Zero\n");
            }
        }


        //Public print call
        GPUHEADER
        void print(){
            //printf("Hashlist:");
            for (int i = 0; i < hn; i++) {
                //printf("%i,", hashlist[i]);
            }
            //printf("\n");
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

//Method to fill ClearyCuckooBucketedtable
GPUHEADER_G
#ifdef GPUCODE
void fillClearyCuckooBucketed(int N, uint64_cu* vals, ClearyCuckooBucketed* H, int* failFlag=nullptr, addtype begin = 0, int id = 0, int s = 1)
#else
void fillClearyCuckooBucketed(int N, uint64_cu* vals, ClearyCuckooBucketed* H, SpinBarrier* barrier, int* failFlag = nullptr, addtype begin = 0, int id = 0, int s = 1)
#endif
{
#ifdef GPUCODE
    int index = threadIdx.x;
    int stride = blockDim.x;
#else
    int index = id;
    int stride = s;
#endif
    ////printf("Thread %i Starting\n", getThreadID());
    for (int i = index + begin; i < N + begin; i += stride) {
#ifdef GPUCODE
        if (!(H->insert(vals[i], stride))) {
#else
        if (!(H->insert(vals[i], stride, barrier))) {
#endif
            if (failFlag != nullptr) {
                (*failFlag) = true;
            }
            break;
        }

        H->print();
    }
    ////printf("Insertions %i Over\n", getThreadID());
#ifdef DUPCHECK
#ifndef GPUCODE
    barrier->signalThreadStop();
#endif
#endif
}


#ifdef GPUCODE
//Method to fill ClearyCuckooBucketedtable with a failCheck on every insertion
GPUHEADER_G
void fillClearyCuckooBucketed(int N, uint64_cu* vals, ClearyCuckooBucketed * H, addtype* occupancy, int* failFlag, int id = 0, int s = 1)
{
#ifdef GPUCODE
    int index = threadIdx.x;
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
        if (!(H->insert(vals[i], stride))) {
#else
        if (!(H->insert(vals[i], stride))) {
#endif
            atomicCAS(&(failFlag[0]), 0, 1);
            break;
        }
        atomicAdd(&occupancy[0], 1);
    }
}
#endif

#ifndef GPUCODE
//Method to fill ClearyCuckooBucketed table with a failCheck on every insertion
GPUHEADER_G
void fillClearyCuckooBucketed(int N, uint64_cu* vals, ClearyCuckooBucketed * H, SpinBarrier* barrier, std::atomic<addtype>* occupancy, std::atomic<bool>* failFlag, int id = 0, int s = 1)
{
#ifdef GPUCODE
    int index = threadIdx.x;
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
        if (!(H->insert(vals[i], stride))) {
#else
        if (!(H->insert(vals[i], stride, barrier))) {
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

//Method to check whether a ClearyCuckooBucketed table contains a set of values
GPUHEADER_G
void checkClearyCuckooBucketed(int N, uint64_cu* vals, ClearyCuckooBucketed* H, bool* res, int id = 0, int s = 1)
{
#ifdef GPUCODE
    int index = threadIdx.x;
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

//Method to do lookups in a ClearyCuckooBucketed table on an array of values
GPUHEADER_G
void lookupClearyCuckooBucketed(int N, int start, int end, uint64_cu* vals, ClearyCuckooBucketed* H, int id = 0, int s = 1) {
#ifdef GPUCODE
    int index = threadIdx.x;
    int stride = blockDim.x;
#else
    int index = id;
    int stride = s;
#endif

    for (int i = index; i < N; i += stride) {
        H->lookup(vals[(i + start) % end]);
    }
}
