#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <math.h>
#include <iterator>
#include <set>
#include <inttypes.h>
#include <atomic>
#include <random>

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
#include "ClearyCuckooEntry.cu"




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

        //Vars at Construction
        int AS;                    //AdressSize
        int RS;                         //RemainderSize
        int tablesize;
        int occupancy = 0;

        //Hash tables
        ClearyCuckooEntry<addtype, remtype>* T;

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

        GPUHEADER
        void createHashList(int* list) {
            //printf("\tCreating Hashlist\n");
            for (int i = 0; i < hn; i++) {
                list[i] = i;
            }
            return;
        }

        GPUHEADER
        void iterateHashList(int* list) {
            //printf("\tUpdating Hashlist\n");
            for (int i = 0; i < hn; i++) {
                list[i] = (list[i]+1+i)%32;
            }
            return;
        }

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
        GPUHEADER_D
        bool setFlag(int* loc, int val, bool strict=true) {
            int val_i = val == 0 ? 1 : 0;
            ////printf("%i:\t\tAttempting exch val:%i val_i:%i actual:%i, strict:%i\n", getThreadID(), val ,val_i, (*loc), strict);
            //TODO Remove
            int counter = 0;
            while (true) {
                counter++;
                //assert(counter < 300);
                if(failFlag){
                  //printf("%i:\t\tFailFlag\n", getThreadID());
                  return false;
                }
                //In devices, atomically exchange
                if(!strict){
                ////printf("%i:\t\t\tAtomicCAS:%i val_i:%i actual:%i, strict:%i\n", getThreadID(), val ,val_i, (*loc), strict);
                }
                uint64_cu res = atomicCAS(loc, val_i, val);
                if(!strict){
                ////printf("%i:\t\t\tAtomicCASDone\n", getThreadID());
                }
                //Make sure the value hasn't changed in the meantime
                if ( (res != val_i) && strict) {
                    continue;
                }
                ////printf("%i:\t\t:Flag Set to %i\n", getThreadID(), val);
                return true;
            }
        }

#else
        GPUHEADER
        bool setFlag(std::atomic<int>* loc, int val, bool strict=true) {
            int val_i = val == 0 ? 1 : 0;

            while (true) {
                if(failFlag){
                  return false;
                }
                //printf("%i:\t:Attempting CAS\n", getThreadID());
                if (std::atomic_compare_exchange_strong(loc, &val_i, val)) {
                    //printf("%i:\t:Flag Set\n", getThreadID());
                    return true;
                }else if(!strict){
                    return true;
                }
            }
        }
#endif



        /**
         * Function to label which hash function was used on this value
         **/
        GPUHEADER_D
        bool insertIntoTable(keytype k, ClearyCuckooEntry<addtype, remtype>* T, int* hs, int depth=0){
            ////printf("%i:\t\t:InsertintoTable\n", getThreadID());
            keytype x = k;
            int hash = hs[0];

            //If the key is already inserted don't do anything
            if (lookup(k, T)) {
                //printf("\tAlready Exists\n");
                return false;
            }

            //Start the iteration
            int c = 0;

            while(c < MAXLOOPS){
                //printf("%i:Loop %i ", getThreadID(), c);
                //Get the key of k
                //printf("%i:\t:GetKey\n", getThreadID());
                hashtype hashed1 = RHASH(hash, x);
                addtype add = getAdd(hashed1, AS);
                remtype rem = getRem(hashed1, AS);

                //Place new value
                //printf("%i:\t:Placing New Value at %" PRIu32 "\n", getThreadID(), add);
                ClearyCuckooEntry<addtype, remtype> entry(rem, hash, true, false);
                //printf("%i:\t:ExchValue at %" PRIu32 " \n", getThreadID(), add);
                T[add].exchValue(&entry);

                //Store the old value
                //printf("%i:\t:Store Old Value at %" PRIu32 "\n", getThreadID(), add);
                remtype temp = entry.getR();
                bool wasoccupied = entry.getO();
                int oldhash = entry.getH();

                //printf("%i:\t:Occ Check\n", getThreadID());
                //If the first spot was open return
                if(!wasoccupied){
                    //printf("%i:\t:WasEmpty\n", getThreadID());
                    return true;
                }

                //Otherwise rebuild the original key
                //printf("%i:\t:Rebuild Key\n", getThreadID());
                hashtype h_old = reformKey(add, temp, AS);
                x = RHASH_INVERSE(oldhash, h_old);

                //Hash with the next hash value
                //printf("%i:\t:GetNextHash\n", getThreadID());
                hash = getNextHash(hs, oldhash);

                c++;
            }

#ifdef REHASH
            //printf("Pushing %" PRIu64 "\n", x);
            rehashQueue->push(x);
            if(depth>0){return false;}
            //If MAXLOOPS is reached rehash the whole table
            if(!rehash()){
              //printf("Outer Rehash Call\n");
                //If rehash fails, return
                return false;
            }
            return true;
#else
            return false;
#endif
        };

#ifdef REHASH
        GPUHEADER_D
        bool rehash(int depth){
            //Prevent recursion of rehashing
            if(depth >0){return false;}

            for (int i = 0; i < tablesize; i++) {
                //printf("Checking %i\n", i);
                //Check Occupied
                if (!T[i].getO()) {
                    continue;
                }
                //Check if permissible under new hashlist
                if(!containsHash(hashlist, T[i].getH())) {
                    //Reform Val
                    hashtype h_old = reformKey(i, T[i].getR(), AS);
                    keytype x = RHASH_INVERSE(T[i].getH(), h_old);
                    //Clear Entry
                    T[i].clear();
                    //Reinsert using new hashlist
                    //printf("Reinserting %" PRIu64 "\n", x);
                    if (!insertIntoTable(x, T, hashlist, depth+1)) {
                        return false;
                    }
                }
            }

            return true;
        };
#endif

        GPUHEADER
        bool lookup(uint64_cu k, ClearyCuckooEntry<addtype, remtype>* T){
            //printf("\t\tLookup %" PRIu64 "\n", k);
            for (int i = 0; i < hn; i++) {
                uint64_cu hashed1 = RHASH(hashlist[i], k);
                addtype add = getAdd(hashed1, AS);
                remtype rem = getRem(hashed1, AS);
                if (T[add].getR() == rem && T[add].getO()) {
                    return true;
                }
            }
            //printf("\t\tNone Found\n");
            return false;
        };

        GPUHEADER
        void print(ClearyCuckooEntry<addtype, remtype>* T) {
            ////printf("----------------------------------------------------------------\n");
            ////printf("|    i     |     R[i]       | O[i] |        key         |label |\n");
            ////printf("----------------------------------------------------------------\n");
            ////printf("Tablesize %i\n", tablesize);
            for (int i = 0; i < tablesize; i++) {
                if (T[i].getO()) {
                    remtype rem = T[i].getR();
                    int label = T[i].getH();
                    hashtype h = reformKey(i, rem, AS);
                    keytype k = RHASH_INVERSE(label, h);

                    ////printf("|%-10i|%-16" PRIu64 "|%-6i|%-20" PRIu64 "|%-6i|\n", i, T[i].getR(), T[i].getO(), k, T[i].getH());
                }
            }
            ////printf("------------------------------------------------------------\n");
        }


    public:
        /**
         * Constructor
         */
        ClearyCuckoo() {}

        ClearyCuckoo(int adressSize) : ClearyCuckoo(adressSize, 4) {}

        ClearyCuckoo(int adressSize, int hashNumber){
            //printf("Creating ClearyCuckoo Table\n");
            AS = adressSize;
            RS = HS-AS;
            tablesize = (int) pow(2,AS);

            int queueSize = std::max(100, (int)(tablesize / 10));

            hn = hashNumber;
#ifdef GPUCODE
            failFlag = false;
            rehashFlag = false;
#else
            failFlag.store(false);
            rehashFlag.store(false);
#endif
            //printf("\tAllocating Memory %" PRIu32 "\n", tablesize);
            #ifdef GPUCODE
            gpuErrchk(cudaMallocManaged(&T, tablesize * sizeof(ClearyCuckooEntry<addtype,remtype>)));
            gpuErrchk(cudaMallocManaged(&hashlist, hn * sizeof(int)));
            gpuErrchk(cudaMallocManaged((void**)&rehashQueue, sizeof(SharedQueue<int>)));
            new (rehashQueue) SharedQueue<int>(queueSize);
            #else
            T = new ClearyCuckooEntry<addtype, remtype>[tablesize];
            hashlist = new int[hn];
            rehashQueue = new SharedQueue<keytype>(queueSize);
            #endif

            //Default MAXLOOPS Value
            MAXLOOPS = round(104.49226591 * log(3.80093894 * (AS - 3.54270024)) - 88.47034412);

            //printf("\tInitializing Entries\n");
            for(int i=0; i<tablesize; i++){
                new (&T[i]) ClearyCuckooEntry<addtype, remtype>();
            }

            createHashList(hashlist);
            //printf("\tDone\n");

        }

        /**
         * Destructor
         */
        ~ClearyCuckoo(){
            //printf("Destroying Table\n");
            #ifdef GPUCODE
            gpuErrchk(cudaFree(T));
            gpuErrchk(cudaFree(hashlist));
            gpuErrchk(cudaFree(rehashQueue));
            #else
            delete[] T;
            delete[] hashlist;
            delete rehashQueue;
            #endif
        }

        GPUHEADER_D
        bool insert(uint64_cu k){
            //Succesful Insertion
            ////printf("%i:\tInserting val %" PRIu64 "\n", getThreadID(), k);
#ifdef REHASH
#ifdef GPUCODE
            if (failFlag) {
                ////printf("%i:\t:Check FailFlag\n", getThreadID());
                return false;
            }while (rehashFlag) {
                __syncthreads();
                if (failFlag) {
                    return false;
                }
            }
            ////printf("%i:\t:Rehash/Fail Flag Not Set\n", getThreadID());
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
            if(insertIntoTable(k,T, hashlist,0)){
                //Reset the Hash Counter
#ifdef REHASH
                hashcounter = 0;
#endif
                //print();
                /*
#ifdef GPUCODE
                atomicAdd(&occupation, 1);
#else
                occupation += 1;
#endif
                */
                return true;
            }
            ////printf("Set FailFlag\n");
#ifdef REHASH
            setFlag(&failFlag, 1, false);
#endif
            ////printf("Insert Fail\n");
            return false;
        };

#ifdef REHASH
        GPUHEADER_D
        bool rehash(){
            ////printf("%i:\t:Start Rehash\n", getThreadID());
            //printf("Rehash call %i\n", hashcounter);

            if(!setFlag(&rehashFlag, 1)){
                return false;
            }
            //printf("%i:\t\t:Rehash Flag Set\n", getThreadID());

            //Looping Rehash
            while(hashcounter<MAXREHASHES){
                //printf("\tRehash call %i\n", hashcounter);
                iterateHashList(hashlist);
                hashcounter++;

                if (!rehash(0)) {
                    continue;
                }

                ////printf("Inserting RehashQueue\n");
                //rehashQueue->print();
                while (!rehashQueue->isEmpty()) {
                    keytype next = rehashQueue->pop();
                    if (!insertIntoTable(next, T, hashlist, 1)) { break; }
                }

                if (rehashQueue->isEmpty()) {
                    break;
                }

            };


            //If counter tripped return
            if(hashcounter >= MAXREHASHES){
                ////printf("\t +++++++++++++++++++++++++++++++++++++++Rehash Loop FAIL++++++++++++++++++++++++++++++++++++++\n");
                setFlag(&failFlag, 1, false);
                __syncthreads();
                ////printf("Returning\n");
                return false;
            }
            //Rehash done
            ////printf("\t +++++++++++++++++++++++++++++++++++++++Rehash Loop SUCCESS++++++++++++++++++++++++++++++++++++++\n");



            setFlag(&rehashFlag, 0);
            __syncthreads();

            return true;
        }
#endif

        GPUHEADER
        bool lookup(uint64_cu k){
            return lookup(k, T);
        };

        GPUHEADER
        void clear(){
            for(int i=0; i<tablesize; i++){
                new (&T[i]) ClearyCuckooEntry<addtype, remtype>();
            }
        }

        GPUHEADER
        int getSize(){
            return tablesize;
        }

        GPUHEADER
        int* getHashlistCopy() {
            int* res = new int[hn];
            for (int i = 0; i < hn; i++) {
                res[i] = hashlist[i];
            }
            return res;
        }

        GPUHEADER_H
        std::vector<uint64_cu> toList() {
            std::vector<uint64_cu> list;
            for (int i = 0; i < tablesize; i++) {
                if (T[i].getO()) {
                    hashtype h_old = reformKey(i, T[i].getR(), AS);
                    keytype x = RHASH_INVERSE(T[i].getH(), h_old);
                    list.push_back(x);
                }
            }
            return list;
        }

        void readEverything(int N) {
            int j = 0;
            int step = 1;

            if (N < tablesize) {
                step = std::ceil(((float)tablesize) / ((float)N));
            }

            for (int i = 0; i < N; i+=step) {
                j += T[i%tablesize].getR();
            }

            if (j != 0) {
                //printf("Not all Zero\n");
            }
        }



        GPUHEADER
        void print(){
            ////printf("Hashlist:");
            for (int i = 0; i < hn; i++) {
                ////printf("%i,", hashlist[i]);
            }
            ////printf("\n");
            print(T);
        }

        GPUHEADER
        void debug(uint64_cu i) {

        }

        void setMaxRehashes(int x){
            MAXREHASHES = x;
        }

        void setMaxLoops(int x){
            MAXLOOPS = x;
        }

        int getHashNum() {
            return hn;
        }

};

GPUHEADER_G
void fillClearyCuckoo(int N, uint64_cu* vals, ClearyCuckoo* H, int* failFlag=nullptr, addtype begin = 0, int id = 0, int s = 1)
{
    //printf("\tThread Started\n");
#ifdef GPUCODE
    int index = threadIdx.x;
    int stride = blockDim.x;
#else
    int index = id;
    int stride = s;
#endif
    for (int i = index + begin; i < N + begin; i += stride) {
        //printf("Inserting %" PRIu64 "\n", vals[i]);
        if (!(H->insert(vals[i]))) {
            if (failFlag != nullptr) {
                (*failFlag) = true;
            }
            break;
        }
    }
}

#ifdef GPUCODE
GPUHEADER_G
void fillClearyCuckoo(int N, uint64_cu* vals, ClearyCuckoo* H, addtype* occupancy, int* failFlag, int id = 0, int s = 1)
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
        if (!(H->insert(vals[i]))) {
            atomicCAS(&(failFlag[0]), 0, 1);
            break;
        }
        atomicAdd(&occupancy[0], 1);
    }
}
#endif

#ifndef GPUCODE
GPUHEADER_G
void fillClearyCuckoo(int N, uint64_cu* vals, ClearyCuckoo* H, std::atomic<addtype>* occupancy, std::atomic<bool>* failFlag, int id = 0, int s = 1)
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
        if (!(H->insert(vals[i]))) {
            (*failFlag).store(true);
            break;
        }
        (*occupancy).fetch_add(1);
    }
}
#endif

GPUHEADER_G
void checkClearyCuckoo(int N, uint64_cu* vals, ClearyCuckoo* H, bool* res, int id = 0, int s = 1)
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
            //printf("\t\tVal %" PRIu64 " Missing\n", vals[i]);
            res[0] = false;
        }
    }
}


GPUHEADER_G
void lookupClearyCuckoo(int N, int start, int end, uint64_cu* vals, ClearyCuckoo* H, int id = 0, int s = 1) {
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