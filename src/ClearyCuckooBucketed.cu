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

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

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


//Taken from Better GPUs
template <int tile_sz>
struct bucket {
    // Constructor to load the key-value pair of the bucket 2
    GPUHEADER
    bucket(ClearyCuckooEntry<addtype, remtype>* ptr, cg::thread_block_tile<tile_sz> tile, int bucketIndex, int bucketSize) : ptr_(ptr), tile_(tile) {
        tIndex = bucketIndex * bucketSize + tile_.thread_rank();
        lane_pair_ = ptr[tIndex];
    }

    // Compute the load of the bucket
    GPUHEADER
    int compute_load() {
        auto load_bitmap = tile_.ballot((lane_pair_.getO()));
        return __popc(load_bitmap);
    }

    // Find the value associated with a key
    GPUHEADER
    bool find(const remtype rem) {
        bool key_exist = (rem == lane_pair_.getR());
        int key_lane = __ffs(tile_.ballot(key_exist));
        if (key_lane == 0) return false;
        return tile_.shfl(true, key_lane - 1);
    }

    // Perform an exchange operation
    GPUHEADER_D
    ClearyCuckooEntry<addtype, remtype> exch_at_location(ClearyCuckooEntry<addtype, remtype> pair, const int loc) {
        ClearyCuckooEntry<addtype, remtype> old_pair;
        printf("%i: \t\t\t\tExch in bucket: thread_rank %i loc:%i\n", getThreadID(), tile_.thread_rank(), loc);
        if (tile_.thread_rank() == loc) {
            printf("%i: \t\t\t\tActual Exch\n", getThreadID());
            ptr_[tIndex].exchValue(&pair);
        }
        printf("%i: \t\t\t\t Exch Done\n", getThreadID());
        return tile_.shfl(pair, loc);
    }

    private:
        ClearyCuckooEntry<addtype, remtype>* ptr_;
        ClearyCuckooEntry<addtype, remtype> lane_pair_;
        const cg::thread_block_tile<tile_sz>  tile_;
        int tIndex = 0;
};

template <int tile_sz>
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
        const int RS;                         //RemainderSize
        const int AS;                         //AdressSize
        const int B;                          //NumBuckets
        const int Bs = tile_sz;               //BucketSize
        
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
        bool insertIntoTable(keytype k, ClearyCuckooEntry<addtype, remtype>* T, int* hs, cg::thread_block_tile<tile_sz> tile, int depth=0){
            printf("%i: \t\tInsert into Table\n", getThreadID());
            keytype x = k;
            int hash = hs[0];

            //If the key is already inserted don't do anything
            printf("%i: \t\t\tLookup\n", getThreadID());
            if (lookup(k, T, tile)) {
                return false;
            }

            //Start the iteration
            int c = 0;

            printf("%i: \t\t\tEntering Loop\n", getThreadID());
            while (c < MAXLOOPS) {
                //Get the add/rem of k
                hashtype hashed1 = RHASH(hash, x);
                addtype add = getAdd(hashed1, AS);
                remtype rem = getRem(hashed1, AS);

                auto cur_bucket = bucket<tile_sz>(T, tile, add, Bs);
                auto load = cur_bucket.compute_load();

                printf("%i: \t\t\tLoad - %i\n", getThreadID(), load);

                addtype bAdd;

                if (load == Bs) {
                    bAdd = RHASH(0, add) % Bs; //select some location within the table
                }
                else {
                    bAdd = load;
                }

                ClearyCuckooEntry<addtype, remtype> entry(rem, hash, true, false);
                entry = cur_bucket.exch_at_location(entry, bAdd);

                
                //Store the old value
                remtype temp = entry.getR();
                bool wasoccupied = entry.getO();
                int oldhash = entry.getH();

                printf("%i: \t\t\told: rem:%" PRIu64 " Occ:%i hash:%i \n", getThreadID(), temp, wasoccupied, oldhash);

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
        bool lookup(uint64_cu k, ClearyCuckooEntry<addtype, remtype>* T, cg::thread_block_tile<tile_sz>& tile){
            for (int i = 0; i < hn; i++) {
                uint64_cu hashed1 = RHASH(hashlist[i], k);
                addtype add = getAdd(hashed1, AS);
                remtype rem = getRem(hashed1, AS);
                
                auto cur_bucket = bucket<tile_sz>(T, tile, add , Bs);
                auto res = cur_bucket.find(rem);

                return res;
            }
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
        ClearyCuckooBucketed() : ClearyCuckooBucketed(4,1,1){}

        ClearyCuckooBucketed(int addressSize, int hashNumber) : 
            AS(addressSize), RS(HS - AS), B( (int)pow(2, AS) ){
            //printf("Constructor\n");
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

        //Taken from Better GPU Hash Tables
        GPUHEADER_D
        bool coopInsert(bool to_insert, keytype k) {
            printf("%i: \tcoopInsert %" PRIu64"\n", getThreadID(), k);
            cg::thread_block thb = cg::this_thread_block();
            auto tile = cg::tiled_partition<tile_sz>(thb);
            printf("%i: \tTiledPartition\n", getThreadID());
            auto thread_rank = tile.thread_rank();
            bool success = true;
            //Perform the insertions
            uint32_t work_queue;
            while (work_queue = tile.ballot(to_insert)) {
                printf("%i: \tThread Starting Insertion\n", getThreadID());
                auto cur_lane = __ffs(work_queue) - 1;
                auto cur_k = tile.shfl(k, cur_lane);
                auto cur_result = insertIntoTable(cur_k, T, hashlist, tile);
                printf("%i: \tthread_rank:%i  cur_lane:%i\n", getThreadID(), tile.thread_rank(), cur_lane);
                if (tile.thread_rank() == cur_lane) {
                    to_insert = false;
                    success = cur_result;
                }
                printf("%i: \tInsertion Done\n", getThreadID());
                if (tile.thread_rank() == 0) {
                    print();
                }
            }
            return success;
        }

        //Public insertion call
        GPUHEADER_D
#ifdef GPUCODE
            bool insert(uint64_cu k) {
#else
            bool insert(uint64_cu k, SpinBarrier * barrier) {
#endif
            printf("%i:Insert %" PRIu64 "\n", getThreadID(), k);

            //Stores success/failure of rehash
            bool finalRes = false;
            if (coopInsert(true, k)) {
                //Reset the Hash Counter

                finalRes = true;
            }

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

        //Public Lookup call
        GPUHEADER_D
        bool coopLookup(bool to_lookup, uint64_cu k){
            printf("%i: Coop Lookup\n", getThreadID());
            //Iterate over hash functions and check if found
            cg::thread_block thb = cg::this_thread_block();
            cg::thread_block_tile<tile_sz> tile = cg::tiled_partition<tile_sz>(thb);
            auto thread_rank = tile.thread_rank();
            bool success = true;
            //Perform the insertions

            uint32_t work_queue;
            while (work_queue = tile.ballot(to_lookup)) {
                auto cur_lane = __ffs(work_queue) - 1;
                auto cur_k = tile.shfl(k, cur_lane);
                auto cur_result = lookup(cur_k, T, tile);
                
                if (tile.thread_rank() == cur_lane) {
                    to_lookup = false;
                    success = cur_result;
                }
            }
            return success;
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

        int getBucketSize() {
            return Bs;
        }

};

//Method to fill ClearyCuckooBucketedtable
template <int tile_sz>
GPUHEADER_G
#ifdef GPUCODE
void fillClearyCuckooBucketed(int N, uint64_cu* vals, ClearyCuckooBucketed<tile_sz>* H, int* failFlag=nullptr, addtype begin = 0, int id = 0, int s = 1)
#else
void fillClearyCuckooBucketed(int N, uint64_cu* vals, ClearyCuckooBucketed<tile_sz>* H, SpinBarrier* barrier, int* failFlag = nullptr, addtype begin = 0, int id = 0, int s = 1)
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
        if (!(H->insert(vals[i]))) {
#else
        if (!(H->insert(vals[i]))) {
#endif
            if (failFlag != nullptr) {
                (*failFlag) = true;
            }
            break;
        }

        //H->print();
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
template <int tile_sz>
GPUHEADER_G
void fillClearyCuckooBucketed(int N, uint64_cu* vals, ClearyCuckooBucketed<tile_sz> * H, addtype* occupancy, int* failFlag, int id = 0, int s = 1)
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
        if (!(H->insert(vals[i]))) {
#else
        if (!(H->insert(vals[i]))) {
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
void fillClearyCuckooBucketed(int N, uint64_cu* vals, ClearyCuckooBucketed<tile_sz> * H, SpinBarrier* barrier, std::atomic<addtype>* occupancy, std::atomic<bool>* failFlag, int id = 0, int s = 1)
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
        if (!(H->insert(vals[i]))) {
#else
        if (!(H->insert(vals[i]))) {
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
template <int tile_sz>
GPUHEADER_G
void checkClearyCuckooBucketed(int N, uint64_cu* vals, ClearyCuckooBucketed<tile_sz>* H, bool* res, int id = 0, int s = 1)
{
#ifdef GPUCODE
    int index = threadIdx.x;
    int stride = blockDim.x;
#else
    int index = id;
    int stride = s;
#endif

    for (int i = index; i < N; i += stride) {
        if ( !(H->coopLookup(true, vals[i])) ) {
            res[0] = false;
        }
    }
}

//Method to do lookups in a ClearyCuckooBucketed table on an array of values
template <int tile_sz>
GPUHEADER_G
void lookupClearyCuckooBucketed(int N, int start, int end, uint64_cu* vals, ClearyCuckooBucketed<tile_sz>* H, int id = 0, int s = 1) {
#ifdef GPUCODE
    int index = threadIdx.x;
    int stride = blockDim.x;
#else
    int index = id;
    int stride = s;
#endif

    for (int i = index; i < N; i += stride) {
        H->coopLookup(true, vals[(i + start) % end]);
    }
}


