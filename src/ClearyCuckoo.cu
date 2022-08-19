#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <math.h>
#include <iterator>
#include <set>
#include <inttypes.h>
#include <atomic>
#include <random>

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
        std::atomic<bool> failFlag;
        std::atomic<int> occupation;
        std::atomic<int> rehashFlag;
#endif


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

        /**
         * Function to label which hash function was used on this value
         **/
        GPUHEADER
        bool insertIntoTable(keytype k, ClearyCuckooEntry<addtype, remtype>* T, int* hs, int depth=0){
            //printf("\tInsertintoTable\n");
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
                //Get the key of k
                hashtype hashed1 = RHASH(hash, x);
                addtype add = getAdd(hashed1, AS);
                remtype rem = getRem(hashed1, AS);

                //Place new value
                //printf("\tPlacing New Value\n");
                ClearyCuckooEntry<addtype, remtype> entry(rem, hash, true, false);
                T[add].exchValue(&entry);

                //Store the old value
                remtype temp = entry.getR();
                bool wasoccupied = entry.getO();
                int oldhash = entry.getH();


                //If the first spot was open return
                if(!wasoccupied){
                    return true;
                }

                //Otherwise rebuild the original key
                hashtype h_old = reformKey(add, temp, AS);
                x = RHASH_INVERSE(oldhash, h_old);

                //Hash with the next hash value
                hash = getNextHash(hs, oldhash);

                c++;
            }
            
            if(depth>0){return false;}
            //If MAXLOOPS is reached rehash the whole table
            //printf("Rehash\n");
            if(!rehash()){
                //If rehash fails, return
                //printf("Rehash Failed\n");
                return false;
            }
            //printf("Next insertion\n");
            if(insertIntoTable(x, T, hs, depth)){return true;}

            return false;
        };

        GPUHEADER
        bool rehash(int depth, int* hs){
            printf("\t\tRehash call\n");
            for (int i = 0; i < hn; i++) {
                printf("%i, ", hs[i]);
            }
            printf("\n");
            //Prevent recursion of rehashing
            if(depth >0){return false;}

            ClearyCuckooEntry<addtype, remtype>* T_copy;

#ifdef GPUCODE
            gpuErrchk(cudaMallocManaged(&T_copy, tablesize * sizeof(ClearyCuckooEntry<addtype, remtype>)));
#else
            T_copy = new ClearyCuckooEntry<addtype, remtype>[tablesize];
#endif

            //Initialize the table copy
            for (int i = 0; i < tablesize; i++) {
                new (&T_copy[i]) ClearyCuckooEntry<addtype, remtype>();
            }

            //Copy entries into new table
            for (int i = 0; i < tablesize; i++) {
                if (T[i].getO()) {
                    hashtype h_old = reformKey(i, T[i].getR(), AS);
                    hashtype k = RHASH_INVERSE(T[i].getH(), h_old);

                    if (!insertIntoTable(k, T_copy, hs, depth + 1)) {
                        return false;
                    }
                    print(T_copy);
                }
            }

            //Copy the table into the table
            for (int i = 0; i < tablesize; i++) {
                T[i].setValue(T_copy[i].getValue());
            }

            //set the new hashlist
            for (int i = 0; i < hn; i++) {
                hashlist[i] = hs[i];
            }
            
#ifdef GPUCODE
            gpuErrchk(cudaFree(T_copy));
            gpuErrchk(cudaFree(hs));
#else
            delete[] T_copy;
            delete[] hs;
#endif

            //printf("\tRehash Done\n");
            return true;
        };

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
            printf("----------------------------------------------------------------\n");
            printf("|    i     |     R[i]       | O[i] |        key         |label |\n");
            printf("----------------------------------------------------------------\n");
            printf("Tablesize %i\n", tablesize);
            for (int i = 0; i < tablesize; i++) {
                if (T[i].getO()) {
                    remtype rem = T[i].getR();
                    int label = T[i].getH();
                    hashtype h = reformKey(i, rem, AS);
                    keytype k = RHASH_INVERSE(label, h);

                    printf("|%-10i|%-16" PRIu64 "|%-6i|%-20" PRIu64 "|%-6i|\n", i, T[i].getR(), T[i].getO(), k, T[i].getH());
                }
            }
            printf("------------------------------------------------------------\n");
        }


    public:
        /**
         * Constructor
         */
        ClearyCuckoo() {}

        ClearyCuckoo(int adressSize, int hashNumber){
            //printf("Creating ClearyCuckoo Table\n");
            AS = adressSize;
            RS = HS-AS;
            tablesize = (int) pow(2,AS);

            hn = hashNumber;
            /*
#ifdef GPUCODE
            failFlag = false;
            occupation = 0;
#else
            failFlag.store(false);
            occupation.store(0);
#endif
            */
            //printf("\tAllocating Memory\n");
            #ifdef GPUCODE
            gpuErrchk(cudaMallocManaged(&T, tablesize * sizeof(ClearyCuckooEntry<addtype,remtype>)));
            gpuErrchk(cudaMallocManaged(&hashlist, hn * sizeof(int)));
            #else
            T = new ClearyCuckooEntry<addtype, remtype>[tablesize];
            hashlist = new int[hn];
            #endif

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
            #else
            delete[] T;
            delete[] hashlist;
            #endif
        }

        GPUHEADER_D
        bool insert(uint64_cu k){
            //Succesful Insertion
            //printf("\tInserting val %" PRIu64 "\n", k);
#ifdef GPUCODE
            if (failFlag) {
                return false
            }
            while (rehashFlag) {
                if (failFlag) {
                    return false;
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

            if(insertIntoTable(k,T, hashlist,0)){
                //Reset the Hash Counter
                hashcounter = 0;
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
#ifdef GPUCODE
            atomicExch(&failFlag, 1);
#else
            failFlag.store(true);
#endif
            return false;
        };

        GPUHEADER
        bool rehash(){
#ifdef GPUCODE
            atomicExch(&rehashFlag, 1);
#else
            rehashFlag.store(1);
#endif

            //Create the new hashlist
            int* hashlist_new;
#ifdef GPUCODE
            gpuErrchk(cudaMallocManaged(&hashlist_new, hn * sizeof(int)));
#else
            hashlist_new = new int[hn];
#endif
            for (int i = 0; i < hn; i++) {
                hashlist_new[i] = hashlist[i];
            }

            iterateHashList(hashlist_new);

            //Local counter for number of rehashes
            while(!rehash(0, hashlist_new) && hashcounter<MAXREHASHES){
                //printf("\tRehash call %i\n", hashcounter);
                iterateHashList(hashlist_new);
                hashcounter++;
            };
            //If counter tripped return
            if(hashcounter >= MAXREHASHES){
                printf("\t +++++++++++++++++++++++++++++++++++++++Rehash FAIL++++++++++++++++++++++++++++++++++++++\n");
                return false;
            }
            //Rehash done
            printf("\t +++++++++++++++++++++++++++++++++++++++Rehash SUCCESS++++++++++++++++++++++++++++++++++++++\n");
            hashcounter=0;

#ifdef GPUCODE
            atomicExch(&rehashFlag, 0);
#else
            rehashFlag.store(0);
#endif

            return true;
        }

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
        int* getHashlist() {
            int* res = new int[3];
            for (int i = 0; i < hn; i++) {
                res[i] = hashlist[i];
            }
            return res;
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
                printf("Not all Zero\n");
            }
        }

        GPUHEADER
        void print(){
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

};

GPUHEADER_G
void fillClearyCuckoo(int N, uint64_cu* vals, ClearyCuckoo* H, addtype begin = 0, int id = 0, int s = 1)
{
#ifdef GPUCODE
    int index = threadIdx.x;
    int stride = blockDim.x;
#else
    int index = id;
    int stride = s;
#endif

    for (int i = index + begin; i < N + begin; i += stride) {
        if (!(H->insert(vals[i]))) {
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
            printf("\tSetting Res:Val %" PRIu64 " Missing\n", vals[i]);
            res[0] = false;
        }
    }
}