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

    //Allows for easy changing of the types
    using addtype = uint32_t;
    using remtype = uint64_cu;
    using hashtype = uint64_cu;
    using keytype = uint64_cu;

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
        bool failFlag = false;
        int occupation = 0;
#else
        std::atomic<bool> failFlag;
        std::atomic<int> occupation;
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
                list[i] = (list[i]+1)%32;
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
        bool insertIntoTable(keytype k, ClearyCuckooEntry<addtype, remtype>* T, int depth=0){
            //printf("\tInsertintoTable\n");
            keytype x = k;
            int hash = hashlist[0];

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
                hash = getNextHash(hashlist, oldhash);

                c++;
            }
            /*
            if(depth>0){return false;}
            //If MAXLOOPS is reached rehash the whole table
            if(!rehash()){
                //If rehash fails, return
                return false;
            }*/

            //if(insertIntoTable(x, T, depth)){return true;}

            return false;
        };

        GPUHEADER
        bool rehash(int depth){
            //printf("Rehash\n");
            //Prevent recursion of rehashing
            if(depth >0){return false;}

            //printf("\tCreate Hashlist\n");
            iterateHashList(hashlist);

            //Insert the old values in the new table
            //printf("\tCheck all values for correctness\n");
            for(int i=0; i<tablesize; i++){
                if ( ( !containsHash(hashlist, T[i].getH()) ) && T[i].getO()) {
                    //printf("\tVal no longer correct\n");
                    //Store the old value
                    remtype temp = T[i].getR();
                    int oldhash = T[i].getH();

                    //Delete Entry
                    new (&T[i]) ClearyCuckooEntry<addtype, remtype>();

                    //Insert
                    hashtype h_old = reformKey(i, temp, AS);
                    keytype k_old = RHASH_INVERSE(oldhash, h_old);
                    insertIntoTable(k_old, T, depth+1);
                }
            }
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

            failFlag.store(false);
            occupation.store(0);

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

        GPUHEADER
        bool insert(uint64_cu k){
            //Succesful Insertion
            //printf("\tInserting val %" PRIu64 "\n", k);
            if (failFlag.load() || occupation.load() == tablesize) {
                return false;
            }
            if(insertIntoTable(k,T,0)){
                //Reset the Hash Counter
                hashcounter = 0;
                //print();
                occupation += 1;
                return true;
            }
            failFlag.store(false);
            return false;
        };

        GPUHEADER
        bool rehash(){
            //Local counter for number of rehashes
            while(!rehash(0) && hashcounter<MAXREHASHES){
                hashcounter++;
            };
            //If counter tripped return
            if(hashcounter >= MAXREHASHES){
                return false;
            }
            hashcounter++;
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
            printf("----------------------------------------------------------------\n");
            printf("|    i     |     R[i]       | O[i] |        key         |label |\n");
            printf("----------------------------------------------------------------\n");
            printf("Tablesize %i\n", tablesize);
            for(int i=0; i<tablesize; i++){
                if(T[i].getO()){
                    remtype rem = T[i].getR();
                    int label = T[i].getH();
                    hashtype h = reformKey(i, rem, AS);
                    keytype k = RHASH_INVERSE(label, h);

                    printf("|%-10i|%-16" PRIu64 "|%-6i|%-20" PRIu64 "|%-6i|\n", i, T[i].getR(), T[i].getO(), k, T[i].getH());
                }
            }
            printf("------------------------------------------------------------\n");
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
    int index = getThreadID();
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
    int index = getThreadID();
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
void checkClearyCuckoo(int N, uint64_cu* vals, ClearyCuckoo* H, bool* res, int id = 0, int s = 1)
{
#ifdef GPUCODE
    int index = getThreadID();
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