#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <math.h>
#include <iterator>
#include <set>
#include <inttypes.h>
#include <random>

#include <curand.h>
#include <curand_kernel.h>

#include "int_cu.h"


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

        __host__ __device__
        addtype getAdd(keytype key){
            hashtype mask = ((hashtype) 1 << AS) - 1;
            addtype add = key & mask;
            return add;
        }

        __host__ __device__
            remtype getRem(keytype key) {
            remtype rem = key >> AS;
            return rem;
        }

        __host__ __device__
        uint64_cu reformKey(addtype add, remtype rem){
            rem = rem << AS;
            rem += add;
            return rem;
        }

        __host__ __device__
        void createHashList(int* list) {
            //printf("\tCreating Hashlist\n");
            for (int i = 0; i < hn; i++) {
                list[i] = i;
            }
            return;
        }

        __host__ __device__
            void iterateHashList(int* list) {
            //printf("\tUpdating Hashlist\n");
            for (int i = 0; i < hn; i++) {
                list[i] = (list[i]+1)%32;
            }
            return;
        }

        __host__ __device__
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

        __host__ __device__
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
        __host__ __device__
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
                addtype add = getAdd(hashed1);
                remtype rem = getRem(hashed1);

                //Place new value
                //printf("\tPlacing New Value\n");
                ClearyCuckooEntry<addtype, remtype> entry = ClearyCuckooEntry<addtype, remtype>(rem, hash, true, false);
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
                hashtype h_old = reformKey(add, temp);
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

        __host__ __device__
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
                    hashtype h_old = reformKey(i, temp);
                    keytype k_old = RHASH_INVERSE(oldhash, h_old);
                    insertIntoTable(k_old, T, depth+1);
                }
            }
            //printf("\tRehash Done\n");
            return true;
        };

        __host__ __device__
        bool lookup(uint64_cu k, ClearyCuckooEntry<addtype, remtype>* T){
            //printf("\t\tLookup %" PRIu64 "\n", k);
            for (int i = 0; i < hn; i++) {
                uint64_cu hashed1 = RHASH(hashlist[i], k);
                addtype add = getAdd(hashed1);
                remtype rem = getRem(hashed1);
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

            //printf("\tAllocating Memory\n");
            gpuErrchk(cudaMallocManaged(&T, tablesize * sizeof(ClearyCuckooEntry<addtype,remtype>)));
            gpuErrchk(cudaMallocManaged(&hashlist, hn * sizeof(int)));

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

            gpuErrchk(cudaFree(T));
            gpuErrchk(cudaFree(hashlist));
        }

        __device__ __host__
        bool insert(uint64_cu k){
            //Succesful Insertion
            //printf("\tInserting val %" PRIu64 "\n", k);
            if(insertIntoTable(k,T,0)){
                //Reset the Hash Counter
                hashcounter = 0;
                return true;
            }
            return false;
        };

        __host__ __device__
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

        __host__ __device__
        bool lookup(uint64_cu k){
            return lookup(k, T);
        };

        __host__ __device__
        void clear(){
            for(int i=0; i<tablesize; i++){
                T[i] = ClearyCuckooEntry<addtype, remtype>();
            }
        }

        __host__ __device__
        int getSize(){
            return tablesize;
        }

        __host__ __device__
        void print(){
            printf("----------------------------------------------------------------\n");
            printf("|    i     |     R[i]       | O[i] |        key         |label |\n");
            printf("----------------------------------------------------------------\n");
            printf("Tablesize %i\n", tablesize);
            for(int i=0; i<tablesize; i++){
                if(T[i].getO()){
                    remtype rem = T[i].getR();
                    int label = T[i].getH();
                    hashtype h = reformKey(i, rem);
                    keytype k = RHASH_INVERSE(label, h);

                    printf("|%-10i|%-16" PRIu64 "|%-6i|%-20" PRIu64 "|%-6i|\n", i, T[i].getR(), T[i].getO(), k, T[i].getH());
                }
            }
            printf("------------------------------------------------------------\n");
        }

        __host__ __device__
        void debug(uint64_cu i) {

        }

        void setMaxRehashes(int x){
            MAXREHASHES = x;
        }

        void setMaxLoops(int x){
            MAXLOOPS = x;
        }

};
