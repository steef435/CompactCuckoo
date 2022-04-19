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

#ifndef HASHTABLE
#define HASHTABLE
#include "HashTable.h"
#endif

#ifndef HASHINCLUDED
#define HASHINCLUDED
#include "hashfunctions.cu"
#endif

#include "ClearyCuckooEntry.cu"

class ClearyCuckoo{

/*
*
*  Global Variables
*
*/

    //Allows for easy changing of the types
    using addtype = uint32_t;
    using remtype = uint64_t;
    using hashtype = uint64_t;
    using keytype = uint64_t;

    private:
        //Constant Vars
        const static int HS = 59;       //HashSize
        int MAXLOOPS = 25;
        int MAXREHASHES = 30;

        //Vars at Construction
        int AS;                    //AdressSize
        int RS;                         //RemainderSize
        int tablesize;
        
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
        uint64_t reformKey(addtype add, remtype rem){
            rem = rem << AS;
            rem += add;
            return rem;
        }

        __host__ __device__
        void createHashList(int* list) {
            printf("\tCreating Hashlist\n");
            for (int i = 0; i < hn; i++) {
                list[i] = i;
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
            printf("\tInsertintoTable\n");
            keytype x = k;
            int hash = hashlist[0];

            //If the key is already inserted don't do anything
            if (lookup(k, T)) {
                printf("\tAlready Exists\n");
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
                printf("\tPlacing New Value\n");
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
                keytype k_old = RHASH_INVERSE(oldhash, h_old);

                //Hash with the opposite hash value
                hash = getNextHash(hashlist, oldhash);
                
                c++;
            }

            if(depth>0){return false;}
            //If MAXLOOPS is reached rehash the whole table
            if(!rehash()){
                //If rehash fails, return
                return false;
            }

            if(insertIntoTable(x, T, depth)){return true;}

            return false;
        };

        __host__ __device__
        bool rehash(int depth){
            //Prevent recursion of rehashing
            if(depth >0){return false;}

            createHashList(hashlist);

            //Insert the old values in the new table
            for(int i=0; i<tablesize; i++){

                if (!containsHash(hashlist, T[i].getH())) {
                    //Store the old value
                    remtype temp = T[i].getR();
                    int oldhash = T[i].getH();

                    //Delete Entry
                    T[i] = ClearyCuckooEntry<addtype, remtype>();

                    //Insert
                    hashtype h_old = reformKey(i, temp);
                    keytype k_old = RHASH_INVERSE(oldhash, h_old);
                    insertIntoTable(k_old, T, depth);
                }
            }            
            return true;
        };

        __host__ __device__
        bool lookup(uint64_t k, ClearyCuckooEntry<addtype, remtype>* T){
            for (int i = 0; i < hn; i++) {
                uint64_t hashed1 = RHASH(hashlist[i], k);
                addtype add = getAdd(hashed1);
                remtype rem = getRem(hashed1);
                if (T[add].getR() == rem && T[add].getO()) {
                    return true;
                }
            }
            return false;
        };

    
    public:
        /**
         * Constructor
         */
        ClearyCuckoo() {}

        ClearyCuckoo(int adressSize, int hashNumber){
            printf("Creating ClearyCuckoo Table\n");
            AS = adressSize;
            RS = HS-AS;
            tablesize = (int) pow(2,AS);

            hn = hashNumber;

            printf("\tAllocating Memory\n");
            cudaMallocManaged(&T, tablesize * sizeof(ClearyCuckooEntry<addtype,remtype>));
            cudaMallocManaged(&hashlist, hn * sizeof(int));

            printf("\tInitializing Entries\n");
            for(int i=0; i<tablesize; i++){
                new (&T[i]) ClearyCuckooEntry<addtype, remtype>();
            }
            
            createHashList(hashlist);
            printf("\tDone\n");

        }

        /**
         * Destructor
         */
        ~ClearyCuckoo(){
            printf("Destroying Table\n");

            cudaFree(T);
            cudaFree(hashlist);
        }

        __host__ __device__
        bool ClearyCuckoo::insert(uint64_t k){
            //Succesful Insertion
            printf("\tInserting %" PRIu64 "\n", k);
            if(insertIntoTable(k,T,0)){
                //Reset the Hash Counter
                hashcounter = 0;
                return true;
            }
            return false;
        };

        __host__ __device__
        bool ClearyCuckoo::rehash(){
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
        bool ClearyCuckoo::lookup(uint64_t k){
            return lookup(k, T);
        };

        __host__ __device__
        void ClearyCuckoo::clear(){
            for(int i=0; i<tablesize; i++){
                T[i] = ClearyCuckooEntry<addtype, remtype>();
            }
        }

        __host__ __device__
        int ClearyCuckoo::getSize(){
            return tablesize;
        }

        __host__ __device__
        void ClearyCuckoo::print(ClearyCuckooEntry<addtype, remtype>* T){
            printf("------------------------------------------------------------\n");
            printf("|    i     |     R[i]       | O[i] |      key       |label |\n");
            printf("------------------------------------------------------------\n");
            for(int i=0; i<tablesize; i++){
                if(T[i].getO()){
                    remtype rem = T[i].getR();
                    int label = T[i].getH();
                    hashtype h = reformKey(i, rem);
                    keytype k = RHASH_INVERSE(label, h);

                    printf("|%-10i|%-16" PRIu64 "|%-6i|%-16" PRIu64 "|%-6i|\n", i, T[i].getR(), T[i].getO(), k, T[i].getH());
                }
            }
            printf("------------------------------------------------------------\n");
        }

        __host__ __device__
        void ClearyCuckoo::print(){
            print(T);
        }

        __host__ __device__
        void debug(uint64_t i) {
            //printf("%" PRIu64, T[0]);
            printf("%" PRIu64, i);
        }
        
        void setMaxRehashes(int x){
            MAXREHASHES = x;
        }

        void setMaxLoops(int x){
            MAXLOOPS = x;
        }

};