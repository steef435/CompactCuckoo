#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <math.h>
#include <iterator>
#include <set>
#include <inttypes.h>

#ifndef HASHTABLE
#define HASHTABLE
#include "HashTable.h"
#endif

#ifndef HASHINCLUDED
#define HASHINCLUDED
#include "hashfunctions.cu"
#endif

#include "ClearyCuckooEntry.cu"

class ClearyCuckoo : public HashTable{
    //Allows for easy changing of the types
    using addtype = uint32_t;
    using remtype = uint64_t;
    using hashtype = uint64_t;
    using keytype = uint64_t;
    typedef std::pair<addtype, remtype> keyTuple; 

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

        __device__
        keyTuple splitKey(keytype key){
            hashtype mask = ((hashtype) 1 << AS) - 1;
            addtype add = key & mask;
            remtype rem = key >> AS ;
            return std::make_pair(add,rem);
        }

        __device__
        uint64_t reformKey(addtype add, remtype rem){
            rem = rem << AS;
            rem += add;
            return rem;
        }

        __host__ __device__
        int* createHashList(int n) {
            int* list = new int[n];

            for (int i = 0; i < n; i++) {
                int newhash = rand() % 32;
                
                bool alreadyexists = false;
                for (int j = 0; j < i; j++) {
                    if (list[j] == newhash) { alreadyexists = true; }
                }

                if (alreadyexists) {
                    i--;
                }
                else {
                    list[i] = newhash;
                }
            }

            return list;
        }

        __device__
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

        __device__
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
        __device__
        bool insertIntoTable(keytype k, ClearyCuckooEntry<addtype, remtype>* T, int depth=0){
            keytype x = k;
            int hash = hashlist[0];

            //If the key is already inserted don't do anything
            if (lookup(k, T)) {
                return false;
            }

            //Start the iteration
            int c = 0;

            while(c < MAXLOOPS){
                //Get the key of k
                hashtype hashed1 = RHASH(hash, x);
                keyTuple split1 = splitKey(hashed1);

                //Store the old value
                remtype temp = T[split1.first].getR();
                bool wasoccupied = T[split1.first].getO();
                int oldhash = T[split1.first].getH();

                //Place new value
                //TODO use atomicCAS
                T[split1.first].setR(split1.second);
                T[split1.first].setO(true);
                T[split1.first].setH(hash);

                //If the first spot was open return
                if(!wasoccupied){
                    return true;
                }

                //Otherwise rebuild the original key
                hashtype h_old = reformKey(split1.first, temp);
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

        __device__
        bool rehash(int depth){
            //Prevent recursion of rehashing
            if(depth >0){return false;}

            hashlist = createHashList(hn);

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

        __device__
        bool lookup(uint64_t k, ClearyCuckooEntry<addtype, remtype>* T){
            for (int i = 0; i < 32; i++) {
                uint64_t hashed1 = RHASH(hashlist[i], k);
                keyTuple split1 = splitKey(hashed1);
                if (T[split1.first].getR() == split1.second && T[split1.first].getO()) {
                    return true;
                }
            }
            return false;
        };

    
    public:
        /**
         * Constructor
         */
        ClearyCuckoo(int adressSize, int hashNumber){
            AS = adressSize;
            RS = HS-AS;
            tablesize = (int) pow(2,AS);

            T = new ClearyCuckooEntry<addtype, remtype>[tablesize];

            for(int i=0; i<tablesize; i++){
                T[i] = ClearyCuckooEntry<addtype, remtype>();
            }
            
            hn = hashNumber;
            int* hashlist = createHashList(hn);

        }

        /**
         * Destructor
         */
        ~ClearyCuckoo(){
            delete[] T;

            delete[] hashlist;
        }

        __device__
        bool ClearyCuckoo::insert(keytype k){
            //Succesful Insertion
            if(insertIntoTable(k,T)){
                //Reset the Hash Counter
                hashcounter = 0;
                return true;
            }
            return false;
        };

        __device__
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
        
        __device__
        bool ClearyCuckoo::lookup(uint64_t k){
            return lookup(k, T);
        };

        __device__
        void ClearyCuckoo::clear(){
            for(int i=0; i<tablesize; i++){
                T[i] = ClearyCuckooEntry<addtype, remtype>();
            }
        }

        __device__
        int ClearyCuckoo::getSize(){
            return tablesize;
        }

        __device__
        void ClearyCuckoo::print(ClearyCuckooEntry<addtype, remtype>* T){
            printf("-----------------------------------\n");
            printf("|i|r|O[i]|key|label|\n");
            for(int i=0; i<tablesize; i++){
                if(T[i].getO()){
                    remtype rem = T[i].getR();
                    int label = T[i].getH();
                    hashtype h = reformKey(i, rem);
                    keytype k = RHASH_INVERSE(label, h);

                    printf("|%-3i|%-10" PRIu64 "|%-3i|%-10" PRIu64 "|%-4i|\n", i, T[i].getR(), T[i].getO(), k, T[i].getH());
                }
            }
            printf("-----------------------------------\n");
        }

        __device__
        void ClearyCuckoo::print(){
            print(T);
        }

        __device__
        void ClearyCuckoo::debug(){}
        
        void setMaxRehashes(int x){
            MAXREHASHES = x;
        }

        void setMaxLoops(int x){
            MAXLOOPS = x;
        }

};