#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <math.h>
#include <iterator>
#include <set>

#ifndef HASHTABLE
#define HASHTABLE
#include "HashTable.h"
#endif

#ifndef HASHINCLUDED
#define HASHINCLUDED
#include "Hash.cpp"
#endif

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
        remtype *t;
        bool *O;
        //Backup Vars
        remtype *t_backup;
        bool *O_backup;
        int h1_backup, h2_backup;
        std::set<addtype>* delta = new std::set<addtype>();

        int hashcounter = 0;

        //Hash function ID
        int h1, h2;

        keyTuple splitKey(keytype key, int hash){
            hashtype mask = ((hashtype) 1 << AS) - 1;
            addtype add = key & mask;
            remtype rem = key >> AS ;
            return std::make_pair(add,labelKey(rem, hash));
        }

        uint64_t reformKey(keyTuple split){
            remtype rem = split.second;
            hashtype reform = unlabelKey(rem);
            reform = reform << AS;
            reform += split.first;
            return reform;
        }
        
        /**
         * Function to label which hash function was used on this value
         **/
        remtype labelKey(remtype rem, int h){
            remtype hnew = h;
            hnew = hnew << ((HS-AS));
            return hnew + rem;
        }

        remtype unlabelKey(remtype rem){
            remtype mask = ((remtype)1 << ((HS-AS))) - 1 ;
            rem = rem & mask;
            return rem;
        }

        int getLabel(remtype rem){
            return rem >> ((HS-AS));
        }

        bool insertIntoTable(keytype k, remtype *rems, bool *occs, int hash1, int hash2, int depth=0){
            keytype x = k;
            int hash = hash1;

            //If the key is already inserted don't do anything
            if (lookup(k, rems, occs)) {
                return false;
            }

            //Start the iteration
            int c = 0;

            while(c < MAXLOOPS){
                //Get the key of k
                hashtype hashed1 = RHASH(hash, x);
                keyTuple split1 = splitKey(hashed1, hash);

                //Store the old value
                remtype temp = rems[split1.first];
                bool wasoccupied = occs[split1.first];

                //Place new value
                rems[split1.first] = split1.second;
                occs[split1.first] = true;
                delta->insert(split1.first);

                //If the first spot was open return
                if(!wasoccupied){
                    return true;
                }

                //Rebuild the original key
                int oldhash = getLabel(temp);
                hashtype h_old = reformKey(std::make_pair(split1.first, temp));
                keytype k_old = RHASH_INVERSE(oldhash, h_old);

                //Hash with the opposite hash value
                if(oldhash == hash1){
                    hash = hash2;
                }else{
                    hash = hash1;
                }
                
                //Do the same procedure again
                hashtype hashed2 = RHASH(hash, k_old);
                keyTuple split2 = splitKey(hashed2, hash);
                delta->insert(split2.first);

                temp = rems[split2.first];
                wasoccupied = occs[split2.first];

                rems[split2.first] = split2.second;
                occs[split2.first] = true;
                //If the second spot was open return
                if(!wasoccupied){
                    return true;
                }

                oldhash = getLabel(temp);
                //Hash with the opposite hash value
                if(oldhash == hash1){
                    hash = hash2;
                }else{
                    hash = hash1;
                }
                h_old = reformKey(std::make_pair(split2.first, temp));
                x = RHASH_INVERSE(oldhash, h_old);
                c++;
            }
            if(depth>0){return false;}
            //If MAXLOOPS is reached rehash the whole table
            if(!rehash()){
                //If rehash fails, return
                return false;
            }
            if(insertIntoTable(x, rems, occs, h1, h2, depth)){return true;}
            return false;
        };

        bool rehash(int depth){
            //Prevent recursion of rehashing
            if(depth >0){return false;}

            int hash1 = rand() % 32;
            int hash2 = rand() % 32;

            //Store the table in a temporary array
            remtype *t_new = new remtype[tablesize];
            bool *o_new = new bool[tablesize];
            for(int i=0; i<tablesize; i++){
                t_new[i] = 0;
                o_new[i] = 0;
            }

            //Insert the old values in the new table
            for(int i=0; i<tablesize; i++){
                if(O[i]){
                    remtype rem = t[i];
                    int label = getLabel(rem);
                    hashtype h = reformKey(std::make_pair(i, rem));
                    keytype k = RHASH_INVERSE(label, h);
                    //If Insertion Fails (New Rehash) Return
                    if(!insertIntoTable(k,t_new,o_new,hash1,hash2,depth+1)){
                        delete [] t_new;
                        delete [] o_new;
                        return false;
                    }
                }
            }

            //If all vals are inserted, copy new into global and delete
            h1 = hash1;
            h2 = hash2;

            for(int i=0; i<tablesize; i++){
                t[i] = t_new[i];
                O[i] = o_new[i];
            }
            
            delete[] t_new;
            delete[] o_new;

            return true;
        };

        bool lookup(uint64_t k, remtype* rems, bool* occs){
            uint64_t hashed1 = RHASH(h1, k);
            keyTuple split1 = splitKey(hashed1, h1);
            if( rems[split1.first] == split1.second && occs[split1.first] ){
                return true;
            }

            uint64_t hashed2 = RHASH(h2, k);
            keyTuple split2 = splitKey(hashed2, h2);
            if( rems[split2.first] == split2.second && occs[split2.first]){
                return true;
            }

            return false;
        };

        void revertTable(){
            for(int i: *delta){
                t[i] = t_backup[i];
                O[i] = O_backup[i];
            }
            h1 = h1_backup;
            h2 = h2_backup;
            delta->clear();
        }

        void updateBackup(){
            for(int i: *delta){
                t_backup[i] = t[i];
                O_backup[i] = O[i];
            }
            h1_backup = h1;
            h2_backup = h2;
            delta->clear();
        }

    
    public:
        /**
         * Constructor
         */
        ClearyCuckoo(int adressSize){
            AS = adressSize;
            RS = HS-AS;
            tablesize = (int) pow(2,AS);

            t = new remtype[tablesize];
            O = new bool[tablesize];

            t_backup = new remtype[tablesize];
            O_backup = new bool[tablesize];

            for(int i=0; i<tablesize; i++){
                t[i] = 0;
                O[i] = false;

                t_backup[i] = 0;
                O_backup[i] = false;
            }
            h1 = 1;
            h2 = 2;

            h1_backup = h1;
            h2_backup = h2;
        }

        /**
         * Destructor
         */
        ~ClearyCuckoo(){
            delete [] t;
            delete [] O;

            delete[] t_backup;
            delete[] O_backup;
            delete delta;
        }

        bool insert(keytype k){
            //Succesful Insertion
            if(insertIntoTable(k,t,O,h1,h2)){
                //Reset the Hash Counter
                hashcounter = 0;
                //Update the Backup Table
                updateBackup();
                return true;
            }
            //On Fail Revert the Table
            revertTable();
            return false;
        };

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

        bool lookup(uint64_t k){
            return lookup(k, t, O);
        };

        void clear(){
            for(int i=0; i<tablesize; i++){
                t[i] = 0;
                O[i] = false;
            }
        }

        int getSize(){
            return tablesize;
        }

        void print(remtype* rems, bool* occs){
            const char separator = ' ';
            std::cout << "-----------------------------------\n";
            std::cout << "|" << std::setw(6) << std::setfill(separator) << "i" << "|";
            std::cout << std::setw(20)<< std::setfill(separator) << "r" << "|";
            std::cout << std::setw(5)<< std::setfill(separator) << "O[i]" << "|";
            std::cout << std::setw(20)<< std::setfill(separator) << "key" << "|";
            std::cout << std::setw(5)<< std::setfill(separator) << "label" << "|\n";
            for(int i=0; i<tablesize; i++){
                if(occs[i]){
                    remtype rem = t[i];
                    int label = getLabel(rem);
                    hashtype h = reformKey(std::make_pair(i, rem));
                    keytype k = RHASH_INVERSE(label, h);

                    std::cout << "|" << std::setw(6) << std::setfill(separator) << i << "|";
                    std::cout << std::setw(20)<< std::setfill(separator) << rems[i] << "|";
                    std::cout << std::setw(5)<< std::setfill(separator) << O[i] << "|";
                    std::cout << std::setw(20)<< std::setfill(separator) << k << "|";
                    std::cout << std::setw(5)<< std::setfill(separator) << label << "|\n";
                }
            }
            std::cout << "-----------------------------------\n";
        }

        void print(){
            print(t,O);
        }

        void debug(){}
        
        void setMaxRehashes(int x){
            MAXREHASHES = x;
        }

        void setMaxLoops(int x){
            MAXLOOPS = x;
        }

};