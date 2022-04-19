#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <math.h>
#include <string> 
#include <math.h>

#include <bitset>
#include <inttypes.h>

#ifndef HASHTABLE
#define HASHTABLE
#include "HashTable.h"
#endif

#ifndef HASHINCLUDED
#define HASHINCLUDED
#include "hashfunctions.cu"
#endif

#include "ClearyEntry.cu"

//Types to allow for changes

using addtype = uint32_t;
using remtype = uint64_t;
using hashtype = uint64_t;
using keytype = uint64_t;

//Enum for searching

enum direction{up, down, here};


class Cleary{
    //Allows for easy changing of the types
    
    private:
        //Constant Vars
        const static int HS = 59;       //HashSize
        const static int BUFFER = 0; //Space assigned for overflow
        const static int MAXLOOPS = 24;
        //Vars assigned at construction
        int AS;                  //AdressSize
        int RS;                  //RemainderSize
        int size;                //Allocated Size of Table
        int tablesize;              //Actual size of table with buffer
        addtype MAX_ADRESS;
        addtype MIN_ADRESS = 0;

        //Tables
        ClearyEntry<addtype, remtype>* T;

        //Hash function ID
        int h1;

        __host__ __device__
            addtype getAdd(keytype key) {
            hashtype mask = ((hashtype)1 << AS) - 1;
            addtype add = key & mask;
            return add;
        }

        __host__ __device__
            remtype getRem(keytype key) {
            remtype rem = key >> AS;
            return rem;
        }

        __host__ __device__
            uint64_t reformKey(addtype add, remtype rem) {
            rem = rem << AS;
            rem += add;
            return rem;
        }

        __host__ __device__
        addtype findIndex(uint64_t k){
            hashtype h = RHASH(h1, k);
            addtype j = getAdd(h);
            remtype rem =  getRem(h);

            printf("\t\t findIndex:add:%" PRIu32 " rem:%" PRIu64 "\n", j, rem);

            addtype i = j;
            int cnt = 0;

            //Find first well defined A value
            while(T[i].getA() == 64 && i!=MIN_ADRESS){
                cnt = cnt - (T[i].getV() ? 1 : 0);
                i=i-1;
            };
            cnt = cnt + T[i].getA();

            //Look for the relevant group
            direction dir = up;
            if(cnt < 0){
                dir = up;
                while(cnt != 0 && i != MAX_ADRESS){
                    i = i+1;
                    cnt = cnt + (T[i].getC() ? 1 : 0);
                };
                if(T[i].getR() >= rem){
                    dir = here;
                }
            }else if(cnt > 0){
                dir = down;
                while(cnt != 0 && i != MIN_ADRESS){
                    cnt = cnt - (T[i].getC() ? 1 : 0);
                    i = i - 1;
                }
                if(T[i].getR() <= rem){dir = here;}
            }else{
                if(T[i].getR() > rem){dir = down;}
                else if(T[i].getR() < rem){dir = up;}
                else{dir = here;}
            }

            //Look inside of the group
            switch (dir)
            {
                case here:
                    break;

                case down:
                    while(dir != here){
                        if(T[i].getC() == 1 || i==MIN_ADRESS){dir = here;}
                        else{
                            i=i-1;
                            if(T[i].getR() <= rem){
                                dir = here;
                            }
                        }
                    };

                case up:
                    while(dir != here){
                        if(i == MAX_ADRESS){
                          dir = here;
                        }else if(T[i+1].getC() == 1){
                            dir = here;
                        }else{
                            i = i+1;
                            if(T[i].getR() >= rem){
                                dir = here;
                            }
                        }
                    }

                default:
                    break;
            };
            printf("\t\tfoundindex:%" PRIu32 "\n", i);
            return i;
        }


        addtype leftLock(addtype i) {
            if (i == MIN_ADDRESS) {
                return i;
            }
            while (T[i-1].getO() && i>MIN_ADDRESS) {
                i -= 1;
            }
            return i;
        }

        addtype rightLock(addtype i) {
            if (i == MAX_ADDRESS) {
                return i;
            }
            while (T[i+1].getO() && i<MAX_ADDRESS) {
                i += 1;
            }
            return i;
        }


        __host__ __device__
        insertIntoTable(keytype k) {
            printf("\tInserting Into Table %" PRIu64 "\n", k);

            hashtype h = RHASH(h1, k);
            addtype j = getAdd(h);
            remtype rem = getRem(h);

            bool newgroup = false;

            //Check virgin bit and set
            if (!T[j].getV()) {
                T[j].setV(true);
                newgroup = true;
            }

            //Find insertion index
            addtype i = findIndex(k);
            printf("\t\tFind index %" PRIu32 "\n", i);

            bool groupstart = T[i].getC() == 1 && T[i].getO() != false;
            bool groupend;
            if (i != MAX_ADRESS) { groupend = T[i + 1].getC() == 1 && T[i].getO() != false; }
            else { groupend = true; }

            //Check whether i should be 0 (Check all smaller Vs)
            bool setStart = false;
            if (i == MIN_ADRESS && j != MIN_ADRESS && !T[MIN_ADRESS].getV()) {
                setStart = true;
                for (int x = 1; x < j; x++) {
                    if (T[x].getV() != 0) {
                        setStart = false;
                        break;
                    }
                }
            }
            //If a new group needs to be formed, look for the end of the group
            if (newgroup && T[i].getO() && !setStart) {
                direction dir = up;
                while (dir != here) {
                    if (i == MAX_ADRESS) {
                        dir = here;
                    }
                    else if (T[i + 1].getC() == 1) {
                        i++;
                        dir = here;
                    }
                    else {
                        i = i + 1;
                    }
                };
            }

            //Decide to shift mem up or down
            //TODO: Maybe randomize
            int shift = 1;

            //Prevent Overflows
            if (T[MAX_ADRESS].getO() && !T[MIN_ADRESS].getO()) {
                shift = -1;
            }
            else if (T[MIN_ADRESS].getO() && !T[MAX_ADRESS].getO()) {
                shift = 1;
            }
            else if (T[MIN_ADRESS].getO() && T[MAX_ADRESS].getO()) {
                //Look which side will be shifted
                int k = MIN_ADRESS;
                int l = MAX_ADRESS;
                while (k != i && l != i && (T[k].getO() || T[l].getO())) {
                    if (T[k].getO()) { k++; }
                    if (T[l].getO()) { l--; }
                }
                if (k == i) {
                    shift = 1;
                }
                else if (l == i) {
                    shift = -1;
                }
            }

            //Edge cases where the location must be shifted
            bool setC = false;
            if (shift == -1) {
                if (groupstart && (!newgroup) && (T[i].getR() > rem) && T[i].getO() && (i != MIN_ADRESS)) {
                    T[i].setC(false);
                    setC = true;
                    i--;
                }
                else if (!newgroup && T[i].getR() > rem && T[i].getO() && i != MIN_ADRESS) {
                    i--;
                }
                else if (newgroup && T[i].getO() && i != MIN_ADRESS) {
                    if (i == MAX_ADRESS && j != MAX_ADRESS) {
                        bool checkPos = true;
                        for (int m = j + 1; m <= MAX_ADRESS; m++) {
                            if (T[m].getV()) { checkPos = false; break; }
                        }
                        if (!checkPos) {
                            i--;
                        }
                    }
                    else if (i != MAX_ADRESS) {
                        i--;
                    }
                }
            }
            if (shift == 1) {
                if (groupend && (!newgroup) && (T[i].getR() < rem) && T[i].getO() && (i != MAX_ADRESS)) {
                    i++;
                    T[i].setC(false);
                    setC = true;
                }
                else if (!newgroup && T[i].getR() < rem && T[i].getO() && i != MAX_ADRESS) {
                    i++;
                }
                else if (j == MIN_ADRESS && newgroup) {
                    i = MIN_ADRESS;
                }
            }

            //Store where the search started for later
            addtype startloc = i;
            //Check whether location is empty
            bool wasoccupied = T[i].getO();

            //Store values at found location
            printf("\t\tStoring Values\n");
            remtype R_old = T[i].getR();
            bool C_old = T[i].getC();
            bool O_old = T[i].getO();

            //Insert new values
            T[i].setR(rem);
            T[i].setO(true);
            if ((shift == 1) && !setC) {
                T[i].setC(C_old);
            }
            else if (shift == -1) {
                T[i].setC(newgroup);
            }

            if (setC && shift == -1) { T[i].setC(true); }

            //Update C Value
            if (shift == 1 && !newgroup) {
                C_old = setC;
            }

            //If the space was occupied shift mem
            if (wasoccupied) {
                printf("\t\tShifting Mem\n");
                while (O_old) {
                    i += shift;
                    //Store the values
                    remtype R_temp = T[i].getR();
                    bool C_temp = T[i].getC();
                    bool O_temp = T[i].getO();

                    //Put the old values in the new location
                    T[i].setR(R_old);
                    T[i].setO(true);
                    T[i].setC(C_old);

                    //Store the old values again
                    R_old = R_temp;
                    C_old = C_temp;
                    O_old = O_temp;

                    if (i == MIN_ADRESS || i == MAX_ADRESS) {
                        break;
                    }

                }
            }

            addtype x = (startloc < i) ? startloc : i;
            if (newgroup) {
                x = (j < x) ? j : x;
            }

            //Update the A values
            printf("\t\tUpdating A Values\n");
            while (T[x].getO() && x <= MAX_ADRESS) {
                int A_old;
                //Starting Value for A
                if (((int)x - 1) >= 0) {
                    A_old = T[x - 1].getA();
                }
                else {
                    A_old = 0;
                }

                //Update Based on C and V
                if (T[x].getC()) {
                    A_old += 1;
                }
                if (T[x].getV()) {
                    A_old -= 1;
                }
                T[x].setA(A_old);
                x++;
            }

            return true;
        }


    public:
        /**
         * Constructor
         */

        //Default constructor for mem-alloc
        Cleary() {}

        Cleary(int adressSize){
            printf("Creating Cleary Table\n");
            AS = adressSize;
            RS = HS-AS;
            tablesize = (int) pow(2,AS) + 2*BUFFER;
            size = (int) pow(2,AS);
            MAX_ADRESS = tablesize - 1;

            printf("\tAllocating Memory\n");
            cudaMallocManaged(&T, tablesize * sizeof(ClearyEntry<addtype, remtype>));

            printf("\tInitializing Entries\n");
            for(int i=0; i<tablesize; i++){
                new (&T[i]) ClearyEntry<addtype, remtype>();
            }

            h1 = 1;
            printf("\tDone\n");
        }

        /**
         * Destructor
         */
        ~Cleary(){
            cudaFree(T);
        }

        __host__ __device__
        bool Cleary::insert(keytype k){
            //Calculate Hash
            hashtype h = RHASH(h1, k);
            addtype j = getAdd(h);
            remtype rem = getRem(h);

            //Try Non-Exclusive Write
            ClearyEntry<addtype,remtype> old = 
                T[j].compareAndSwap(ClearyEntry<addtype, remtype>(), ClearyEntry<addtype, remtype>(rem, true, true, true, 0, false));

            //If not locked + not occupied then success
            if ((!old.getL()) && (!old.getO()) {
                return true;
            }

            //Get the locks
            addtype left = leftLock(j);
            addtype right = rightLock(j);

            if (!T[left].lock()) {
                return insert(k);
            }

            if (!T[right].lock()) {
                T[left].unlock();
                return insert(k);
            }
            
            //Do a read
            if (lookup(k)) {
                //Val already exists
                return false;
            }

            //Write
            bool res = insertIntoTable(k);
            T[left].unlock();
            T[right].unlock();

            return res;
        };

        __host__ __device__
        bool Cleary::lookup(uint64_t k){
            //Hash Key
            hashtype h = RHASH(h1, k);
            addtype j = getAdd(h);
            remtype rem = getRem(h);

            //If no values with add exist, return
            if(T[j].getV() == 0){
                return false;
            };

            int i = findIndex(k);

            if(T[i].getR() == rem){
                return true;
            }
        };

        __host__ __device__
        void Cleary::clear(){
            for(int i=0; i<tablesize; i++){
                T[i] = ClearyEntry<addtype, remtype>();
            }
        }

        __host__ __device__
        int Cleary::getSize(){
            return size;
        }

        __host__ __device__
        void Cleary::print(){
            printf("----------------------------------------------------------------\n");
            printf("|    i     |     R[i]       | C[i] | V[i] | O[i] | A[i] | L[i] |\n");
            printf("----------------------------------------------------------------\n");
            for(int i=0; i<tablesize; i++){
                if(T[i].getO()){
                    printf("|%-10i|%-16" PRIu64 "|%-6i|%-6i|%-6i|%-6i|%-6i|\n", i, T[i].getR(), T[i].getC(), T[i].getV(), T[i].getO(), T[i].getA(), T[i].getL());
                }
            }
            printf("----------------------------------------------------------------\n");
        }

        //No rehash
        __host__ __device__
        bool Cleary::rehash(){return true;}

};
