#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <math.h>
#include <assert.h>

#include <bitset>
#include <inttypes.h>

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

#include "ClearyEntry.cu"

//Types to allow for changes

using addtype = uint32_t;
using remtype = uint64_cu;
using hashtype = uint64_cu;
using keytype = uint64_cu;

//Enum for searching

enum direction{up, down, here};


class Cleary : public HashTable{
    //Allows for easy changing of the types

    private:
        //Constant Vars
        const static int HS = 59;       //HashSize
        const static int BUFFER = 0; //Space assigned for overflow
        const static int MAXLOOPS = 24;
        const static int A_UNDEFINED = 0;

        bool GPU;

        //Vars assigned at construction
        int AS;                  //AdressSize
        int RS;                  //RemainderSize
        int size;                //Allocated Size of Table
        int tablesize;              //Actual size of table with buffer
        int occupancy = 0;

        addtype MAX_ADRESS;
        addtype MIN_ADRESS = 0;

        //Tables
        ClearyEntry<addtype, remtype>* T;

        //Hash function ID
        int h1;

        GPUHEADER
            addtype getAdd(keytype key) {
            hashtype mask = ((hashtype)1 << AS) - 1;
            addtype add = key & mask;
            return add;
        }

        GPUHEADER
            remtype getRem(keytype key) {
            remtype rem = key >> AS;
            return rem;
        }

        GPUHEADER
            uint64_cu reformKey(addtype add, remtype rem) {
            rem = rem << AS;
            rem += add;
            return rem;
        }

        GPUHEADER
        addtype findIndex(uint64_cu k){           
            hashtype h = RHASH(h1, k);
            addtype j = getAdd(h);
            remtype rem = getRem(h);

            addtype i = j;
            int cnt = 0;

            //printf("\t\t\t\t\t\t\t\t\t\t\t%i: Finding Index from %" PRIu32 "\n", getThreadID(), i);

            //Find first well defined A value
            while(T[i].getA() == A_UNDEFINED && i>=MIN_ADRESS && T[i].getO()){
                cnt = cnt - (T[i].getV() ? 1 : 0);
                i=i-1;
                if (i > MAX_ADRESS) {
                    break;
                }
            };

            //printf("\t\t\t\t\t\t\t\t\t\t\t%i: First well defined: %" PRIu32 "\n", getThreadID(), i);
            if (i <= MAX_ADRESS && i >= MIN_ADRESS) {
                cnt = cnt + T[i].getA();
                //printf("\t\t\t\t\t\t\t\t\t\t\t%i: Cnt: %i\n", getThreadID(), cnt);
            }

            //Look for the relevant group
            //printf("\t\t\t\t\t\t\t\t\t\t\t%i: Find relevant group\n", getThreadID());
            direction dir = up;
            if(cnt < 0){
                dir = up;
                //printf("\t\t\t\t\t\t\t\t\t\t\t%i: Dir Up %" PRIu32 "\n", getThreadID(), i);
                while(cnt != 0 && i != MAX_ADRESS){
                    i = i+1;
                    cnt = cnt + (T[i].getC() ? 1 : 0);
                };
                if(T[i].getR() >= rem){
                    dir = here;
                }
            }else if(cnt > 0){
                dir = down;
                //printf("\t\t\t\t\t\t\t\t\t\t\t%i: Dir down %" PRIu32 "\n", getThreadID(), i);
                while(cnt != 0 && i != MIN_ADRESS){
                    cnt = cnt - (T[i].getC() ? 1 : 0);
                    i = i - 1;
                }
                if(T[i].getR() <= rem){dir = here;}
            }else{
                //printf("\t\t\t\t\t\t\t\t\t\t\t%i: End Case\n", getThreadID());
                if (i > MAX_ADRESS) {
                    i = 0;
                    //IF val is being inserted first time, stop here
                    if (!T[j].getV()) {
                        return i;
                    }
                }

                if(T[i].getR() > rem){dir = down;}
                else if(T[i].getR() < rem){dir = up;}
                else{
                    //printf("\t\t\t\t\t\t\t\t\t\t\t%i: End Else Case\n", getThreadID());
                    dir = here;}
            }
            //printf("\t\t\t\t\t\t\t\t\t\t\t%i: Relevant Group: %" PRIu32 "\n", getThreadID(), i);

            //Look inside of the group
            //printf("\t\t\t\t\t\t\t\t\t\t\t%i: Look inside group\n", getThreadID());
            switch (dir){
                case here:
                    //printf("\t\t\t\t\t\t\t\t\t\t\t%i: Here\n", getThreadID());
                    break;

                case down:
                    while (dir != here) {
                        assert(i <= MAX_ADRESS);
                        //printf("\t\t\t\t\t\t\t\t\t\t\t%i: Going Down %" PRIu32 "\n", getThreadID(), i);
                        if (T[i].getC() == 1 || i == MIN_ADRESS) { dir = here; }
                        else {
                            i = i - 1;
                            if (T[i].getR() <= rem) {
                                dir = here;
                            }
                        }
                    }
                    break;

                case up:
                    while (dir != here) {
                        assert(i <= MAX_ADRESS);
                        //printf("\t\t\t\t\t\t\t\t\t\t\t%i: Going Up %" PRIu32 "\n", getThreadID(), i);
                        if (i == MAX_ADRESS) {
                            dir = here;
                        }
                        else if (T[i + 1].getC() == 1) {
                            dir = here;
                        }
                        else {
                            i = i + 1;
                            if (T[i].getR() >= rem) {
                                dir = here;
                            }
                        }
                    }
                    break;

                default:
                    break;
            };
            return i;
        }

        GPUHEADER
        addtype leftLock(addtype i) {
            if (i == MIN_ADRESS) {
                return i;
            }
            while (T[i].getO() && i>MIN_ADRESS) {
                i -= 1;
            }
            return i;
        }

        GPUHEADER
        addtype rightLock(addtype i) {
            if (i == MAX_ADRESS) {
                return i;
            }
            while (T[i].getO() && i<MAX_ADRESS) {
                i += 1;
            }
            return i;
        }


        GPUHEADER
        bool insertIntoTable(keytype k, addtype left, addtype right) {
            //printf("\t\t\t\t\t\t\t%i: Inserting Into Table\n", getThreadID());

            hashtype h = RHASH(h1, k);
            addtype j = getAdd(h);
            remtype rem = getRem(h);

            bool newgroup = false;

            //Find insertion index
            addtype i = findIndex(k);
            //printf("\t\t\t\t\t\t\t%i: Index Found %" PRIu32 "\n", getThreadID(), i);

            //Check virgin bit and set
            if (!T[j].getV()) {
                //printf("\t\t\t\t\t\t\t%i: Set VBit at %" PRIu32 "\n", getThreadID(), j);
                T[j].setV(true);
                newgroup = true;
            }

            //printf("\t\t\t\t\t\t\t%i: Group Start/Groupend\n", getThreadID());
            bool groupstart = T[i].getC() == 1 && T[i].getO() != false;
            bool groupend;
            if (i != MAX_ADRESS) { groupend = T[i + 1].getC() == 1 && T[i].getO() != false; }
            else { groupend = true; }

            //Check whether i should be 0 (Check all smaller Vs
            //printf("\t\t\t\t\t\t\t\t\t\t%i: Check if i is 0 \n", getThreadID());
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
            //printf("\t\t\t\t\t\t\t\t\t\t%i: Look for new group\n",getThreadID());
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
            //printf("\t\t\t\t\t\t\t\t\t\t%i: Prevent Overflows %" PRIu32 "\n", getThreadID(), i);
            //Prevent Overflows
            if (T[left].getO()) {
                shift = 1;
            }
            else if (T[right].getO()) {
                shift = -1;
            }
            

            //Edge cases where the location must be shifted
            //printf("\t\t\t\t\t\t\t\t\t\t%i: Edge Cases %" PRIu32 "\n", getThreadID(), i);
            bool setC = false;
            if (shift == -1) {
                //printf("\t\t\t\t\t\t\t\t\t\t%i: Shift -1\n",getThreadID());
                if (groupstart && (!newgroup) && (T[i].getR() > rem) && T[i].getO() && (i != MIN_ADRESS)) {
                    //printf("\t\t\t\t\t\t\t\t\t\t%i: Shift Case 1\n",getThreadID());
                    T[i].setC(false);
                    setC = true;
                    i--;
                }
                else if (!newgroup && T[i].getR() > rem && T[i].getO() && i != MIN_ADRESS) {
                    //printf("\t\t\t\t\t\t\t\t\t\t%i: Shift Case 2\n",getThreadID());
                    i--;
                }
                else if (newgroup && T[i].getO() && i != MIN_ADRESS) {
                    //printf("\t\t\t\t\t\t\t\t\t\t%i: Shift Case 3\n",getThreadID());
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
                        //printf("\t\t\t\t\t\t\t\t\t\t%i: Shift Case 4\n",getThreadID());
                        i--;
                    }
                }
            }
            if (shift == 1) {
                //printf("\t\t\t\t\t\t\t\t\t\t%i: Shift 1\n", getThreadID());
                if (groupend && (!newgroup) && (T[i].getR() < rem) && T[i].getO() && (i != MAX_ADRESS)) {
                    //printf("\t\t\t\t\t\t\t\t\t\t%i: Shift Case 5:%" PRIu32 "\n",getThreadID(),i);
                    i++;
                    //printf("\t\t\t\t\t\t\t\t\t\t%i: Iter Case 5:%" PRIu32 "\n",getThreadID(),i);
                    T[i].setC(false);
                    //printf("\t\t\t\t\t\t\t\t\t\t%i: SetC 5:%" PRIu32 "\n",getThreadID(),i);
                    setC = true;
                    //printf("\t\t\t\t\t\t\t\t\t\t%i: Done Shift Case 5\n",getThreadID());
                }
                else if (!newgroup && T[i].getR() < rem && T[i].getO() && i != MAX_ADRESS) {
                    //printf("\t\t\t\t\t\t\t\t\t\t%i: Shift Case 6\n",getThreadID());
                    i++;
                }
                else if (j == MIN_ADRESS && newgroup) {
                    //printf("\t\t\t\t\t\t\t\t\t\t%i: Shift Case 7\n",getThreadID());
                    i = MIN_ADRESS;
                }
            }

            //printf("\t\t\t\t\t\t\t\t\t\t%i: Storing searchstart %" PRIu32 "\n", getThreadID(), i);

            //Store where the search started for later
            addtype startloc = i;
            assert(0<=i && i<=MAX_ADRESS);

            //Check whether location is empty
            bool wasoccupied = T[i].getO();

            //printf("\t\t\t\t\t\t\t\t\t\t%i: Storing old Values at %" PRIu32 "\n", getThreadID(), i);
            //Store values at found location
            remtype R_old = T[i].getR();
            bool C_old = T[i].getC();
            bool O_old = T[i].getO();

            //Insert new values
            //printf("\t\t\t\t\t\t\t\t\t\t%i: Setting new Values at %" PRIu32 "\n", getThreadID(), i);
            T[i].setR(rem);

            T[i].setO(true);
            if ((shift == 1) && !setC) {
                T[i].setC(C_old);
            }
            else if (shift == -1) {
                T[i].setC(newgroup);
            }

            //printf("\t\t\t\t\t\t\t\t\t\t%i: Update C %" PRIu32 "\n", getThreadID(), i);
            if (setC && shift == -1) { T[i].setC(true); }
            //Update C Value
            if (shift == 1 && !newgroup) {
                C_old = setC;
            }

            //If the space was occupied shift mem
            //printf("\t\t\t\t\t\t\t\t\t\t%i: Shifting Mem from %" PRIu32 "\n", getThreadID(), i);
            if (wasoccupied) {
                while (O_old) {
                    //printf("\t\t\t\t\t\t\t\t\t\t%i: Shift%" PRIu32 "\n", getThreadID(), i);
                    i += shift;
                    assert(0<=i && i<=MAX_ADRESS);
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

            if(A_UNDEFINED != 0){
                //Find the first well-defined A
                //printf("\t\t\t\t\t\t\t\t\t\t%i: Find Start of Group %" PRIu32 "\n", getThreadID(), i);
                addtype x = startloc;
                while(T[x].getA() == A_UNDEFINED && x!=MIN_ADRESS) {
                    x--;
                }
                if (x != MAX_ADRESS && !T[x].getO()) {
                    x++;
                }

                //Update the A values
                //printf("\t\t\t\t\t\t\t\t\t\t%i: Updating A from %" PRIu32 "\n", getThreadID(), x);
                int A_old = 0;
                while (T[x].getO() && x <= MAX_ADRESS) {
                    //printf("\t\t\t\t\t\t\t\t\t\t\t%i: Setting A %" PRIu32 "\n", getThreadID(), x);
                    assert(0<=x && x<=MAX_ADRESS);
                    //Update Based on C and V
                    if (T[x].getC()) {
                        A_old += 1;
                    }
                    if (T[x].getV()) {
                        A_old -= 1;
                    }
                    T[x].setA(A_old);
                    x++;
                    if (x > MAX_ADRESS) {
                        break;
                    }
                }
            }

            //printf("\t\tAfterallupdates");

            return true;
        }


    public:
        /**
         * Constructor
         */

        //Default constructor for mem-alloc
        Cleary() {}

        Cleary(int adressSize){
            //printf("Creating Cleary Table\n");
            AS = adressSize;
            RS = HS-AS;
            tablesize = (int) pow(2,AS) + 2*BUFFER;
            size = (int) pow(2,AS);
            MAX_ADRESS = tablesize - 1;

            //printf("\tAllocating Memory\n");
            #ifdef GPUCODE
            gpuErrchk(cudaMallocManaged(&T, tablesize * sizeof(ClearyEntry<addtype, remtype>)));
            #else
            T = new ClearyEntry<addtype, remtype>[tablesize];
            #endif

            //printf("\tInitializing Entries\n");
            for(int i=0; i<tablesize; i++){
                new (&T[i]) ClearyEntry<addtype, remtype>();
            }

            h1 = 0;
            //printf("\tDone\n");
        }

        /**
         * Destructor
         */
        ~Cleary() {
            #ifdef GPUCODECODE
            gpuErrchk(cudaFree(T));
            #else
            delete[] T;
            #endif
        }

        GPUHEADER_D
        bool insert(keytype k){
            //printf("\tInserting %" PRIu64 "\n", k);
            //Calculate Hash
            hashtype h = RHASH(h1, k);
            addtype j = getAdd(h);
            remtype rem = getRem(h);

            int counter=0;

            while (true) {
                //__syncthreads();
                counter++;
                assert(0<=j && j<=MAX_ADRESS);
                //assert(counter < 30000);

                //Try Non-Exclusive Write
                //printf("\t\t\t\t\t\t\t\t%i: Trying Non-Exclusive Write at %" PRIu32 "\n", getThreadID(), j);
                ClearyEntry<addtype, remtype> def(0, false, false, true, 0, false, false);
                ClearyEntry<addtype, remtype> newval(rem, true, true, true, 0, false, false);

                ClearyEntry<addtype, remtype> old(T[j].compareAndSwap(&def, &newval));

                //If not locked + not occupied then success
                if ((!old.getL()) && (!old.getO())) {
                    //printf("\t\t\t\t\t\t\t\t%i: Non-Exclusive Success\n", getThreadID());
                    return true;
                }

                //Else Need Exclusivity
                addtype left = leftLock(j);
                addtype right = rightLock(j);

                assert(0<=left && left<=MAX_ADRESS);
                assert(0<=right && right<=MAX_ADRESS);

                if (!T[left].lock(left == MIN_ADRESS)) {
                    //printf("\t\t\t\t\t\t\t\t%i: Left Failed at%" PRIu32 "\n", getThreadID(), left);
                    //__nanosleep(1000);
                    continue;
                }
                //printf("\t\t\t\t\t\t\t\t%i: Left Retrieved at%" PRIu32 "\n", getThreadID(), left);
                //printf("\t\t\t\t\t\t\t\t%i: Trying Right at%" PRIu32 "\n", getThreadID(), right);
                if (!T[right].lock(right == MAX_ADRESS)) {
                    //printf("\t\t\t\t\t\t\t\t%i: Right Failed at%" PRIu32 "\n", getThreadID(), right);
                    T[left].unlock();
                    //printf("\t\t\t\t\t\t\t\t%i: Left Unlocked at%" PRIu32 "\n", getThreadID(), left);
                    //__nanosleep(1000);
                    continue;
                }
                //printf("\t\t\t\t\t\t\t\t%i: Right Retrieved at%" PRIu32 "\n", getThreadID(), right);

                //Do a read
                if (lookup(k)) {
                    //Val already exists
                    //printf("\t\tVal Already Exists\n");
                    T[left].unlock();
                    T[right].unlock();
                    //printf("\t\t\t\t\t\t\t\t%i: Left Unlocked at%" PRIu32 "\n", getThreadID(), left);
                    //printf("\t\t\t\t\t\t\t\t%i: Right Unlocked at%" PRIu32 "\n", getThreadID(), right);
                    return false;
                }

                //Write
                //printf("\t\t\t\t\t\t\t\t%i: Exclusive Write\n", getThreadID());
                bool res = insertIntoTable(k, left, right);
                T[left].unlock();
                T[right].unlock();
                //printf("\t\t\t\t\t\t\t\t%i: Left Unlocked at%" PRIu32 "\n", getThreadID(), left);
                //printf("\t\t\t\t\t\t\t\t%i: Right Unlocked at%" PRIu32 "\n", getThreadID(), right);
                //printf("\t\t\t\t\t\t\t\t%i: Insertion Success\n", getThreadID());
                //printf("\tAfterInsertion");
                return res;
            }
        };

        GPUHEADER
        bool lookup(uint64_cu k){
            //printf("\t\tLookup %" PRIu64 "\n", k);
            //Hash Key
            hashtype h = RHASH(h1, k);
            addtype j = getAdd(h);
            remtype rem = getRem(h);

            //If no values with add exist, return
            if(T[j].getV() == 0){
                //printf("\t\t\tV not set\n");
                return false;
            };

            addtype i = findIndex(k);
            assert(0<=i && i<=MAX_ADRESS);
            //printf("\t\tFind Add   %" PRIu32 "\n", j);
            //printf("\t\tFind Index %" PRIu32 "\n", i);

            if(T[i].getR() == rem){
                return true;
            }
            else {
                //printf("\t\t\tOriginalIndex:%" PRIu32 " FoundIndex:%" PRIu32 "\n",j, i);
                //printf("\t\t\tFoundR:%" PRIu64 " ActualR:%" PRIu64 "\n", T[i].getR(), rem);
            }

            return false;
        };

        GPUHEADER
        void clear(){
            for(int i=0; i<tablesize; i++){
                new (&T[i]) ClearyEntry<addtype, remtype>();
            }
        }

        GPUHEADER
        int getSize(){
            return size;
        }

        GPUHEADER
        void print(){
            printf("----------------------------------------------------------------\n");
            printf("|    i     |     R[i]       | C[i] | V[i] | O[i] | A[i] | L[i] |\n");
            printf("----------------------------------------------------------------\n");
            for(int i=0; i<tablesize; i++){
                if(true){
                    printf("|%-10i|%-16" PRIu64 "|%-6i|%-6i|%-6i|%-6i|%-6i|\n", i, T[i].getR(), T[i].getC(), T[i].getV(), T[i].getO(), T[i].getA(), T[i].getL());
                }
            }
            printf("----------------------------------------------------------------\n");
        }

        //No rehash
        GPUHEADER
        bool rehash(){return true;}

};
