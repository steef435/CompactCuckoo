#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <math.h>
#include <string> 
#include <math.h>

#include <bitset>

#ifndef HASHTABLE
#define HASHTABLE
#include "HashTable.h"
#endif

#ifndef HASHINCLUDED
#define HASHINCLUDED
#include "Hash.cpp"
#endif

#include "ClearyEntry.cu"

//Types to allow for changes

using addtype = uint32_t;
using remtype = uint64_t;
using hashtype = uint64_t;
using keytype = uint64_t;

//Enum for searching

enum direction{up, down, here};


class Cleary : public HashTable{
    //Allows for easy changing of the types
    
    typedef std::pair<addtype, remtype> keyTuple;

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

        keyTuple splitKey(keytype key){
            hashtype mask = ((hashtype) 1 << AS) - 1;
            addtype add = key & mask;
            remtype rem = key >> AS ;
            return std::make_pair(add + BUFFER,rem);
        }

        uint64_t reformKey(keyTuple split){
            remtype rem = split.second;
            hashtype reform = rem;
            reform = reform << AS;
            reform += (split.first - BUFFER);
            return reform;
        }

        int findIndex(uint64_t k){
            keyTuple js = splitKey( RHASH(h1, k) );
            addtype j = js.first;
            remtype rem =  js.second;
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
            return i;
        }


    public:
        /**
         * Constructor
         */
        Cleary(int adressSize){

            AS = adressSize;
            RS = HS-AS;
            tablesize = (int) pow(2,AS) + 2*BUFFER;
            size = (int) pow(2,AS);
            MAX_ADRESS = tablesize - 1;

            T = new ClearyEntry<addtype, remtype>[tablesize];

            for(int i=0; i<tablesize; i++){
                T[i] = ClearyEntry<addtype, remtype>(0, false, false, true, 0);
            }

            h1 = 1;
        }

        /**
         * Destructor
         */
        ~Cleary(){
            delete [] T;
        }

        bool insert(keytype k){
            //If the key is already inserted don't do anything
            if (lookup(k)) {
                return false;
            }

            keyTuple js = splitKey( RHASH(h1, k) );
            addtype j = js.first;
            remtype rem =  js.second;

            bool newgroup = false;

            //Check virgin bit and set
            if(!T[j].getV()){
                T[j].setV(true);
                newgroup = true;
            }

            //Find insertion index
            addtype i = findIndex(k);

            bool groupstart = T[i].getC() == 1 && T[i].getO() != false;
            bool groupend;
            if(i!=MAX_ADRESS){groupend = T[i+1].getC() == 1 && T[i].getO() != false;}
            else{groupend = true;}

            //Check whether i should be 0 (Check all smaller Vs)
            bool setStart = false;
            if(i == MIN_ADRESS && j!= MIN_ADRESS && !T[MIN_ADRESS].getV()){
                setStart = true;
                for(int x=1; x<j; x++){
                    if(T[x].getV() != 0){
                        setStart = false;
                        break;
                    }
                }
            }
            //If a new group needs to be formed, look for the end of the group
            if(newgroup && T[i].getO() && !setStart){
                direction dir = up;
                while(dir != here){
                    if(i==MAX_ADRESS){
                        dir = here;
                    }
                    else if(T[i+1].getC() == 1){
                        i++;
                        dir = here;
                    }
                    else{
                        i=i+1;
                    }
                };
            }

            //Decide to shift mem up or down
            int shift = (rand() % 2 == 0) ? -1 : 1;
            //Prevent Overflows
            if(T[MAX_ADRESS].getO() && !T[MIN_ADRESS].getO()){
                shift = -1;
            }else if(T[MIN_ADRESS].getO() && !T[MAX_ADRESS].getO()){
                shift = 1;
            }else if(T[MIN_ADRESS].getO() && T[MAX_ADRESS].getO()){
                //Look which side will be shifted
                int k = MIN_ADRESS;
                int l = MAX_ADRESS;
                while(k!=i && l!=i && (T[k].getO() || T[l].getO())){
                    if(T[k].getO()){k++;}
                    if(T[l].getO()){l--;}
                }
                if(k == i){
                    shift = 1;
                }else if(l == i){
                    shift = -1;
                }
            }

            //Edge cases where the location must be shifted
            bool setC = false;
            if(shift==-1){
                if(groupstart && (!newgroup) && (T[i].getR() > rem) && T[i].getO() && (i!=MIN_ADRESS)){
                    T[i].setC(false);
                    setC = true;
                    i--;
                }
                else if(!newgroup && T[i].getR() > rem && T[i].getO() && i!=MIN_ADRESS){
                    i--;
                }
                else if(newgroup && T[i].getO() && i!=MIN_ADRESS){
                    if(i == MAX_ADRESS && j != MAX_ADRESS){
                        bool checkPos = true;
                        for(int m=j+1; m<=MAX_ADRESS; m++){
                            if(T[m].getV()){checkPos = false;break;}
                        }
                        if(!checkPos){
                            i--;
                        }
                    }else if(i != MAX_ADRESS){
                        i--;
                    }
                }
            }
            if(shift==1){
                if(groupend && (!newgroup) && (T[i].getR() < rem) && T[i].getO() && (i!=MAX_ADRESS)){
                    i++;
                    T[i].setC(false);
                    setC = true;
                }
                else if(!newgroup && T[i].getR() < rem && T[i].getO() && i!= MAX_ADRESS){
                    i++;
                }else if(j==MIN_ADRESS && newgroup){
                    i=MIN_ADRESS;
                }
            }

            //Store where the search started for later
            addtype startloc = i;
            //Check whether location is empty
            bool wasoccupied = T[i].getO();

            //Store values at found location
            remtype R_old = T[i].getR();
            bool C_old= T[i].getC();
            bool O_old = T[i].getO();

            //Insert new values
            T[i].setR(rem);
            T[i].setO(true);
            if((shift == 1) && !setC){
                T[i].setC(C_old);
            }else if(shift == -1){
                T[i].setC(newgroup);
            }

            if(setC && shift == -1){T[i].setC(true);}

            //Update C Value
            if(shift == 1 && !newgroup){
                C_old = setC;
            }

            //If the space was occupied shift mem
            if(wasoccupied){
                while(O_old){
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

                    if(i == MIN_ADRESS || i == MAX_ADRESS){
                        break;
                    }

                }
            }

            addtype x = (startloc<i) ? startloc : i;
            if(newgroup){
                x = (j < x) ? j : x;
            }

            //Update the A values
            while(T[x].getO() && x<=MAX_ADRESS){
                int A_old;
                //Starting Value for A
                if(((int)x-1) >= 0){
                    A_old = T[x-1].getA();
                }else{
                    A_old = 0;
                }

                //Update Based on C and V
                if(T[x].getC()){
                    A_old += 1;
                }
                if(T[x].getV()){
                    A_old -= 1;
                }
                T[x].setA(A_old);
                x++;
            }

            return true;
        };

        bool lookup(uint64_t k){
            //Hash Key
            keyTuple js = splitKey( RHASH(h1, k) );
            addtype j = js.first;
            remtype rem =  js.second;

            //If no values with add exist, return
            if(T[j].getV() == 0){
                return false;
            };

            int i = findIndex(k);

            if(T[i].getR() == rem){
                return true;
            }else{
                return false;
            }
        };

        void clear(){
            for(int i=0; i<tablesize; i++){
                T[i] = ClearyEntry<addtype, remtype>();
            }
        }

        int getSize(){
            return size;
        }

        void print(){
            const char separator = ' ';
            std::cout << "-----------------------------------\n";
            std::cout << "|" << std::setw(6) << std::setfill(separator) << "i" << "|";
            std::cout << std::setw(20)<< std::setfill(separator) << "R[i]" << "|";
            std::cout << std::setw(5)<< std::setfill(separator) << "C[i]" << "|";
            std::cout << std::setw(5)<< std::setfill(separator) << "V[i]" << "|";
            std::cout << std::setw(5)<< std::setfill(separator) << "O[i]" << "|";
            std::cout << std::setw(5)<< std::setfill(separator) << "A[i]" << "|\n";
            for(int i=0; i<tablesize; i++){
                if(T[i].getO()){
                    std::cout << "|" << std::setw(6) << std::setfill(separator) << i << "|";
                    std::cout << std::setw(20)<< std::setfill(separator) << T[i].getR() << "|";
                    std::cout << std::setw(5)<< std::setfill(separator) << T[i].getC() << "|";
                    std::cout << std::setw(5)<< std::setfill(separator) << T[i].getV() << "|";
                    std::cout << std::setw(5)<< std::setfill(separator) << T[i].getO() << "|";
                    std::cout << std::setw(5)<< std::setfill(separator) << static_cast<int16_t>(T[i].getA()) << "| ";
                    T[i].print();
                }
            }
            std::cout << "-----------------------------------\n";
        }

        //No rehash
        bool rehash(){return true;}

};
