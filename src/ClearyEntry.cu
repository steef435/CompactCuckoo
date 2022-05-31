#include "int_cu.h"


#ifndef ENTRYINCLUDED
#define ENTRYINCLUDED
#include "TableEntry.h"
#endif
#include <math.h>


template <class ADD, class REM>
class ClearyEntry : TableEntry <ADD, REM> {

private:
    int Rindex[2] = {  1, 56 };
    int Oindex[2] = { 57, 57 };
    int Vindex[2] = { 58, 58 };
    int Cindex[2] = { 59, 59 };
    int Lindex[2] = { 60, 60 };
    int Aindex[2] = { 61, 63 };


public:
    __host__ __device__
    ClearyEntry(REM R, bool O, bool V, bool C, int A, bool L, bool onDevice = true) {
        TableEntry<ADD, REM>::val = 0;
        setR(R, onDevice);
        setO(O, onDevice);
        setV(V, onDevice);
        setC(C, onDevice);
        setA(A, onDevice);
        setL(L, onDevice);
    }

    __host__ __device__
    ClearyEntry(uint64_cu x) {
        TableEntry<ADD, REM>::val = x;
    }

    __host__ __device__
    ClearyEntry() : ClearyEntry(0, false, false, true, 0, false) {}

    __host__ __device__
    void exchValue(ClearyEntry* x) {
        //Atomically set this value to the new one
        uint64_cu old = atomicExch(TableEntry<ADD, REM>::getValPtr(), x->getValue());
        //Return an entry with prev val
        x->setValue(old);
        return;
    }

    __host__ __device__
    void setR(REM x, bool onDevice = true) {
        TableEntry<ADD, REM>::setBits(Rindex[0], Rindex[1], x, onDevice);
    }

    __host__ __device__
    REM getR() {
        return (REM)TableEntry<ADD, REM>::getBits(Rindex[0], Rindex[1]);
    }

    __host__ __device__
    void setO(bool x, bool onDevice = true) {
        TableEntry<ADD, REM>::setBits(Oindex[0], Oindex[1], x, onDevice);
    }

    __host__ __device__
    bool getO() {
        return (bool)TableEntry<ADD, REM>::getBits(Oindex[0], Oindex[1]);
    }

    __host__ __device__
    void setV(bool x, bool onDevice = true) {
        TableEntry<ADD, REM>::setBits(Vindex[0], Vindex[1], x, onDevice);
    }

    __host__ __device__
    bool getV() {
        return (bool)TableEntry<ADD, REM>::getBits(Vindex[0], Vindex[1]);
    }

    __host__ __device__
    void setC(bool x, bool onDevice = true) {
        TableEntry<ADD, REM>::setBits(Cindex[0], Cindex[1], x, onDevice);
    }

    __host__ __device__
    bool getC() {
        return (bool)TableEntry<ADD, REM>::getBits(Cindex[0], Cindex[1]);
    }

    __host__ __device__
    void setA(int x, bool onDevice = true) {
        int Amin = -pow(2, (Aindex[1] - Aindex[0]) - 1);
        int Amax = pow(2, (Aindex[1] - Aindex[0]) - 1);

        //printf("Amin:%i Amax:%i", Amin, Amax);

        if (x > Amax-1) {
            x = Amax;
        }
        if (x < Amin) {
            x = Amax;
        }

        TableEntry<ADD, REM>::setBits(Aindex[0], Aindex[1], TableEntry<ADD, REM>::signed_to_unsigned(x, Aindex[1]-Aindex[0]), onDevice);
    }

    __host__ __device__
    int getA() {
        return TableEntry<ADD, REM>::unsigned_to_signed(TableEntry<ADD, REM>::getBits(Aindex[0], Aindex[1]), Aindex[1] - Aindex[0]);
    }

    __host__ __device__
    void setL(bool x, bool onDevice = true) {
        TableEntry<ADD, REM>::setBits(Lindex[0], Lindex[1], x, onDevice);
    }

    __host__ __device__
    bool getL() {
        return TableEntry<ADD, REM>::getBits(Lindex[0], Lindex[1]);
    }

    //Need to do with CAS
    __host__ __device__
    bool lock() {
        //Store old TableEntry<ADD, REM>::value
        uint64_cu oldval = TableEntry<ADD, REM>::val;
        //Make the new value with lock locked
        uint64_cu newval = TableEntry<ADD, REM>::val;
        TableEntry<ADD, REM>::setBits(Lindex[0], Lindex[1], ((uint64_cu) 1), &newval, false);

        //If Lockbit was set return false
        if (TableEntry<ADD, REM>::getBits(Lindex[0], Lindex[1], oldval)) {
            //printf("\t\t\tLockbit Already Set\n");
            return false;
        }

        //Swap if the old value hasn't changed
        uint64_cu res = atomicCAS(TableEntry<ADD, REM>::getValPtr(), oldval, newval);

        //Check if lockbit is now set and wasn't already
        if (TableEntry<ADD, REM>::getBits(Lindex[0], Lindex[1]) && !TableEntry<ADD, REM>::getBits(Lindex[0], Lindex[1], res)) {
            //printf("\t\t\tSuccess\n");
            return true;
        }
        else {
            //printf("\t\t\tFail\n");
            return false;
        }
    }

    __host__ __device__
    bool unlock() {
        //Store old Value
        uint64_cu oldval = TableEntry<ADD, REM>::val;
        //Make the new value with lock unlocked
        uint64_cu newval = TableEntry<ADD, REM>::val;
        TableEntry<ADD, REM>::setBits(Lindex[0], Lindex[1], ((uint64_cu) 0), &newval, false);

        //If Lockbit was already free return
        if (!TableEntry<ADD, REM>::getBits(Lindex[0], Lindex[1], oldval)) {
            return true;
        }

        //Swap if the old value hasn't changed
        uint64_cu res = atomicCAS(TableEntry<ADD, REM>::getValPtr(), oldval, newval);

        //Check if lockbit is now not set
        if (!TableEntry<ADD, REM>::getBits(Lindex[0], Lindex[1])) {
            return true;
        }
        else {
            return false;
        }
    }

    __host__ __device__
    void print() {
        printf("%" PRIu64  "\n", TableEntry<ADD, REM>::val);
    }

    __host__ __device__
    ClearyEntry<ADD, REM> compareAndSwap(ClearyEntry<ADD, REM> comp, ClearyEntry<ADD, REM> swap) {
        uint64_cu newVal = atomicCAS(TableEntry<ADD, REM>::getValPtr(), comp.getValue(), swap.getValue());
        return ClearyEntry(newVal);
    }

};
