#ifndef ENTRYINCLUDED
#define ENTRYINCLUDED
#include "TableEntry.h"
#endif


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
        val = 0;
        setR(R, onDevice);
        setO(O, onDevice);
        setV(V, onDevice);
        setC(C, onDevice);
        setA(A, onDevice);
        setL(L, onDevice);
    }

    __host__ __device__
    ClearyEntry(uint64_t x) {
        val = x;
    }

    __host__ __device__
    ClearyEntry() : ClearyEntry(0, false, false, true, 0, false) {}

    __host__ __device__
    void exchValue(ClearyEntry* x) {
        //Atomically set this value to the new one
        uint64_t old = atomicExch(&val, x->getValue());
        //Return an entry with prev val
        x->setValue(old);
        return;
    }

    __host__ __device__
    void setR(REM x, bool onDevice = true) {
        setBits(Rindex[0], Rindex[1], x, onDevice);
    }

    __host__ __device__
    REM getR() {
        return (REM)getBits(Rindex[0], Rindex[1]);
    }

    __host__ __device__
    void setO(bool x, bool onDevice = true) {
        setBits(Oindex[0], Oindex[1], x, onDevice);
    }

    __host__ __device__
    bool getO() {
        return (bool)getBits(Oindex[0], Oindex[1]);
    }

    __host__ __device__
    void setV(bool x, bool onDevice = true) {
        setBits(Vindex[0], Vindex[1], x, onDevice);
    }

    __host__ __device__
    bool getV() {
        return (bool)getBits(Vindex[0], Vindex[1]);
    }

    __host__ __device__
    void setC(bool x, bool onDevice = true) {
        setBits(Cindex[0], Cindex[1], x, onDevice);
    }

    __host__ __device__
    bool getC() {
        return (bool)getBits(Cindex[0], Cindex[1]);
    }

    __host__ __device__
    void setA(int x, bool onDevice = true) {
        setBits(Aindex[0], Aindex[1], signed_to_unsigned(x, Aindex[1]-Aindex[0]), onDevice);
    }

    __host__ __device__
    int getA() {
        return unsigned_to_signed(getBits(Aindex[0], Aindex[1]), Aindex[1] - Aindex[0]);
    }

    __host__ __device__
    void setL(bool x, bool onDevice = true) {
        setBits(Lindex[0], Lindex[1], x, onDevice);
    }

    __host__ __device__
    bool getL() {
        return getBits(Lindex[0], Lindex[1]);
    }

    //Need to do with CAS
    __host__ __device__
    bool lock() {
        //Store old Value
        uint64_t oldval = val;
        //Make the new value with lock locked
        uint64_t newval = val;
        setBits(Lindex[0], Lindex[1], ((uint64_t) 1), &newval, false);

        //If Lockbit was set return false
        if (getBits(Lindex[0], Lindex[1], oldval)) {
            //printf("\t\t\tLockbit Already Set\n");
            return false;
        }

        //Swap if the old value hasn't changed
        uint64_t res = atomicCAS(&val, oldval, newval);

        //Check if lockbit is now set and wasn't already
        if (getBits(Lindex[0], Lindex[1]) && !getBits(Lindex[0], Lindex[1], res)) {
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
        uint64_t oldval = val;
        //Make the new value with lock unlocked
        uint64_t newval = val;
        setBits(Lindex[0], Lindex[1], ((uint64_t) 0), &newval, false);

        //If Lockbit was already free return
        if (!getBits(Lindex[0], Lindex[1], oldval)) {
            return true;
        }

        //Swap if the old value hasn't changed
        uint64_t res = atomicCAS(&val, oldval, newval);

        //Check if lockbit is now not set
        if (!getBits(Lindex[0], Lindex[1])) {
            return true;
        }
        else {
            return false;
        }
    }

    __host__ __device__
    void print() {
        printf("%" PRIu64  "\n", val);
    }

    __host__ __device__
    ClearyEntry<ADD, REM> compareAndSwap(ClearyEntry<ADD, REM> comp, ClearyEntry<ADD, REM> swap) {
        uint64_t newVal = atomicCAS(&val, comp.getValue(), swap.getValue());
        return ClearyEntry(newVal);
    }

};