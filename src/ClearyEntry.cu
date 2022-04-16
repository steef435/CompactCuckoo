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
    int Aindex[2] = { 60, 62 };
    int Lindex[2] = { 63, 63 };


public:
    __host__ __device__
    ClearyEntry(ADD R, bool O, bool V, bool C, int A, bool L) {
        val = 0;
        setR(R);
        setO(O);
        setV(V);
        setC(C);
        setA(A);
        setL(L);
    }

    __host__ __device__
    ClearyEntry() {
        ClearyEntry(0, false, false, true, 0, false);
    }

    __host__ __device__
    void exchValue(ClearyEntry* x) {
        //Atomically set this value to the new one
        uint64_t old = atomicExch(&val, x->getValue());
        //Return an entry with prev val
        x->setValue(old);
        return;
    }

    __host__ __device__
    void setR(REM x) {
        setBits(Rindex[0], Rindex[1], x);
    }

    __host__ __device__
    REM getR() {
        return (REM)getBits(Rindex[0], Rindex[1]);
    }

    __host__ __device__
    void setO(bool x) {
        setBits(Oindex[0], Oindex[1], x);
    }

    __host__ __device__
    bool getO() {
        return (bool)getBits(Oindex[0], Oindex[1]);
    }

    __host__ __device__
    void setV(bool x) {
        setBits(Vindex[0], Vindex[1], x);
    }

    __host__ __device__
    bool getV() {
        return (bool)getBits(Vindex[0], Vindex[1]);
    }

    __host__ __device__
    void setC(bool x) {
        setBits(Cindex[0], Cindex[1], x);
    }

    __host__ __device__
    bool getC() {
        return (bool)getBits(Cindex[0], Cindex[1]);
    }

    __host__ __device__
    void setA(int x) {
        setBits(Aindex[0], Aindex[1], signed_to_unsigned(x, Aindex[1]-Aindex[0]));
    }

    __host__ __device__
    int getA() {
        return unsigned_to_signed(getBits(Aindex[0], Aindex[1]), Aindex[1] - Aindex[0]);
    }

    __host__ __device__
    void setL(bool x) {
        setBits(Lindex[0], Lindex[1], x);
    }

    __host__ __device__
    bool getL() {
        return getBits(Lindex[0], Lindex[1]);
    }

    __host__ __device__
    void print() {
        std::cout << std::bitset<64>(val) << "\n";
    }

};