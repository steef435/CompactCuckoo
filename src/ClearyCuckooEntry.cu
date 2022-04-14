#ifndef ENTRYINCLUDED
#define ENTRYINCLUDED
#include "TableEntry.h"
#endif


template <class ADD, class REM>
class ClearyCuckooEntry : TableEntry <ADD, REM> {

private:
    int Rindex[2] = { 1, 56 };
    int Hindex[2] = { 57, 62 };
    int Oindex[2] = { 63, 64 };

public:
    __host__ __device__
    ClearyCuckooEntry(ADD R, int H, bool O) {
        val = 0;
        setR(R);
        setH(H);
        setO(O);
    }

    __host__ __device__
    ClearyCuckooEntry() {
        val = 0;
    }


    __host__ __device__
    void exchValue(ClearyCuckooEntry* x) {
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
    void setH(int x) {
        setBits(Hindex[0], Hindex[1], x);
    }

    __host__ __device__
    int getH() {
        return (int) getBits(Hindex[0], Hindex[1]);
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
    void print() {
        std::cout << std::bitset<64>(val) << "\n";
    }

};