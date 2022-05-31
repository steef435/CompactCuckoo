#include "int_cu.h"

#ifndef ENTRYINCLUDED
#define ENTRYINCLUDED
#include "TableEntry.h"
#endif


template <class ADD, class REM>
class ClearyCuckooEntry : TableEntry <ADD, REM> {

private:
    int Rindex[2] = { 1, 56 };
    int Hindex[2] = { 57, 62 };
    int Oindex[2] = { 63, 63 };

public:
    __host__ __device__
    ClearyCuckooEntry(REM R, int H, bool O, bool onDevice=true) {
        TableEntry<ADD, REM>::val= 0;
        setR(R, onDevice);
        setH(H, onDevice);
        setO(O, onDevice);
    }

    __host__ __device__
    ClearyCuckooEntry() {
        TableEntry<ADD, REM>::val = 0;
    }


    __host__ __device__
    void exchValue(ClearyCuckooEntry* x) {
        //Atomically set this TableEntry<ADD, REM>::value to the new one
        //printf("\t\tBefore: %" PRIu64 ", %" PRIu64 "\n", TableEntry<ADD, REM>::val, x->getValue());
        uint64_cu old = atomicExch(TableEntry<ADD, REM>::getValPtr(), x->getValue());
        //Return an entry with prev TableEntry<ADD, REM>::val
        x->setValue(old);
        //printf("\t\tAfter: %" PRIu64 ", %" PRIu64 "\n", TableEntry<ADD, REM>::val, x->getValue());
        return;
    }


    __host__ __device__
    void setR(REM x, bool onDevice=true) {
        TableEntry<ADD, REM>::setBits(Rindex[0], Rindex[1], x, onDevice);

    }

    __host__ __device__
    REM getR() {
        return (REM)TableEntry<ADD, REM>::getBits(Rindex[0], Rindex[1]);
    }

    __host__ __device__
    void setH(int x, bool onDevice = true) {
        TableEntry<ADD, REM>::setBits(Hindex[0], Hindex[1], x, onDevice);
    }

    __host__ __device__
    int getH() {
        return (int) TableEntry<ADD, REM>::getBits(Hindex[0], Hindex[1]);
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
    void print() {
        printf("%" PRIu64  "\n", TableEntry<ADD, REM>::val);
    }

};
