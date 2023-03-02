#ifndef MAIN
#define MAIN
#include "main.h"
#endif

#ifndef ENTRYINCLUDED
#define ENTRYINCLUDED
#include "TableEntry.h"
#endif


template <class ADD, class REM>
class ClearyCuckooEntry : public TableEntry <ADD, REM> {

private:
    int Rindex[2] = { 1, 58 };
    int Hindex[2] = { 59, 62 };
    int Oindex[2] = { 63, 63 };

public:
    GPUHEADER
    ClearyCuckooEntry ( REM R, int H, bool O, bool onDevice=true) noexcept {
        TableEntry<ADD, REM>::val= 0;
        setR(R, onDevice);
        setH(H, onDevice);
        setO(O, onDevice);
        return;
    }

    GPUHEADER
    ClearyCuckooEntry() noexcept {
        TableEntry<ADD, REM>::val = 0;
        return;
    }

    GPUHEADER
        ClearyCuckooEntry(uint64_cu x) noexcept {
        TableEntry<ADD, REM>::val = x;
        return;
    }


    GPUHEADER
    void exchValue(ClearyCuckooEntry* x) {
        //Atomically set this TableEntry<ADD, REM>::value to the new one
        #ifdef GPUCODE
        uint64_cu old = atomicExch(TableEntry<ADD, REM>::getValPtr(), x->getValue());
        #else
        uint64_cu old = (*(TableEntry<ADD, REM>::getAtomValPtr())).exchange(x->getValue());
        #endif
        //Return an entry with prev TableEntry<ADD, REM>::val
        x->setValue(old);
        return;
    }


    GPUHEADER
    void setR(REM x, bool onDevice=true) {
        TableEntry<ADD, REM>::setBits(Rindex[0], Rindex[1], x, onDevice);
        return;

    }

    GPUHEADER
    REM getR() {
        return (REM)TableEntry<ADD, REM>::getBits(Rindex[0], Rindex[1]);
    }

    GPUHEADER
    void setH(int x, bool onDevice = true) {
        TableEntry<ADD, REM>::setBits(Hindex[0], Hindex[1], x, onDevice);
        return;
    }

    GPUHEADER
    int getH() {
        return (int) TableEntry<ADD, REM>::getBits(Hindex[0], Hindex[1]);
    }

    GPUHEADER
    void setO(bool x, bool onDevice = true) {
        TableEntry<ADD, REM>::setBits(Oindex[0], Oindex[1], x, onDevice);
        return;
    }

    GPUHEADER
    bool getO() {
        return (bool)TableEntry<ADD, REM>::getBits(Oindex[0], Oindex[1]);
    }

    GPUHEADER
    void clear() {
        TableEntry<ADD, REM>::val = 0;
    }

    GPUHEADER
    void print() {
#ifdef GPUCODE
        printf("%" PRIu64  "\n", TableEntry<ADD, REM>::val);
#else
        printf("%" PRIu64  "\n", TableEntry<ADD, REM>::val.load());
#endif
        return;
    }

};
