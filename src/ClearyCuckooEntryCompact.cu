#ifndef MAIN
#define MAIN
#include "main.h"
#endif

#ifndef ENTRYINCLUDED
#define ENTRYINCLUDED
#include "TableEntry.h"
#endif


template <class ADD, class REM>
class ClearyCuckooEntryCompact : public TableEntry <ADD, REM> {

private:
    int valSize = 8;

    int Rindex[2] = { 1, 4 };
    int Hindex[2] = { 5, 6 };
    int Oindex[2] = { 7, 7 };

    GPUHEADER
    int indexCalc(int locIndex, int valIndex) {
        return locIndex * valSize + valIndex;
    }

public:

    GPUHEADER
    ClearyCuckooEntryCompact() noexcept {
        TableEntry<ADD, REM>::val = 0;
        return;
    }

    GPUHEADER
    void exchValue(ClearyCuckooEntryCompact* x, int locIndex) {
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
    void setR(REM x, int locIndex,  bool onDevice=true) {
        TableEntry<ADD, REM>::setBits(indexCalc(locIndex, Rindex[0]), indexCalc(locIndex, Rindex[1]), x, onDevice);
        return;

    }

    GPUHEADER
    REM getR(int locIndex) {
        return (REM)TableEntry<ADD, REM>::getBits(indexCalc(locIndex, Rindex[0]), indexCalc(locIndex, Rindex[1]));
    }

    GPUHEADER
    void setH(int x, int locIndex, bool onDevice = true) {
        TableEntry<ADD, REM>::setBits(indexCalc(locIndex, Hindex[0]), indexCalc(locIndex, Hindex[1]), x, onDevice);
        return;
    }

    GPUHEADER
    int getH( int locIndex) {
        return (int) TableEntry<ADD, REM>::getBits(indexCalc(locIndex, Hindex[0]), indexCalc(locIndex, Hindex[1]));
    }

    GPUHEADER
    void setO(bool x, int locIndex, bool onDevice = true) {
        TableEntry<ADD, REM>::setBits(indexCalc(locIndex, Oindex[0]), indexCalc(locIndex, Oindex[1]), x, onDevice);
        return;
    }

    GPUHEADER
    bool getO( int locIndex) {
        return (bool)TableEntry<ADD, REM>::getBits(indexCalc(locIndex, Oindex[0]), indexCalc(locIndex, Oindex[1]));
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
