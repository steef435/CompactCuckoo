#include <utility>
#include <iostream>
#include <bitset>
#include <inttypes.h>
#include <atomic>


template <class ADD, class REM>
class TableEntry {

protected:
    #ifdef GPUCODE
    uint64_cu val;
    #else
    std::atomic<uint64_cu> val;
    #endif

    GPUHEADER
    uint64_cu setValBits(int start, int end, uint64_cu ins, uint64_cu* loc) {
        uint64_cu mask = ((((uint64_cu)1) << end) - 1) ^ ((((uint64_cu)1) << (start - 1)) - 1);
        uint64_cu oldval = *loc;
        uint64_cu tempval = oldval & ~mask;      //Remove all of the bits currently in the positions
        ins = ins << (start - 1);   //Shift new val to correct position
        ins = ins & mask;       //Mask the new val to prevent overflow
        return tempval | ins;        //Place the new val
    }

    GPUHEADER
    uint64_cu setValBits(int start, int end, uint64_cu ins, std::atomic<uint64_cu>* loc) {
        uint64_cu mask = ((((uint64_cu)1) << end) - 1) ^ ((((uint64_cu)1) << (start - 1)) - 1);
        uint64_cu oldval = (*loc).load();
        uint64_cu tempval = oldval & ~mask;      //Remove all of the bits currently in the positions
        ins = ins << (start - 1);   //Shift new val to correct position
        ins = ins & mask;       //Mask the new val to prevent overflow
        return tempval | ins;        //Place the new val
    }

    GPUHEADER
    void setBits(int start, int end, uint64_cu ins, bool onDevice=true) {
#ifdef GPUCODE
        if(onDevice){
          setBitsDevice(start, end, ins, &val, onDevice);
          return;
        }
#endif
        setBits(start, end, ins, &val, onDevice);
    }


    GPUHEADER
    void setBits(int start, int end, uint64_cu ins, std::atomic<uint64_cu>* loc, bool onDevice = true) {
        while (true) {

            uint64_cu oldval = (*loc).load();
            uint64_cu newval = setValBits(start, end, ins, loc);

            if (std::atomic_compare_exchange_strong(loc, &oldval, newval)) {
                break;
            }
        }
    }



#ifdef GPUCODE
    GPUHEADER_D
    void setBitsDevice(int start, int end, uint64_cu ins, uint64_cu* loc, bool onDevice = true) {
        while (true) {
            uint64_cu oldval = *loc;
            uint64_cu newval = setValBits(start, end, ins, loc);

            //In devices, atomically exchange
            uint64_cu res = atomicCAS(loc, oldval, newval);
            //Make sure the value hasn't changed in the meantime
            if (res == oldval) {
                return;
            }
            continue;
        }
    }

    GPUHEADER
    void setBits(int start, int end, uint64_cu ins, uint64_cu* loc, bool onDevice = true) {
        uint64_cu newval = setValBits(start, end, ins, loc);

        *loc = newval;
        return;
    }
#else
    GPUHEADER
    void setBits(int start, int end, uint64_cu ins, uint64_cu* loc, bool onDevice = true) {
            uint64_cu newval = setValBits(start, end, ins, loc);
            *loc = newval;
            return;
    }
#endif

    GPUHEADER
    uint64_cu getBits(int start, int end, uint64_cu* x) {
        if(start == -1){
          return 0;
        }
        uint64_cu res = (*x);
        uint64_cu mask = ((((uint64_cu)1) << end) - ((uint64_cu)1)) ^ ((((uint64_cu)1) << (start - 1)) - ((uint64_cu)1));
        res = res & mask;
        res = res >> (start - 1);
        return res;
    }

    GPUHEADER
        uint64_cu getBits(int start, int end, std::atomic<uint64_cu>* x) {
        if (start == -1) {
            return 0;
        }
        uint64_cu res = (*x).load();
        uint64_cu mask = ((((uint64_cu)1) << end) - ((uint64_cu)1)) ^ ((((uint64_cu)1) << (start - 1)) - ((uint64_cu)1));
        res = res & mask;
        res = res >> (start - 1);
        return res;
    }

    GPUHEADER
    uint64_cu getBits(int start, int end) {
        return getBits(start, end, &val);
    }

    GPUHEADER
    int signed_to_unsigned(int n, int size) {
        int res = 0;
        if (n < 0) {
            res = 1 << size;
            res = res | (-n);
            return res;
        }
        res = res | n;
        return res;
    }

    GPUHEADER
    int unsigned_to_signed(uint64_cu n, int size)
    {
        uint64_cu mask = ((((uint64_cu)1) << size) - 1) ^ ((((uint64_cu)1) << (1 - 1)) - 1);
        int res = n & mask;
        if (n >> size == 1) {
            res = -res;
        }
        return res;
    }

    GPUHEADER
    uint64_cu* getValPtr(){
#ifdef GPUCODE
        return &val;
#else
        return nullptr;
#endif
    }

#ifndef GPUCODE
    __host__ __device__
    std::atomic<uint64_cu>* getAtomValPtr() {
        return &val;
    }
#endif

public:
    GPUHEADER
    TableEntry() noexcept {
#ifdef GPUCODE
        val = 0;
#else
        std::atomic_init(&val, 0);
#endif

    }

    GPUHEADER
    TableEntry(uint64_cu x) noexcept {
#ifdef GPUCODE
        val = x;
#else
        std::atomic_init(&val, x);
#endif
    }

    GPUHEADER
    uint64_cu getValue() {
#ifdef GPUCODE
        return val;
#else
        return val.load();
#endif

    }

    GPUHEADER
    void setValue(uint64_cu x) {
#ifdef GPUCODE
        val = x;
#else
        val.store(x);
#endif
    }

    GPUHEADER_D
    void exchValue(TableEntry* x) {
        //Atomically set this value to the new one
        #ifdef  GPUCODE
        uint64_cu old =  atomicExch(&val, x->getValue());
        #else
        int64_cu old = val.load();
        val.exchange(x->getValue());
        #endif
        x->setValue(old);
        return;
    }

    GPUHEADER
    void print() {
        //std::cout << std::bitset<64>(val) << "\n";
        printf("EntryVal:%" PRIu64 "\n", val);
    }

};
