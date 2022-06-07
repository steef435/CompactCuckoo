#include <utility>
#include <iostream>
#include <bitset>
#include <inttypes.h>


template <class ADD, class REM>
class TableEntry {

protected:
    uint64_cu val;

    __host__ __device__
    void setBits(int start, int end, uint64_cu ins, bool onDevice=true) {
        setBits(start, end, ins, &val, onDevice);
    }

    __host__ __device__
    void setBits(int start, int end, uint64_cu ins, uint64_cu* loc, bool onDevice = true) {
    while( true ){
      uint64_cu mask = ((((uint64_cu)1) << end) - 1) ^ ((((uint64_cu)1) << (start - 1)) - 1);
      uint64_cu oldval = *loc;
      uint64_cu tempval = oldval & ~mask;      //Remove all of the bits currently in the positions
      ins = ins << (start - 1);   //Shift new val to correct position
      ins = ins & mask;       //Mask the new val to prevent overflow
      uint64_cu newval = tempval | ins;        //Place the new val
      //In devices, atomically exchange
      #ifdef  __CUDA_ARCH__
      if (onDevice) {
          printf("\t\t\t\t\t\t\t\t\t\t%i: Trying to write\n", threadIdx.x);
          uint64_cu res = atomicCAS(loc, oldval, newval);
          //Make sure the value hasn't changed in the meantime
          if(res == oldval){
            return;
          }
      }
      else {
          *loc = newval;
          return;
      }
      #else
          *loc = newval;
          return;
      #endif
      }
    }

    __host__ __device__
        uint64_cu getBits(int start, int end, uint64_cu x) {
        uint64_cu res = x;
        uint64_cu mask = ((((uint64_cu)1) << end) - ((uint64_cu)1)) ^ ((((uint64_cu)1) << (start - 1)) - ((uint64_cu)1));
        res = res & mask;
        res = res >> (start - 1);
        return res;
    }

    __host__ __device__
    uint64_cu getBits(int start, int end) {
        return getBits(start, end, val);
    }

    __host__ __device__
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

    __host__ __device__
    int unsigned_to_signed(uint64_cu n, int size)
    {
        uint64_cu mask = ((((uint64_cu)1) << size) - 1) ^ ((((uint64_cu)1) << (1 - 1)) - 1);
        int res = n & mask;
        if (n >> size == 1) {
            res = -res;
        }
        return res;
    }

    __host__ __device__
    uint64_cu getValue() {
        return val;
    }

    __host__ __device__
    void setValue(uint64_cu x) {
        val = x;
    }

    __host__ __device__
    uint64_cu* getValPtr(){
      return &val;
    }

public:
    __host__ __device__
    TableEntry() {
        val = 0;
    }

    __host__ __device__
        TableEntry(uint64_cu x) {
        val = x;
    }

    __device__
    void exchValue(TableEntry* x) {
        //Atomically set this value to the new one
        uint64_cu old =  atomicExch(&val, x->getValue());
        //Return an entry with prev val
        x->setValue(old);
        return;
    }

    __host__ __device__
    void print() {
        //std::cout << std::bitset<64>(val) << "\n";
        printf("EntryVal:%" PRIu64 "\n", val);
    }

};
