#include "TableEntry.h"

template <class ADD, class REM>
class ClearyEntry : TableEntry<ADD, REM> {

private:
    uint64_t val;
    std::pair<int, int> Rindex = std::pair<int, int>(1, 56);
    std::pair<int, int> Hindex = std::pair<int, int>(57, 62);
    std::pair<int, int> Oindex = std::pair<int, int>(63, 64);

    void setBits(int start, int end, uint64_t ins) {
        uint64_t mask = ((((uint64_t)1) << end) - 1) ^ ((((uint64_t)1) << (start - 1)) - 1);
        val = val & ~mask;      //Remove all of the bits currently in the positions
        ins = ins << (start - 1);   //Shift new val to correct position
        ins = ins & mask;       //Mask the new val to prevent overflow
        val = val | ins;        //Place the new val
    }

    uint64_t getBits(int start, int end) {
        uint64_t res = val;
        uint64_t mask = ((((uint64_t)1) << end) - ((uint64_t)1)) ^ ((((uint64_t)1) << (start - 1)) - ((uint64_t)1));
        res = res & mask;
        res = res >> (start - 1);
        return res;
    }

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

    int unsigned_to_signed(unsigned n, int size)
    {
        uint64_t mask = ((((uint64_t)1) << size) - 1) ^ ((((uint64_t)1) << (1 - 1)) - 1);
        int res = n & mask;
        if (n >> size == 1) {
            res = -res;
        }
        return res;
    }

public:
    ClearyEntry(ADD R, int H, bool O) {
        val = 0;
        setR(R);
        setH(H);
        setO(O);
    }

    ClearyEntry() {
        ClearyEntry(0, 0, false);
    }

    void setR(REM x) {
        setBits(Rindex.first, Rindex.second, x);
    }

    REM getR() {
        return (REM)getBits(Rindex.first, Rindex.second);
    }

    void setH(int x) {
        setBits(Hindex.first, Hindex.second, x);
    }

    int getH() {
        return (int) getBits(Hindex.first, Hindex.second, x);
    }

    void setO(bool x) {
        setBits(Oindex.first, Oindex.second, x);
    }

    bool getO() {
        return (bool)getBits(Oindex.first, Oindex.second);
    }

    void print() {
        std::cout << std::bitset<64>(val) << "\n";
    }

};