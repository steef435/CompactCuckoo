#include "TableEntry.h"

template <class ADD, class REM>
class ClearyEntry : TableEntry <ADD, REM>{

private:
    std::pair<int, int> Rindex = std::pair<int, int>(1, 56);
    std::pair<int, int> Oindex = std::pair<int, int>(57, 57);
    std::pair<int, int> Vindex = std::pair<int, int>(58, 58);
    std::pair<int, int> Cindex = std::pair<int, int>(59, 59);
    std::pair<int, int> Aindex = std::pair<int, int>(60, 63);


public:
    ClearyEntry(ADD R, bool O, bool V, bool C, int A) {
        val = 0;
        setR(R);
        setO(O);
        setV(V);
        setC(C);
        setA(A);
    }

    ClearyEntry() {
        ClearyEntry(0, false, false, true, 0);
    }

    void setR(REM x) {
        setBits(Rindex.first, Rindex.second, x);
    }

    REM getR() {
        return (REM)getBits(Rindex.first, Rindex.second);
    }

    void setO(bool x) {
        setBits(Oindex.first, Oindex.second, x);
    }

    bool getO() {
        return (bool)getBits(Oindex.first, Oindex.second);
    }

    void setV(bool x) {
        setBits(Vindex.first, Vindex.second, x);
    }

    bool getV() {
        return (bool)getBits(Vindex.first, Vindex.second);
    }

    void setC(bool x) {
        setBits(Cindex.first, Cindex.second, x);
    }

    bool getC() {
        return (bool)getBits(Cindex.first, Cindex.second);
    }

    void setA(int x) {
        setBits(Aindex.first, Aindex.second, signed_to_unsigned(x, 3));
    }

    int getA() {
        return unsigned_to_signed(getBits(Aindex.first, Aindex.second), 3);
    }

    void print() {
        std::cout << std::bitset<64>(val) << "\n";
    }

};