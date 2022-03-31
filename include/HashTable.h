#include <cstdint>
#include <list>

/**
 * General HashTable class to inherit from
 **/
class HashTable{
    public:
        /**
         * Constructor Method
         * Input:
         *      tablesize   : How large the table should be
         *      elmtsize    : How large the elements are
         * Output:
         *      new instance of a ClearyCuckoo table
         **/
        HashTable(){};

        virtual ~HashTable(){};

        /**
         *  Insert Method
         *  Input:
         *      Key k
         *  Output:
         *      Key k is inserted
         **/
        virtual bool insert(uint64_t k);

        /**
         *  Lookup Method
         *  Input:
         *      Key k
         *  Output:
         *      Object stored at key k
         **/
        virtual bool lookup(uint64_t k);

        /**
         *  Rehash
         **/
        virtual bool rehash();

        /**
         * Method to clear all values in the table
         **/
        virtual void clear();

        virtual int getSize();
        
        /**
         * Method to print
         * (Mostly for debugging/testing)
         * */
        virtual void print();
};
