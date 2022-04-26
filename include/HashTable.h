#include <cstdint>
#include <list>

/**
 * General HashTable class to inherit from
 **/
class HashTable{
    public:
        /**
         * Constructor Method
         **/
        HashTable() {};

        ~HashTable() {};

        /**
         *  Insert Method
         *  Input:
         *      Key k
         *  Output:
         *      Key k is inserted
         **/
        __host__ __device__
        bool HashTable::insert(uint64_t k) { return false; };

        /**
         *  Lookup Method
         *  Input:
         *      Key k
         *  Output:
         *      Object stored at key k
         **/
        __host__ __device__
        bool HashTable::lookup(uint64_t k) { return false; };

        /**
         *  Rehash
         **/
        __host__ __device__
        bool HashTable::rehash() { return false; };

        /**
         * Method to clear all values in the table
         **/
        __host__ __device__
            void HashTable::clear() {};

        /*
         * Method to get the size of the table
         **/
        __host__ __device__
        int HashTable::getSize() { return 0; };

        
        /**
         * Method to print
         * (Mostly for debugging/testing)
         * */
        __host__ __device__
        void HashTable::print() {};
};
