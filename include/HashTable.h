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
        HashTable() = default;

        virtual ~HashTable() = default;

        /**
         *  Insert Method
         *  Input:
         *      Key k
         *  Output:
         *      Key k is inserted
         **/
        __host__ __device__
        virtual bool HashTable::insert(uint64_t k) = 0;

        /**
         *  Lookup Method
         *  Input:
         *      Key k
         *  Output:
         *      Object stored at key k
         **/
        __host__ __device__
        virtual bool HashTable::lookup(uint64_t k) = 0;

        /**
         *  Rehash
         **/
        __host__ __device__
        virtual bool HashTable::rehash() = 0;

        /**
         * Method to clear all values in the table
         **/
        __host__ __device__
        virtual void HashTable::clear() = 0;

        /*
         * Method to get the size of the table
         **/
        __host__ __device__
        virtual int HashTable::getSize() = 0;

        
        /**
         * Method to print
         * (Mostly for debugging/testing)
         * */
        __host__ __device__
        virtual void HashTable::print() = 0;
};
