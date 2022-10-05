#include <atomic>

#ifndef MAIN
#define MAIN
#include "main.h"
#endif

template <class T>
class SharedQueue {
private:
    T* queue;
    bool* Occ;

    int maxSize;
#ifdef GPUCODE
    int insertIndex = 0;
    int headIndex = 0;
#else
    std::atomic<int> insertIndex;
    std::atomic<int> headIndex;
#endif


public:
    SharedQueue(int max) {
        //printf("Making SharedQueue size %i \n", max);
        maxSize = max;

#ifdef GPUCODE
        insertIndex = 0;
        headIndex = 0;
#else
        std::atomic_init(&insertIndex, 0);
        std::atomic_init(&headIndex, 0);
#endif

#ifdef GPUCODE
        gpuErrchk(cudaMallocManaged(&queue, maxSize * sizeof(T)));
        gpuErrchk(cudaMallocManaged(&Occ, maxSize * sizeof(bool)));
#else
        queue = new T[maxSize];
        Occ = new bool[maxSize];
#endif

        for (int i = 0; i < maxSize; i++) {
            queue[i] = 0;
            Occ[i] = false;
        }
    }

    ~SharedQueue() {
#ifdef GPUCODE
        cudaFree(queue);
        cudaFree(Occ);
#else
        delete[] queue;
        delete[] Occ;
#endif
    }

    GPUHEADER_D
    void push(T val) {
#ifdef GPUCODE
        int index = atomicAdd(&insertIndex, 1) % maxSize;
#else
        int index = insertIndex.fetch_add(1) % maxSize;
#endif
        //printf("\tPushIndex %i\n", index);
        if (!Occ[index]) {
            queue[index] = val;
            Occ[index] = true;
        }
        else {
            printf("WARNING: Value in queue was overwritten\n");
        }
        //print();
    }

    GPUHEADER_D
    T pop() {
        if (!isEmpty()) {
            //Increment and clear
#ifdef GPUCODE
            int index = atomicAdd(&headIndex, 1) % maxSize;
#else
            int index = headIndex.fetch_add(1) % maxSize;
#endif
            Occ[index] = false;
            return queue[index];
        }
        //print();
        //Otherwise return a default value
        return 0;
    }

    GPUHEADER_D
    bool isEmpty() {
#ifdef GPUCODE
        //printf("IsEmptyIndex: %i, headIndex %i, maxSize %i\n", headIndex % maxSize, headIndex, maxSize);
        return !Occ[headIndex % maxSize];
#else
        //printf("\tChecking %i", headIndex.load() % maxSize);
        return !Occ[headIndex.load() % maxSize];
#endif
    }

    GPUHEADER
    void print() {
#ifdef GPUCODE
        printf("insertIndex: %i  headIndex: %i", insertIndex, headIndex);
#else
        printf("insertIndex: %i  headIndex: %i\n", insertIndex.load(), headIndex.load());
#endif
        printf("----------------------------------------\n");
        printf("|    i     |  Occ |         val        |\n");
        printf("----------------------------------------\n");
        for (int i = 0; i < maxSize; i++) {
                printf("|%-10i|%6i||%-20" PRIu64 "|\n", i, Occ[i], queue[i]);
        }
        printf("------------------------------------------------------------\n");
    }

    GPUHEADER_H
    std::vector<uint64_cu> toVector() {
        std::vector<uint64_cu> list;
        for (int i = 0; i < maxSize; i++) {
            if (Occ[i]) {
                list.push_back(queue[i]);
            }
        }
        return list;
    }
};
