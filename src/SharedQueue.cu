#include <atomic>

#ifndef MAIN
#define MAIN
#include "main.h"
#endif

template <class T>
class SharedQueue {
private:
    T* queue;
    int maxSize;
#ifdef GPUCODE
    int insertIndex = 0;
    int headIndex = 0;
#else
    std::atomic<int> insertIndex;
    std::atomic<int> headIndex;
#endif


public:
    SharedQueue(int maxSize) {
        this->maxSize = maxSize;

#ifdef GPUCODE
        insertIndex = 0;
        headIndex = 0;
#else
        std::atomic_init(&insertIndex, 0);
        std::atomic_init(&headIndex, 0);
#endif

#ifdef GPUCODE
        gpuErrchk(cudaMallocManaged(&queue, maxSize * sizeof(T)));
#else
        queue = new T[maxSize];
#endif

        for (int i = 0; i < maxSize; i++) {
            queue[i] = -1;
        }
    }

    ~SharedQueue() {
#ifdef GPUCODE
        cudaFree(queue);
#else
        delete[] queue;
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
        if (queue[index] == -1) {
            queue[index] = val;
        }
        else {
            printf("WARNING: Value in queue was overwritten\n");
        }
    }

    GPUHEADER_D
    T pop() {
#ifdef GPUCODE
        int index = atomicAdd(&headIndex, 1) % maxSize;
#else
        int index = headIndex.fetch_add(1) % maxSize;
#endif
        //printf("\tPopIndex %i\n", index);
        T retVal = queue[index];
        //printf("retVal:%i", retVal);
        queue[index] = -1;
        return retVal;
    }
    /*
    bool isEmpty() {
        return insertIndex == headIndex;
    }*/
};