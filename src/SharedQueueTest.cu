#ifndef SHAREDQUEUE
#define SHAREDQUEUE
#include "SharedQueue.cu"
#endif

GPUHEADER_G
void fillQueue(int N, SharedQueue<int>* queue, int id = 0, int s = 1)
{
#ifdef GPUCODE
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x;
#else
    int index = id;
    int stride = s;
#endif
    //std::cout << "ThreadID: " << GetCurrentProcessorNumber() << "\n";
    for (int i = index; i < N; i += stride) {
        queue->push(i);
        printf("Inserted: %i\n", i);
    }
}

GPUHEADER_G
void checkQueue(int N, SharedQueue<int>* queue, int id = 0, int s = 1)
{
#ifdef GPUCODE
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x;
#else
    int index = id;
    int stride = s;
#endif
    //std::cout << "ThreadID: " << GetCurrentProcessorNumber() << "\n";
    for (int i = index; i < N; i += stride) {
        int val = queue->pop();
        printf("Popped: %i\n", val);
    }
}

GPUHEADER_G
void fillPopQueue(int N, SharedQueue<int>* queue, int id = 0, int s = 1)
{
#ifdef GPUCODE
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x;
#else
    int index = id;
    int stride = s;
#endif
    for (int i = index; i < N; i += stride) {
        if (i % 2 == 0) {
            queue->push(i/2);
            printf("Inserted: %i\n", i/2);
        }
        else {
            int val = queue->pop();
            printf("Popped: %i\n", val);
        }
    }

}

void queueTest(int maxSize, int numThreads) {
	
#ifdef GPUCODE
    SharedQueue<int>* Q;
    gpuErrchk(cudaMallocManaged((void**)&Q, sizeof(SharedQueue<int>)));
    new (Q) SharedQueue<int>(maxSize);
#else
    SharedQueue<int>* Q = new SharedQueue<int>(maxSize);
#endif

    int N = maxSize;

    printf("===========================SimpleFillTest===========================\n");
    printf("Filling\n");
#ifdef GPUCODE
    fillQueue << <1, numThreads >> > (N, Q);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#else
    std::vector<std::thread> vecThread(numThreads);

    for (int i = 0; i < numThreads; i++) {
        vecThread.at(i) = std::thread(fillQueue, N, Q, i, numThreads);
    }

    //Join Threads
    for (int i = 0; i < numThreads; i++) {
        vecThread.at(i).join();
    }
#endif

    printf("Popping\n");

#ifdef GPUCODE
    checkQueue << <1, numThreads >> > (N, Q);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#else
    std::vector<std::thread> vecThread2(numThreads);

    for (int i = 0; i < numThreads; i++) {
        vecThread2.at(i) = std::thread(checkQueue, N, Q, i, numThreads);
    }

    //Join Threads
    for (int i = 0; i < numThreads; i++) {
        vecThread2.at(i).join();
    }
#endif

    printf("===========================MixedFillTest===========================\n");
#ifdef GPUCODE
    SharedQueue<int>* Q2;
    gpuErrchk(cudaMallocManaged((void**)&Q2, sizeof(SharedQueue<int>)));
    new (Q2) SharedQueue<int>(maxSize);
#else
    SharedQueue<int>* Q2 = new SharedQueue<int>(maxSize);
#endif

#ifdef GPUCODE
    fillPopQueue << <1, numThreads >> > (3*N, Q2);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#else
    std::vector<std::thread> vecThread3(numThreads);

    for (int i = 0; i < numThreads; i++) {
        vecThread3.at(i) = std::thread(fillPopQueue, 3*N, Q2, i, numThreads);
    }

    //Join Threads
    for (int i = 0; i < numThreads; i++) {
        vecThread3.at(i).join();
    }
#endif

    printf("Test Done\n");
}

