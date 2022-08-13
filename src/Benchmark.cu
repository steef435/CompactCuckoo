#ifndef NUMGEN
#define NUMGEN
#include "numbergenerators.cu"
#endif

void readList(uint64_cu* xs, int N, int numLoops, int T = 1, int id = 0) {
    //printf("Reading List\n");
    int begin = 0;
    int end = N;

    if (T > 1) {
        int spread = N / T;
        begin = spread * id;
        end = begin + spread - 1;
        //printf("Begin: %i End:%i", begin, end);
    }

    for (int i = 0; i < numLoops; i++) {
        //printf("Reading List i:%i\n",i);
        for (int j = begin; j < end; j++) {
            //printf("Reading List j:%i\n", j);
            uint64_cu val = xs[j];
        }
    }
}

void warmupThreads(int T, uint64_cu* xs, int N, int numLoops) {
    std::vector<std::thread> vecThread(T);
    for (int i = 0; i < T; i++) {
        vecThread.at(i) = std::thread(readList, xs, N, numLoops, T, i);
    }

    //Join Threads
    for (int i = 0; i < T; i++) {
        vecThread.at(i).join();
    }
}

void BenchmarkFilling(int NUM_TABLES_start, int NUM_TABLES, int INTERVAL, int NUM_SAMPLES, int NUM_THREADS, int NUM_LOOPS, int NUM_HASHES, int PERCENTAGE, int P_STEPSIZE, int DEPTH, std::vector<std::string>* params = nullptr) {

    const int WARMUP = 2;

    printf("=====================================================================\n");
    printf("                     Starting INSERTION BENCHMARK                    \n");
    printf("=====================================================================\n");

    std::ofstream myfile;
    std::string filename = "results/benchfill.csv";

    if (params) {
        printf("Opening\n");
        myfile.open(filename, std::ios_base::app);
        printf("Maybe\n");
    }
    else {
        myfile.open(filename);
    }

    if (!myfile.is_open()) {
        printf("File Failed to Open\n");

        return;
    }
    printf("File Opened\n");

    if (!params) {
        myfile << "tablesize,numthreads,loops,hashes,collision_percentage,samples,type,interval,time\n";
    }

    printf("=====================================================================\n");
    printf("                     Starting Cleary-Cuckoo                \n\n");

    int NUM_GEN_THREADS = 256;

    //Tablesizes
    bool setup = true;
    for (int N = NUM_TABLES_start; N < NUM_TABLES_start + NUM_TABLES; N++) {
        if (params && setup) {
            N = std::stoi(params->at(0));
        }
        printf("Table Size:%i\n", N);

        int size = std::pow(2, N);
        int setsize = (int)(size / INTERVAL);

        if (setsize == 0) {
            printf("Error: Number of Intervals is greater than number of elements\n");
        }

        //Number of Threads
        for (int T = 0; T < NUM_THREADS; T++) {
            if (params && setup) {
                T = std::stoi(params->at(1));
            }
            printf("\tNumber of Threads:%i\n", T);
            //Number of Loops
            for (int L = 0; L < NUM_LOOPS; L++) {
#ifdef GPUCODE
                int numThreads = std::pow(2, T);
#else
                int numThreads = T + 1;
#endif

                if (params && setup) {
                    L = std::stoi(params->at(2));
                }
                printf("\t\tNumber of Loops:%i\n", L);
                //Number of Hashes
                for (int H = 1; H < NUM_HASHES; H++) {
                    printf("\t\t\tNumber of Hashes:%i\n", H);
                    if (params && setup) {
                        H = std::stoi(params->at(3));
                    }

                    for (int P = 0; P < PERCENTAGE; P += P_STEPSIZE) {
                        printf("\t\t\t\tPercentage:%i\n", P);
                        for (int D = 0; D < DEPTH; D += 1) {
                            //Number of samples
                            for (int S = 0; S < NUM_SAMPLES; S++) {
                                printf("\t\t\t\t\tSample:%i\n", S);
                                if (params && setup) {
                                    S = std::stoi(params->at(4));
                                }
                                setup = false;
                                //Init Cleary Cuckoo

#ifdef GPUCODE
                                ClearyCuckoo* cc;
                                gpuErrchk(cudaMallocManaged((void**)&cc, sizeof(ClearyCuckoo)));
                                new (cc) ClearyCuckoo(N, H);
#else
                                ClearyCuckoo* cc = new ClearyCuckoo(N, H);
#endif

                                cc->setMaxLoops(L);
                                int* hs = cc->getHashlist();
                                uint64_cu* vals = generateCollisionSet(size, N, H, hs, P, D);
                                delete[] hs;

                                //Warmup
                                readList(vals, size, 20);
                                cc->readEverything(size * 50);
                                //warmupThreads(numThreads, vals, size, 20);

                                //Loop over intervals
                                for (int j = 0; j < INTERVAL + WARMUP; j++) {
                                    //Fill the table
                                    std::chrono::steady_clock::time_point begin;
                                    std::chrono::steady_clock::time_point end;

                                    if (j < WARMUP) {
                                        //cc->readEverything(20);
                                    }

                                    if (j >= WARMUP) {
                                        begin = std::chrono::steady_clock::now();
#ifdef GPUCODE                  
                                        fillClearyCuckoo << <1, std::pow(2, T) >> > (setsize, vals, cc, setsize * (j - WARMUP));
                                        gpuErrchk(cudaPeekAtLastError());
                                        gpuErrchk(cudaDeviceSynchronize());
#else
                                        std::vector<std::thread> vecThread(numThreads);
                                        for (int i = 0; i < numThreads; i++) {
                                            vecThread.at(i) = std::thread(static_cast<void(*)(int, uint64_cu*, ClearyCuckoo*, addtype, int, int)>(fillClearyCuckoo), setsize, vals, cc, setsize * (j - WARMUP), i, numThreads);
                                        }

                                        //Join Threads
                                        for (int i = 0; i < numThreads; i++) {
                                            vecThread.at(i).join();
                                        }
#endif
                                        //End the timer
                                        end = std::chrono::steady_clock::now();

                                        myfile << N << "," << numThreads << "," << L << "," << H << "," << P << "," << S << ",cuc," << (j - WARMUP) << "," << (std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count()) / setsize << ",\n";
                                    }

                                }
#ifdef GPUCODE
                                gpuErrchk(cudaFree(cc));
                                gpuErrchk(cudaFree(vals));
#else       
                                delete cc;
                                delete vals;
#endif
                            }
                        }
                    }
                }
            }
        }
    }

    printf("=====================================================================\n");
    printf("                     Starting Cleary                \n\n");

    for (int N = NUM_TABLES_start; N < NUM_TABLES_start + NUM_TABLES; N++) {
        if (params && setup) {
            N = std::stoi(params->at(0));
        }
        printf("Table Size:%i\n", N);

        int size = std::pow(2, N);
        int setsize = (int)(size / INTERVAL);
        for (int T = 0; T < NUM_THREADS; T++) {
            printf("\tNumber of Threads:%i\n", T);
#ifdef GPUCODE
            int numThreads = std::pow(2, T);
#else
            int numThreads = T + 1;
#endif
            for (int S = 0; S < NUM_SAMPLES; S++) {
                printf("\t\t\t\tSample Number:%i\n", S);
                //uint64_cu* vals = generateTestSetParallel(size, NUM_GEN_THREADS);
                uint64_cu* vals = generateTestSet(size);

                //Init Cleary
#ifdef GPUCODE
                Cleary* c;
                gpuErrchk(cudaMallocManaged((void**)&c, sizeof(Cleary)));
                new (c) Cleary(N);
#else
                Cleary* c = new Cleary(N);
#endif

                //Loop over intervals
                for (int j = 0; j < INTERVAL + WARMUP; j++) {
                    std::chrono::steady_clock::time_point begin;
                    std::chrono::steady_clock::time_point end;

                    //Fill the table
                    if (j >= WARMUP) {
                        begin = std::chrono::steady_clock::now();
#ifdef GPUCODE

                        fillCleary << <1, numThreads >> > (setsize, vals, c, setsize * (j - WARMUP));
                        gpuErrchk(cudaPeekAtLastError());
                        gpuErrchk(cudaDeviceSynchronize());
#else

                        std::vector<std::thread> vecThread(numThreads);
                        //uint64_cu** valCopy = (uint64_cu**) malloc(sizeof(uint64_cu*) * numThreads);

                        for (int i = 0; i < numThreads; i++) {
                            //valCopy[i] = (uint64_cu*) malloc(sizeof(uint64_cu) * size);
                            //copyArray(vals, valCopy[i], size);
                            vecThread.at(i) = std::thread(fillCleary, setsize, vals, c, setsize * (j - WARMUP), i, numThreads);
                        }

                        //Join Threads
                        for (int i = 0; i < numThreads; i++) {
                            vecThread.at(i).join();
                            //delete[] valCopy[i];
                        }
#endif
                        //End the timer
                        end = std::chrono::steady_clock::now();
                        myfile << N << "," << numThreads << "," << -1 << "," << -1 << "," << S << ",cle," << (j - WARMUP) << "," << (std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count()) / setsize << ",\n";
                    }

                }
#ifdef GPUCODE
                gpuErrchk(cudaFree(c));
                gpuErrchk(cudaFree(vals));
#else       
                delete c;
                delete vals;
#endif
            }
        }
    }

    myfile.close();
    printf("\nBenchmark Done\n");
}

void BenchmarkMaxOccupancy(int TABLESIZES, int NUM_HASHES, int NUM_LOOPS, int NUM_SAMPLES) {

    printf("=====================================================================\n");
    printf("                   Starting MAX Occupancy Benchmark                  \n");
    printf("=====================================================================\n");

    std::ofstream myfile;
    std::string filename = "results/benchmax.csv";
    myfile.open(filename);
    if (!myfile.is_open()) {
        printf("File Failed to Open\n");
        return;
    }
    printf("File Opened");

    myfile << "tablesize,numhashes,numloops,samples,max\n";

    //MAX_LOOPS
    for (int N = 5; N < 5 + TABLESIZES; N++) {
        printf("Table Size:%i\n", N);
        int size = std::pow(2, N);
        for (int j = 1; j < NUM_HASHES; j++) {
            printf("\tNum of Hashes:%i\n", j);
            for (int k = 0; k < NUM_LOOPS; k++) {
                printf("\t\tNum of Loops:%i\n", k);
                for (int S = 0; S < NUM_SAMPLES; S++) {
                    //printf("\t\t'tSample Number:%i\n", S);
                    uint64_cu* vals = generateTestSet(size);

                    int* failFlag;
                    gpuErrchk(cudaMallocManaged(&failFlag, sizeof(int)));
                    failFlag[0] = false;

                    //Init Cleary Cuckoo
                    ClearyCuckoo* cc;
                    gpuErrchk(cudaMallocManaged((void**)&cc, sizeof(ClearyCuckoo)));
                    new (cc) ClearyCuckoo(N, j);
                    cc->setMaxLoops(k);

                    //Var to store num of inserted values
                    addtype* occ;
                    gpuErrchk(cudaMallocManaged(&occ, sizeof(addtype)));
                    occ[0] = 0;

                    //Fill the table
#ifdef GPUCODE
                    fillClearyCuckoo << <1, 256 >> > (size, vals, cc, occ, failFlag);
                    gpuErrchk(cudaPeekAtLastError());
                    gpuErrchk(cudaDeviceSynchronize());

                    myfile << N << "," << j << "," << k << "," << S << "," << occ[0] << ",\n";

                    gpuErrchk(cudaFree(failFlag));
                    gpuErrchk(cudaFree(cc));
                    gpuErrchk(cudaFree(occ));
                    gpuErrchk(cudaFree(vals));
#else
                    delete failFlag;
                    delete cc;
                    delete occ;
                    delete[] vals;
#endif
                }
            }
        }
    }

    myfile.close();

    printf("\n\nBenchmark Done\n");
}