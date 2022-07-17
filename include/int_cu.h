#include <stdio.h>
#include <thread>
#include <sstream>

typedef long long int int64_cu;
typedef unsigned long long int uint64_cu;

#ifndef GPUASSERT
#define GPUASSERT
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#endif

//#define GPUCODE

#ifdef GPUCODE
    #define GPUHEADER __host__ __device__
    #define GPUHEADER_G __global__
    #define GPUHEADER_D __device__
#else
    #define GPUHEADER
    #define GPUHEADER_G
#define GPUHEADER_D
#endif

#ifndef GETID
#define GETID
inline int getThreadID()
{
#ifdef GPUCODE
    return threadIdx.x;
#else
    std::stringstream ss;
    ss << std::this_thread::get_id();
    uint64_t id = std::stoull(ss.str());
    return id;
#endif
}
#endif