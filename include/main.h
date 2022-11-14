#include <stdio.h>
#include <thread>
#include <sstream>
#include <assert.h>

/***
*
* TYPES
*
*/

typedef long long int int64_cu;
typedef unsigned long long int uint64_cu;

using addtype = uint32_t;
using remtype = uint64_cu;
using hashtype = uint64_cu;
using keytype = uint64_cu;

/**
*
* Define this var to switch between GPU and CPU execution
*
**/

//define GPUCODE
//#define REHASH

/**
*
* GPU Headers
*
*/


#ifdef GPUCODE
#define GPUHEADER __host__ __device__
#define GPUHEADER_G __global__
#define GPUHEADER_D __device__
#define GPUHEADER_H __host__
#else
#define GPUHEADER
#define GPUHEADER_G
#define GPUHEADER_D
#define GPUHEADER_H
#endif

/**
*
* GPU Debug Methods
*
*/

#ifndef GPUASSERT
#define GPUASSERT
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
GPUHEADER
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      //if (abort) exit(code);
   }
}
#endif

#ifndef GETID
#define GETID
GPUHEADER_D
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


/**
*
* OTHER HELPER METHODS
*
*/

GPUHEADER
addtype getAdd(keytype key, int AS) {
    hashtype mask = ((hashtype)1 << AS) - 1;
    addtype add = key & mask;
    return add;
}

GPUHEADER
remtype getRem(keytype key, int AS) {
    remtype rem = key >> AS;
    return rem;
}

GPUHEADER
uint64_cu reformKey(addtype add, remtype rem, int AS) {
    rem = rem << AS;
    rem += add;
    return rem;
}

GPUHEADER
void boundaryAssert(int i, addtype min, addtype max) {
    assert(min <= i && i <= max);
}
