#pragma once

#ifdef __CUDACC_X__
    #define HOSTDEVICE __host__ __device__
#else
    #define HOSTDEVICE
#endif
