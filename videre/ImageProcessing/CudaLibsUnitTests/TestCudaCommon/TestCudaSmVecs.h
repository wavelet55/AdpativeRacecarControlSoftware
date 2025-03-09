//
// Created by wavelet on 11/18/16.
//

#ifndef VIDERE_DEV_TESTCUDASMVECS_H
#define VIDERE_DEV_TESTCUDASMVECS_H

#include <cuda_runtime.h>
#include "../../CudaLibs/CudaCommon/Math/SmVectors_Cuda.h"
#include "../../CudaLibs/CudaCommon/Math/SmVectors.h"
#include "../../CudaLibs/CudaCommon/Utils/helper_cuda.h"

class TestCudaSmVecs
{

public:
    TestCudaSmVecs();

    ~TestCudaSmVecs();

    int TestSmVecs();


};


//Std C++ Code for comparison
//extern "C"
void testSmVecClassCpp(const float *vecAinp, const float *vecBinp,
                       float *rtnMtx, int vecSize, int nrows, int ncols);


//Exercise the Matrix Class in this kernal
//Cuda Kernal Code
//extern "C"
__global__ void testSmVecClassKernel(const float* vecAinp, const float* vecBinp,
                                     float* rtnMtx, int vecSize, int nrows, int ncols);


#endif //VIDERE_DEV_TESTCUDASMVECS_H
