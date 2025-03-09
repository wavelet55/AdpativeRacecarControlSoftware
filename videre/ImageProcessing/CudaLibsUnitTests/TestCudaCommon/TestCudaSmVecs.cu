//
// Created by wavelet on 11/18/16.
//

#include "TestCudaSmVecs.h"
#include "../../CudaLibs/CudaCommon/Utils/helper_cuda_timer.h"
#include "../../CudaLibs/CudaCommon/Utils/exception_cuda.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define BLOCK_SIZE 16

TestCudaSmVecs::TestCudaSmVecs()
{

}

TestCudaSmVecs::~TestCudaSmVecs()
{

}

int TestCudaSmVecs::TestSmVecs()
{
    int numberOfErrors = 0;
    int vecSize = 3;
    int nrows = 256;
    int ncols = 256;
    float vecA[3] = {100.234, 200.456, 300.789};
    float vecB[3] = {500.234, 600.456, 700.789};
    float *dVecA;
    float *dVecB;
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    StopWatchInterface *timerCuda = NULL;
    StopWatchInterface *timerCpp = NULL;
    float elapsedTimeInMs = 0.0f;
    cudaEvent_t start, stop;

    size_t vecMemSize = vecSize * sizeof(float);
    size_t resultVecSize = nrows * ncols * sizeof(float);
    float *host_rtnMtx = (float*)malloc(resultVecSize);
    float *cuda_rtnMtx = (float*)malloc(resultVecSize);
    float *device_rtnMtx;

    //Run Cpp Test
    printf( "Start Cpp Matrix Calculation.\n" );
    sdkCreateTimer(&timerCpp);
    sdkStartTimer(&timerCpp);

    testSmVecClassCpp(vecA, vecB, host_rtnMtx, vecSize, nrows, ncols);

    sdkStopTimer(&timerCpp);
    elapsedTimeInMs = sdkGetTimerValue(&timerCpp);
    printf( "Cpp Matrix Calculation Done. Time=%f\n",  elapsedTimeInMs);

    cudaMalloc(&dVecA, vecMemSize);
    cudaMalloc(&dVecB, vecMemSize);
    cudaMalloc(&device_rtnMtx, resultVecSize);
    cudaMemcpy(dVecA, vecA, vecMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dVecB, vecB, vecMemSize, cudaMemcpyHostToDevice);


    //Run the Cuda process
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(nrows / dimBlock.x, ncols / dimBlock.y);

    printf( "Start Cuda Matrix Calculation. \n" );
    sdkCreateTimer(&timerCuda);
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaEventRecord(start, 0));
    testSmVecClassKernel<<<dimGrid, dimBlock>>>(dVecA, dVecB, device_rtnMtx, vecSize, nrows, ncols);

    //Make sure Cuda has finish the calculations.
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timerCuda);
    checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));

    cudaMemcpy(cuda_rtnMtx, device_rtnMtx, resultVecSize, cudaMemcpyDeviceToHost);

    cudaFree(dVecA);
    cudaFree(dVecB);
    cudaFree(device_rtnMtx);

    printf( "Cuda Matrix Calculation Complete.  Time(msec)=%f\n",  elapsedTimeInMs);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();
    sdkDeleteTimer(&timerCpp);
    sdkDeleteTimer(&timerCuda);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    //Compare Results
    for(int i = 0; i < nrows * ncols; i++)
    {
        float delR = fabs( host_rtnMtx[i] - cuda_rtnMtx[i] );
        if( delR > 0.01 )
        {
            printf( "idx = %d, delR = %g\n", i, delR);
        }
    }

    printf("Done\n");

    return numberOfErrors;
}


//Std C++ Code for comparison
void testSmVecClassCpp(const float *vecAinp, const float *vecBinp, float *rtnMtx, int vecSize, int nrows, int ncols)
{
    for(int r = 0; r < nrows; r++)
        for(int c = 0; c < ncols; c++)
        {
            //Create Vectors to work with
            SmVec_i vecIdx(vecSize, 0);
            if( vecSize >= 2 )
            {
                vecIdx.val(0) = r;
                vecIdx.val(1) = c;
                if( vecSize > 2 )
                    vecIdx.val(2) = 10 * r + c;
            }

            SmVec_f vecA(vecSize, vecAinp);
            SmVec_f vecB(vecSize, vecBinp);
            //vecA = vecA + vecIdx.mk_float();
            //vecB = vecB + 2.5 * vecIdx.mk_float();

            SmVec_f vecC = 3.5 * vecA + 7.6 * vecB;
            float na = vecA.l2norm_sqrd();
            float nb = vecB.l1norm();
            float nc = vecC.l2norm_sqrd();
            float ipab = vecA.inner_prod(vecB) / nc;
            if( r < nrows && c < ncols )
                rtnMtx[r * ncols + c] = na + nb, + nc + ipab;
        }
}

//Exercise the Matrix Class in this kernal
//Cuda Kernal Code
__global__ void testSmVecClassKernel(const float* vecAinp, const float* vecBinp,
                                     float* rtnMtx, int vecSize, int nrows, int ncols)
{
    int r = blockDim.x * blockIdx.x + threadIdx.x;
    int c = blockDim.y * blockIdx.y + threadIdx.y;

    //Create Vectors to work with
    SmVecCuda_i vecIdx(vecSize, 0);
    if( vecSize >= 2 )
    {
        vecIdx.val(0) = r;
        vecIdx.val(1) = c;
        if( vecSize > 2 )
            vecIdx.val(2) = 10 * r + c;
    }

    SmVecCuda_f vecA(vecSize, vecAinp);

    SmVecCuda_f vecB(vecSize, vecBinp);

    //ToDo:  There are issues with CMake and running nvlink to properly link in
    //__device__ modules that are separately complied in .cu files.
    //vecA = vecA + vecIdx.mk_float();
    //vecB = vecB + (float)2.5 * vecIdx.mk_float();

    SmVecCuda_f vecC = 3.5 * vecA + 7.6 * vecB;
    float na = vecA.l2norm_sqrd();
    float nb = vecB.l1norm();
    float nc = vecC.l2norm_sqrd();
    float ipab = vecA.inner_prod(vecB) / nc;
    if( r < nrows && c < ncols )
        rtnMtx[r * ncols + c] = na + nb, + nc + ipab;
}
