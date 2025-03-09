/****************************************************************
Test Cuda Routines

H. Direen
Dec. 17, 2014

*****************************************************************/

#include "Matrix_Cuda.h"
#include "SmVectors_Cuda.h"
#include "CubicSpline_Cuda.h"
#include "NeuralNetwork_Cuda.h"
#include "..\Matrix.h"
#include "..\SmVectors.h"
#include "..\SUBR.H"
#include "..\PLANT_EQ.H"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>                           

#include <cuda_runtime.h>
// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#define BLOCK_SIZE 16

//Exercise the Matrix Class in this kernal
//Cuda Kernal Code
__global__ void testMatrixClassKernel(const float *vecAinp, const float *vecBinp, float *rtnMtx, int vecSize, int nrows, int ncols)
{
	int r = blockDim.x * blockIdx.x + threadIdx.x;
	int c = blockDim.y * blockIdx.y + threadIdx.y;

	//Create Vectors to work with
	matrixCuda_i vecIdx(vecSize, 1, 0);
	if( vecSize >= 2 )
	{
		vecIdx.val(0) = r;
		vecIdx.val(1) = c;
		if( vecSize > 2 )
			vecIdx.val(2) = 10 * r + c;
	}

	matrixCuda_f vecA(vecSize, vecAinp);
	matrixCuda_f vecB(vecSize, vecBinp);
	vecA = vecA + vecIdx.mk_float();
	vecB = vecB + 2.5 * vecIdx.mk_float();

	matrixCuda_f vecC = 3.5 * vecA + 7.6 * vecB;
	float na = vecA.l2norm_sqrd();
	float nb = vecB.l1norm();
	float nc = vecC.l2norm_sqrd();
	float ipab = vecA.inner_prod(vecB) / nc;
	if( r < nrows && c < ncols )
		rtnMtx[r * ncols + c] = na + nb, + nc + ipab;
}

//Std C++ Code for comparison
void testMatrixClassCpp(const float *vecAinp, const float *vecBinp, float *rtnMtx, int vecSize, int nrows, int ncols)
{
	for(int r = 0; r < nrows; r++)
		for(int c = 0; c < ncols; c++)
		{
			//Create Vectors to work with
			matrix_i vecIdx(vecSize, 1, 0);
			if( vecSize >= 2 )
			{
				vecIdx.val(0) = r;
				vecIdx.val(1) = c;
				if( vecSize > 2 )
					vecIdx.val(2) = 10 * r + c;
			}

			matrix_f vecA(vecSize, vecAinp);
			matrix_f vecB(vecSize, vecBinp);
			vecA = vecA + vecIdx.mk_float();
			vecB = vecB + 2.5 * vecIdx.mk_float();

			matrix_f vecC = 3.5 * vecA + 7.6 * vecB;
			float na = vecA.l2norm_sqrd();
			float nb = vecB.l1norm();
			float nc = vecC.l2norm_sqrd();
			float ipab = vecA.inner_prod(vecB) / nc;
			if( r < nrows && c < ncols )
				rtnMtx[r * ncols + c] = na + nb, + nc + ipab;
		}
}

void testMatrixClass()
{
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

	testMatrixClassCpp(vecA, vecB, host_rtnMtx, vecSize, nrows, ncols);

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
	testMatrixClassKernel<<<dimGrid, dimBlock>>>(dVecA, dVecB, device_rtnMtx, vecSize, nrows, ncols);

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
			vecA = vecA + vecIdx.mk_float();
			vecB = vecB + 2.5 * vecIdx.mk_float();

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
__global__ void testSmVecClassKernel(const float *vecAinp, const float *vecBinp, float *rtnMtx, int vecSize, int nrows, int ncols)
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
	vecA = vecA + vecIdx.mk_float();
	vecB = vecB + 2.5 * vecIdx.mk_float();

	SmVecCuda_f vecC = 3.5 * vecA + 7.6 * vecB;
	float na = vecA.l2norm_sqrd();
	float nb = vecB.l1norm();
	float nc = vecC.l2norm_sqrd();
	float ipab = vecA.inner_prod(vecB) / nc;
	if( r < nrows && c < ncols )
		rtnMtx[r * ncols + c] = na + nb, + nc + ipab;
}



//__device__ CubicSplineStruct CudaCubicSpline;
//
//__global__ void testCubicSplineKernel(float *rtnMtx, int nrows, int ncols)
//{
//	//Just Copy the Cubic Spline Parameters into the rtnMtx
//	int r = threadIdx.x;
//	if( r < nrows)
//	{
//		int j = 0;
//		rtnMtx[r * ncols + j++] = (float)CudaCubicSpline.N;
//		rtnMtx[r * ncols + j++] = (float)CudaCubicSpline.x_min;
//		rtnMtx[r * ncols + j++] = (float)CudaCubicSpline.x_max;
//		rtnMtx[r * ncols + j++] = (float)CudaCubicSpline.Extrapolation_Method;
//		//**************
//		for(int i = 0; i < 10; i++)
//		{
//			float4 *ptrRow = (float4*)((char*)(CudaCubicSpline.coef) + i * CudaCubicSpline.coefPitch);
//			rtnMtx[r * ncols + j++] = ptrRow->x;
//			rtnMtx[r * ncols + j++] = ptrRow->y;
//			rtnMtx[r * ncols + j++] = ptrRow->z;
//			rtnMtx[r * ncols + j++] = ptrRow->w;
//		}
//		//**********************/
//	}
//}


bool ERROR_CHECK(cudaError_t Status)
{
    if(Status != cudaSuccess)
    {
        printf(cudaGetErrorString(Status));
        return false;
    }   
    return true;
}


void testCubicSplineCPU(Cubic_Spline &csCpu, float *xSamples, float * ySamples, float *rtnMtx, int nrows, int ncols)
{
	for(int r = 0; r < nrows; r++)
	{
		//double y = csCpu.x_max - csCpu.x_min;
		double y = 4.0;
		y = y * (float)r / float(nrows) - 2.0;
		//y = csCpu.x_min + y;
		ySamples[r] = (float)y;
		for(int c = 0; c < ncols; c++)
		{
			//double x = csCpu.x_max - csCpu.x_min;
			double x = 4.0;
			x = x * (float)c / float(ncols) - 2.0;
			//x = csCpu.x_min + x;
			xSamples[c] = (float)x;

			double z1 = csCpu.val(x);
			double z2 = csCpu.val(y);
			rtnMtx[r*ncols + c] = (float)(z1*z2);
		}
	}
}


//Cuda Device Structure to hold Cubic Spline Coeficients
//and related info.
//Works with either device memory __device__
//or constant memory __constant__  with the same performance level.
__constant__ CubicSplineStruct CudaCubicSpline;

__global__ void testCubicSplineKernel(CubicSplineStruct *cudaCubicSpline, float *rtnMtx, int nrows, int ncols)
{
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	int r = blockDim.y * blockIdx.y + threadIdx.y;
	if( r < nrows && c < ncols)
	{
		CubicSpline_Cuda csObj(cudaCubicSpline);
		//float x = CudaCubicSpline.x_max - CudaCubicSpline.x_min;
		float x = 4.0f;
		x = x * (float)c / float(ncols) -2.0f;
		//x = CudaCubicSpline.x_min + x;
		float y1 = csObj.fx(x);

		//x = CudaCubicSpline.x_max - CudaCubicSpline.x_min;
		x = 4.0f;
		x = x * (float)r / float(nrows) - 2.0f;
		//x = CudaCubicSpline.x_min + x;
		float y2 = csObj.fx(x);

		rtnMtx[r * ncols + c] = y1 * y2;
	}
}

void testSmVecClass(void)
{
	int vecSize = 3;
	int nrows = 1024;
	int ncols = 1024;
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

	Cubic_Spline csCpu("Wavelet.DAT");
	
	CubicSpline_HostRef cudaCubicSplineRef(CudaCubicSpline);
	if( cudaCubicSplineRef.ReadFileXYDataPoints("Wavelet.DAT") )
	{
		printf("Error reading the wavelet file and setting up coeffients.\n");
		exit(1);
	}
	cudaCubicSplineRef.setExtrpolationType(0);
	cudaCubicSplineRef.initCubicSplineCudaObject();

	nrows = 1024;
	ncols = 1024;

	size_t vecMemSize = vecSize * sizeof(float);
	size_t resultVecSize = nrows * ncols * sizeof(float);
	float *host_rtnMtx = (float*)malloc(resultVecSize);
	float *cuda_rtnMtx = (float*)malloc(resultVecSize);
	float *xSamples = (float*)malloc(ncols * sizeof(float));
	float *ySamples = (float*)malloc(nrows * sizeof(float));
	float *device_rtnMtx;

	cudaMalloc(&device_rtnMtx, resultVecSize);

	/*testCubicSplineKernel<<<1,nrows>>>(device_rtnMtx, nrows, ncols);
	err = cudaGetLastError();
	ERROR_CHECK(err);
	err = cudaDeviceSynchronize();
	ERROR_CHECK(err);

	err = cudaMemcpy(cuda_rtnMtx, device_rtnMtx, resultVecSize, cudaMemcpyDeviceToHost);
	ERROR_CHECK(err);*/

	//Run SmVec Cpp Test
	printf( "Start CPU Cubic Spline Calculations.\n" );
	sdkCreateTimer(&timerCpp);
	sdkStartTimer(&timerCpp);

	testCubicSplineCPU(csCpu, xSamples, ySamples, host_rtnMtx, nrows, ncols);

	sdkStopTimer(&timerCpp);
	elapsedTimeInMs = sdkGetTimerValue(&timerCpp);
	printf( "SmVec CPU Cubic Spline Calculations. Time=%f\n",  elapsedTimeInMs);

	//Run the Cuda process
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(nrows / dimBlock.x, ncols / dimBlock.y);

	printf( "Start Cuda Matrix Calculation. \n" );
	void *ptrSymbol;
	err = cudaGetSymbolAddress(&ptrSymbol, CudaCubicSpline);
	checkCudaErrors(err);

	sdkCreateTimer(&timerCuda);

	for(int k = 0; k < 10; k++)
	{
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));

		checkCudaErrors(cudaEventRecord(start, 0));
		testCubicSplineKernel<<<dimGrid, dimBlock>>>((CubicSplineStruct *)ptrSymbol, device_rtnMtx, nrows, ncols);

		//Make sure Cuda has finish the calculations.
		checkCudaErrors(cudaEventRecord(stop, 0));
		checkCudaErrors(cudaDeviceSynchronize());
		sdkStopTimer(&timerCuda);
		checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));

		cudaMemcpy(cuda_rtnMtx, device_rtnMtx, resultVecSize, cudaMemcpyDeviceToHost);

		printf( "Cuda Matrix Calculation Complete.  Time(msec)=%f\n",  elapsedTimeInMs);

		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		//Compare Results
		for(int i = 0; i < nrows * ncols; i++)
		{
			float delR = fabs( host_rtnMtx[i] - cuda_rtnMtx[i] );
			float div = fabs( host_rtnMtx[i] ) < 0.001 ? 0.001 : fabs( host_rtnMtx[i] );
			float err = 100.0 * delR / div;
			if( err > 0.01 )
			{
				printf( "idx = %d, Percent Err = %g, v1 = %g, v2 = %g\n", i, err, host_rtnMtx[i], cuda_rtnMtx[i]);
			}
		}
	}

	//Write results to file for plotting.
	write2DArrayToFile(xSamples, ySamples, cuda_rtnMtx, nrows, ncols, "CudaScaleFn.dat");

	cudaCubicSplineRef.deleteCubicSplineObject();

	cudaFree(device_rtnMtx);
	free(host_rtnMtx);
	free(cuda_rtnMtx);
	free(xSamples);
	free(ySamples);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();
	sdkDeleteTimer(&timerCpp);
	sdkDeleteTimer(&timerCuda);
    printf("Done\n");
}


__device__ NeualNetworkCudaStruct CudaNNStructRef;
__constant__ CubicSplineStruct CudaNNCubicSpline;

void testNeuralNetworkClass(void)
{
	printf("Run init_control_surface before running the testNeuralNetworkClass!\n");
	printf("Start Cuda Neural Network Test.\n");

	const int N = 2;
	const int M = 2;
	const float Xc[N] = {0.0f,0.0f};  	// Center of the region of operation 
	SmVec_f X_Center(N,Xc);

	const float Xmax[N] = {1.57f,10.0f};      // Positive quadrant Range of x about the center.
	SmVec_f X_Max(N,Xmax);					 // These are the Plant Limits of Operation.

	const float Xmax_nn[N] = {2.0f,12.0f};    // Positive quadrant Range of x about the center.
	SmVec_f X_Max_nn(N,Xmax_nn);					 // These are the Neural Net. Limits of Operation.


	const float del_X[N] = {0.10f,0.5f};   // GNN node increments in x.
	SmVec_f Del_X(N,del_X);

	// Region of Asymptotic Stability  
	const float del_X_ras[N] = {0.05f,0.25f};   // GNN node increments in x.
	SmVec_f Del_X_ras(N,del_X_ras);


	// The next set of constants are required for the Discreatized version "ds" of the system: 

	const float Xmax_ds[N] = {4.0f,12.0f};      // Positive quadrant Range of x about the center.
	SmVec_f X_Max_ds(N,Xmax_ds);

	const float del_X_ds[N] = {0.1f,0.1f};   // GNN node increments in x.
	SmVec_f Del_X_ds(N,del_X_ds);





	NeuralNetwork_Host cudaNN(CudaNNStructRef, CudaNNCubicSpline, "SCALEFN.DAT", N, M, 
					X_Center, X_Max_nn, Del_X, 8);

	cudaNN.init_output_coef(0, u_initial);

	SmVec_f xInp(2);
	xInp.val(0) = 0.379;
	xInp.val(1) = 2.39;
	SmVec_f deltaX = 2.356 * Del_X;
	for(int i = 0; i < 10; i++)
	{
		float yCudann = cudaNN.compute(xInp, 0);
		float yStdnn = control_nn.compute(xInp, 0);
		float err = fabs(yCudann - yStdnn);
		printf("CudaNN=%f,  StdNN=%f,  Err=%f \n", yCudann, yStdnn, err);
		xInp = xInp + deltaX;
	}


    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits

	cudaNN.deleteNeuralNetworkObject();
    cudaDeviceReset();
    printf("Done\n");

}