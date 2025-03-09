/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Dec. 2017
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
  *******************************************************************/

#include "ImageColorThresholdKernel.h"
#include <cuda_runtime.h>

//#include "CudaBlobTargetDetectorOpenCVSimple.h"
#include "BlobDetectorFixedParameters.h"
#include "Utils/PixelColorValueCuda.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda_types.hpp>


namespace CudaImageProcLibsNS
{

//Generate a output image based on pixels that are within a color range.
//The input image can be from an EO or IR sensor.
//The output image matrix is assumed to be of the shape and size to
//hold the output image results.
//A part of the image can be masked by providing a mask rectangle,
//that part of the image will be black.
__global__ void ImageColorThresholdKernal(
        unsigned char* inputImgPtr,
        unsigned char* bwImagePtr,
        unsigned char* maskImagePtr,
        int imgHeight,
        int imgWidth,
        int numberChans,
        int inpMatStepSize,
        int outpMatStepSize,
        int maskImgMatStepSize,
        unsigned int minColorVal,
        unsigned int maxColorVal )
    {
        unsigned char R, G, B;
        unsigned char *imgInpData, *imgOutData, *mskData;

        PixelColorValueCuda_t minColor(0);
        PixelColorValueCuda_t maxColor(0);
        PixelColorValueCuda_t pixelVal(0);
        minColor.setColorWithFormat(minColorVal);
        maxColor.setColorWithFormat(maxColorVal);

        //2D Index of current thread
        const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
        const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

        if (yIndex < imgHeight && xIndex < imgWidth)
        {
            int inpRowOffset = yIndex * inpMatStepSize;
            imgInpData = (unsigned char *) (inputImgPtr + inpRowOffset + (xIndex * numberChans));

            //Assume single channel for output matrix.
            int outpRowOffset = yIndex * outpMatStepSize;
            imgOutData = (unsigned char *) (bwImagePtr + outpRowOffset + xIndex);

            B = imgInpData[0];
            G = imgInpData[1];
            R = imgInpData[2];

            if (maskImgMatStepSize > 0 && maskImagePtr != NULL)
            {
                int mskRowOffset = yIndex * maskImgMatStepSize;
                mskData = (unsigned char *) (maskImagePtr + mskRowOffset + xIndex);
                if (*mskData == 0)
                {
                    imgOutData[0] = 0;
                }
            } 
            else
            {
                pixelVal.setRGBColor(R, G, B);
                bool HSxFormat = false;
                if(minColor.colorFormat == ImageColorFormatCuda_e::IPCF_HSV)
                {
                    pixelVal.RGBToHSVFormat();
                    HSxFormat = true;
                }
                else if(minColor.colorFormat == ImageColorFormatCuda_e::IPCF_HSL)
                {
                    pixelVal.RGBToHSLFormat();
                    HSxFormat = true;
                }
                else if(minColor.colorFormat == ImageColorFormatCuda_e::IPCF_HSI)
                {
                    pixelVal.RGBToHSIFormat();
                    HSxFormat = true;
                }
                else if (minColor.colorFormat == ImageColorFormatCuda_e::IPCF_YCrCb)
                {
                    pixelVal.RGBToYCrCbFormat();
                }

                bool inRange = true;
                //Special Handling of Hue if HSx format is required
                if(HSxFormat && minColor.c0 >= maxColor.c0)
                {
                    //Hue wrap-around effect.
                    inRange &= pixelVal.c0 >= minColor.c0
                               || pixelVal.c0 <= maxColor.c0;
                }
                else
                {
                    inRange &= pixelVal.c0 >= minColor.c0
                               && pixelVal.c0 <= maxColor.c0;
                }

                if(HSxFormat && minColor.c1 >= maxColor.c1)
                {
                    //There may be some cases where Saturation min and max are reversed.
                    inRange &= pixelVal.c1 >= maxColor.c1
                               && pixelVal.c1 <= minColor.c1;
                }
                else
                {
                    inRange &= pixelVal.c1 >= minColor.c1
                               && pixelVal.c1 <= maxColor.c1;
                }

                inRange &= pixelVal.c2 >= minColor.c2
                           && pixelVal.c2 <= maxColor.c2;

                if(inRange)
                {
                    //Set to White
                    imgOutData[0] = 255;
                }
                else
                {
                    //Set to black
                    imgOutData[0] = 0;
                }
            }
        }
    }



    bool ImageColorThreshold(const cv::cuda::GpuMat &colorImageIn, cv::cuda::GpuMat &bwImageOut,
                             PixelColorValue_t &minColorValue, PixelColorValue_t &maxColorValue,
                             bool syncGPU, const cv::cuda::GpuMat *maskImage)
    {
        bool error = true;
        int imgHeight = colorImageIn.rows;
        int imgWidth = colorImageIn.cols;
        int imgNoChannels = colorImageIn.channels();
        int inpMatStepSize = colorImageIn.step;
        int outpMatStepSize = bwImageOut.step;

        unsigned char *maskDataPtr = nullptr;
        int maskImageStepSize = 0;
        if(maskImage != nullptr && maskImage->rows == imgHeight && maskImage->cols == imgWidth)
        {
            maskDataPtr = maskImage->data;
            maskImageStepSize = maskImage->step;
        }

        try
        {
            //Specify a reasonable block size
            const dim3 block(16,16);
            //Calculate grid size to cover the whole image
            const dim3 grid((imgWidth + block.x - 1)/block.x, (imgHeight + block.y - 1)/block.y);

            ImageColorThresholdKernal << < grid, block >> > (colorImageIn.data,
                    bwImageOut.data,
                    maskDataPtr,
                    imgHeight,
                    imgWidth,
                    imgNoChannels,
                    inpMatStepSize,
                    outpMatStepSize,
                    maskImageStepSize,
                    minColorValue.ToUInt(),
                    maxColorValue.ToUInt() );

            error = false;
            if(syncGPU)
            {
                if(cudaDeviceSynchronize() != cudaSuccess)
                {
                    error = true;
                }
            }
        }
        catch (std::exception &e)
        {
            //LOGERROR("Error ImageColorThreshold: Exception: " << e.what());
            error = true;
        }
        return error;
    }




}
