/* ****************************************************************
 * Blob Target Detector
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Sept 2016
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
 * The Blob Target Detector was orginally developed for the JHART
 * and earlier Huntsman programs by Dr. Hyukseong Kwon in the June 2012
 * timeframe.   It was updated for the Dominator project by Dr. Gruber
 * in March 2014.
 *
 * The Blob Target Detector is being optimized to use NVidia's GPGPU
 * processor technology.  OpenCV has optimizations for NVidia's GPGPU
 * in addition other blob detection algorithms are being optimized
 * to use the GPGPU.
 *
 * The Blob Target Detector makes use of a Blob Library.  I do not have
 * the details of where this library came from other than the header
 * information in the files.  Modifications and optimizations to this
 * library are being made as necessary to suport the Blob Target Detector.
 *
  *******************************************************************/

#include "CudaBlobTargetDetector.h"
#include "BlobDetectorFixedParameters.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <math.h>

// Blolb processing libraries
#include <cuda_runtime.h>
#include "BlobResult.h"
#include "BlobExtraction.h"
#include "Blob.h"

using namespace std;
using namespace cv;


namespace CudaBlobTargetDetectorNS
{


    //Generate a output image based on pixels that are within a color range.
    //The input image can be from an EO or IR sensor.
    //The output image matrix is assumed to be of the shape and size to
    //hold the output image results.
    //A part of the image can be masked by providing a mask rectangle,
    //that part of the image will be black.
    __global__ void ImageColorThresholdKernalKernel(
                                    unsigned char* inputImgPtr,
                                    unsigned char* tmpImg1ChPtr,
                                    int imgHeight,
                                    int imgWidth,
                                    int numberChans,
                                    int inpImgStep,
                                    int4 colorThresholds,
                                    bool IR_Sensor,
                                    bool maskSet,
                                    int2 maskC1,
                                    int2 maskC2 )
    {
        unsigned char R, G, B;
        int CrVal, CbVal;
        unsigned char *imgInpData, *imgOutData;

        //2D Index of current thread
        const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
        const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

        if(yIndex < imgHeight && xIndex < imgWidth)
        {
            int row_inimg = yIndex * inpImgStep;
            imgInpData = (uchar *) inputImgPtr + row_inimg + (xIndex * numberChans);

            //Assume single channel for output matrix.
            int row_outimg = yIndex * imgWidth;
            imgOutData = (uchar *) tmpImg1ChPtr + row_outimg + xIndex;

            B = imgInpData[0];
            G = imgInpData[1];
            R = imgInpData[2];
            if (maskSet != 0
                && xIndex + 1 >= maskC1.x
                && xIndex < maskC2.x
                && yIndex + 1 >= maskC1.y
                && yIndex < maskC2.y)
            {
                imgOutData[0] = 0;
            }
            else
            {
                if (!IR_Sensor)
                {
                    CrVal = (int) (0.4392 * R - 0.3678 * G - 0.0714 * B + 128);    // Cr image
                    CbVal = (int) (-0.1482 * R - 0.2910 * G + 0.4392 * B + 128);    // Cb image

                    //Compare to Color Thresholds
                    if (CrVal <= colorThresholds.x && CrVal >= colorThresholds.y
                        && CbVal <= colorThresholds.z && CbVal >= colorThresholds.w)
                    {
                        imgOutData[0] = 255;
                    } else
                    {
                        imgOutData[0] = 0;
                    }
                } else  //IR sensor
                {
                    CrVal = (int) (0.2566 * R + 0.5046 * G + 0.0978 * B + 16);    // Y image

                    //Compare to Color Thresholds
                    if (CrVal <= colorThresholds.x && CrVal >= colorThresholds.y)
                    {
                        imgOutData[0] = 255;
                    } else
                    {
                        imgOutData[0] = 0;
                    }
                }
            }
        }
    }


    CudaBlobTargetDetector::CudaBlobTargetDetector()
    {
        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);
        LOGINFO("Blob Target Detector (Huntsman/JHART Cuda) Created.")

        _imgTmp1Chan = 0;
        _cudaInpImgRGB = 0;
        _cudaInternalTmpImg1Chan = 0;

        _cudaInpImgRGB = 0;
        _cudaInpImgRGBMemSizeBytes = 0;
        _cudaInternalTmpImg1Chan = 0;
        _cudaInternalTmpImg1ChanMemSizeBytes = 0;
    }

    CudaBlobTargetDetector::~CudaBlobTargetDetector()
    {
        Close();
    }


    bool CudaBlobTargetDetector::Initialize()
    {
        bool error = false;

        //Set Mask Thresholds
        //ToDo:  Look at HOPS to determine the right values for these.
        _mask.MASK_PIXEL_X_MIN = 10;
        _mask.MASK_PIXEL_X_MAX = 100;
        _mask.MASK_PIXEL_Y_MIN = 10;
        _mask.MASK_PIXEL_Y_MAX = 100;

        LOGINFO("Blob Target Detector (Huntsman/JHART Cuda) Initialized.")
        return error;
    }

    void CudaBlobTargetDetector::Close()
    {
        if( _imgTmp1Chan != 0 )
        {
            delete _imgTmp1Chan;
            _imgTmp1Chan = 0;
        }
        if(_cudaInpImgRGB != 0)
        {
            cudaFree(_cudaInpImgRGB);
            _cudaInpImgRGB = 0;
            _cudaInpImgRGBMemSizeBytes = 0;
        }
        if(_cudaInternalTmpImg1Chan != 0)
        {
            cudaFree(_cudaInternalTmpImg1Chan);
            _cudaInternalTmpImg1Chan = 0;
            _cudaInternalTmpImg1ChanMemSizeBytes = 0;
        }
        LOGINFO("Blob Target Detector (Huntsman/JHART Cuda) Closed.")
    }

    //Target Detection Method
    //Handles the Image Processing for detecting targets
    //and returns the results in the provide message.
    //Returns the number of targets found
    //A number less than zero indicates an error.
    int CudaBlobTargetDetector::DetectTargets(cv::Mat *imgInpRGB,
                                          std::vector<BlobTargetInfo_t> *tgtResults)
    {
        int numberOfTargetsFound = 0;
        int imgHeight;
        int imgWidth;
        int imgAreaPixels;
        size_t inpImgSizeBytes;
        NumberOfType1Blobs = 0;
        NumberOfType2Blobs = 0;
        NumberOfType3Blobs = 0;
        cudaError_t cudaErr = cudaSuccess;

        imgHeight = imgInpRGB->rows;
        imgWidth = imgInpRGB->cols;
        imgAreaPixels = imgHeight * imgWidth;
        inpImgSizeBytes = (size_t)(imgInpRGB->step * imgInpRGB->rows);

        tgtResults->clear();

        //Create a temporary cv::Mat for computation... only one is
        //needed and it is reused during the different phases of target detection.
        if(_imgTmp1Chan == 0
           || _imgTmp1Chan->rows != imgInpRGB->rows
           || _imgTmp1Chan->cols != imgInpRGB->cols )
        {
            if( _imgTmp1Chan != 0 )
            {
                //This cv::Mat is the wrong size... delete
                //this one and create a new one ove the correct size.
                delete _imgTmp1Chan;
            }
            _imgTmp1Chan = new cv::Mat(imgHeight, imgWidth, CV_8U);
        }

        //Create the Cuda Device Side Memory if required
        if(_cudaInpImgRGB == 0
                || _cudaInpImgRGBMemSizeBytes < inpImgSizeBytes)
        {
            if(_cudaInpImgRGB !=0)
            {
                //Release the old memory and create new memory.
                cudaFree(_cudaInpImgRGB);
                _cudaInpImgRGB = 0;
                LOGINFO("CudaBlobTargetDetector Image Size Increase.  Old Size="
                                << _cudaInpImgRGBMemSizeBytes
                                << " New Size=" << inpImgSizeBytes);
            }
            _cudaInpImgRGBMemSizeBytes = inpImgSizeBytes;
            cudaErr = cudaMalloc(&_cudaInpImgRGB, _cudaInpImgRGBMemSizeBytes);
            if(cudaErr != cudaSuccess)
            {
                LOGERROR("CudaBlobTargetDetector:  cudaMalloc _cudaInpImgBGR failed.");
                //const char * cudaErrStr = _cudaGetErrorEnum(cudaErr);
                Close();
                return -1;
            }
        }

        if(_cudaInternalTmpImg1Chan == 0
           || _cudaInternalTmpImg1ChanMemSizeBytes < imgAreaPixels)
        {
            if(_cudaInternalTmpImg1Chan !=0)
            {
                //Release the old memory and create new memory.
                cudaFree(_cudaInternalTmpImg1Chan);
                _cudaInternalTmpImg1Chan = 0;
                LOGINFO("CudaBlobTargetDetector Temp Image Size Increase.  Old Size="
                                << _cudaInternalTmpImg1ChanMemSizeBytes
                                << " New Size=" << imgAreaPixels);
            }
            _cudaInternalTmpImg1ChanMemSizeBytes = imgAreaPixels;
            cudaErr = cudaMalloc(&_cudaInternalTmpImg1Chan, _cudaInternalTmpImg1ChanMemSizeBytes);
            if(cudaErr != cudaSuccess)
            {
                LOGERROR("CudaBlobTargetDetector:  cudaMalloc _cudaBWBlobImg failed.");
                //const char * cudaErrStr = _cudaGetErrorEnum(cudaErr);
                Close();
                return -1;
            }
        }

        //Copy Image to the Cuda Device Memory
        cudaErr = cudaMemcpy(_cudaInpImgRGB, imgInpRGB->data, inpImgSizeBytes, cudaMemcpyHostToDevice);
        if(cudaErr != cudaSuccess)
        {
            LOGERROR("CudaBlobTargetDetector:  cudaMemcpy Input Image to Cuda Space failed.");
            //const char * cudaErrStr = _cudaGetErrorEnum(cudaErr);
            Close();
            return -1;
        }


        // Image display for test purposes
        if (IMAGE_DISPLAY == 1)
        {
            cv::namedWindow("Original Image", CV_WINDOW_AUTOSIZE);
            cv::imshow("Original Image", *imgInpRGB);
            waitKey(0);
            cv::destroyWindow("Original Image");
        }

        if(BackgroundModelingEnabled)
        {
            //Modify the image to remove background.
            ImageBackgroundRemovalProcess(imgInpRGB, _imgTmp1Chan);

            // Image display
            if (IMAGE_DISPLAY == 1)
            {
                cv::namedWindow("Background Subtracted Color Image After Filtering", CV_WINDOW_AUTOSIZE);
                cv::imshow("Background Subtracted Color Image After Filtering", *imgInpRGB);
                cv::namedWindow("Background Subtracted BW Image After Filtering", CV_WINDOW_AUTOSIZE);
                cv::imshow("Background Subtracted BW Image After Filtering", *_imgTmp1Chan);
                waitKey(0);
                cv::destroyWindow("Background Subtracted Color Image After Filtering");
                cv::destroyWindow("Background Subtracted BW Image After Filtering");
            }
            // Image save
            if (IMAGE_SAVE == 1)
            {
                //cvSaveImage("resultBackgroundRGB.jpg", imgRGB);
                cv::imwrite("resultBackgroundFiltered.jpg", *_imgTmp1Chan);
            }

        }

        if(DetectType1Targets )
                //&& _targetType1ParamsMsg->TargetTypeCode == 1
                //&& _targetType1ParamsMsg->TargetRGBColorCode != 0)
        {
            //ToDo: Setup the color thresholds for the given target type.
            //This is for a Red target:
            _colorThresholds.SetColorThresholds(250, 230, 100, 80);

            int4 colorThresholds;
            colorThresholds.x = 250;
            colorThresholds.y = 230;
            colorThresholds.z = 100;
            colorThresholds.w = 80;

            int2 maskC1, maskC2;
            maskC1.x = 0;
            maskC1.y = 0;
            maskC2.x = 0;
            maskC2.y = 0;

            //Specify a reasonable block size
            const dim3 block(16,16);
            //Calculate grid size to cover the whole image
            const dim3 grid((imgWidth + block.x - 1)/block.x, (imgHeight + block.y - 1)/block.y);

            ImageColorThresholdKernalKernel<<<grid,block>>>(_cudaInpImgRGB,
                    _cudaInternalTmpImg1Chan,
                    imgHeight,
                    imgWidth,
                    imgInpRGB->channels(),
                    imgInpRGB->step,
                    colorThresholds,
                    IsIRSensor,
                    false, maskC1, maskC2);

            cudaErr = cudaDeviceSynchronize();

            if( cudaErr == cudaSuccess )
            {
                //Ist pass copy results back into
                cudaErr = cudaMemcpy(_imgTmp1Chan->data, _cudaInternalTmpImg1Chan,
                           imgAreaPixels, cudaMemcpyDeviceToHost);
                if(cudaErr != cudaSuccess)
                {
                    LOGERROR("CudaBlobTargetDetector:  cudaMemcpy Temp Image to Host Space failed.");
                    //const char * cudaErrStr = _cudaGetErrorEnum(cudaErr);
                    Close();
                    return -1;
                }


                // Blob extraction
                // CBlobResult_cuda parameters: 	inputImage: Input image. Must be a single channel,
                // maskImage: Image of mask off which is not calculated blobs,
                // threshold: Gray level to consider a pixel black or white,
                // findmoments: calculated moments of blobs or not
                blobsH = CBlobResult_cuda(_imgTmp1Chan, NULL, 0, true);
                blobsH.Filter(blobsH, B_EXCLUDE, CBlobGetArea(), B_LESS, SIZE_TYPE1_PATCH_LOW);
                blobsH.Filter(blobsH, B_EXCLUDE, CBlobGetArea(), B_GREATER, SIZE_TYPE1_PATCH_HIGH);
                numberOfTargetsFound = blobsH.GetNumBlobs();

                if( numberOfTargetsFound > 0 )
                {
                    CBlobGetOrientation blobOrientation;
                    BlobTargetInfo_t tgtInfo;
                    for(int i = 0; i < numberOfTargetsFound; i++)
                    {
                        CBlob_cuda *blobPtr = blobsH.GetBlob(i);
                        tgtInfo.TgtCenterPixel_x = 0.5 * (blobPtr->MaxX() + blobPtr->MinX());
                        tgtInfo.TgtCenterPixel_y = 0.5 * (blobPtr->MaxY() + blobPtr->MinY());
                        tgtInfo.TgtOrientationAngleDeg = blobOrientation(blobPtr);
                        tgtInfo.TgtAreaSqPixels = blobPtr->Area();
                        tgtInfo.TgtParimeterPixels = blobPtr->Perimeter();
                        tgtResults->push_back(tgtInfo);
                    }

                }
            }
            else
            {
                std::string cudaErrStr = _cudaGetErrorEnum(cudaErr);
                LOGERROR("CudaBlobTargetDetector:  ImageColorThresholdKernalKernel failed: " << cudaErrStr);
                return -1;
            }
        }

        return numberOfTargetsFound;
    }


    //Modify the image to remove background.
    //ToDo:  Optimize for Cuda... it is ideal for optimization
    bool CudaBlobTargetDetector::ImageBackgroundRemovalProcess(cv::Mat *imgInp, cv::Mat *imgOutBkGnd)
    {
        bool error = false;
        uchar *imgData;
        int imgHeight = imgInp->rows;
        int imgWidth = imgInp->cols;

        // Make Background (nRG) image
        int numBackgroundBlobs = 0;
        int numForegroundBlobs = 0;

        // With background modeling
        // Filtering with the background model
        double sumRGB, nRval, nGval;

        for (int i = 0; i < imgHeight; i++)
        {
            for (int j = 0; j < imgWidth; j++)
            {
                // Convert into nRG value
                int row = i * imgInp->step;
                int col = j * imgInp->channels();
                imgData = (uchar *)imgInp->data + row;
                sumRGB = (double) imgData[col] + (double) imgData[col + 1] + (double) imgData[col + 2];

                nRval = (double) numBGSample * (double) imgData[col + 0] / sumRGB;
                nGval = (double) numBGSample * (double) imgData[col + 1] / sumRGB;

                if (nRval == (double) numBGSample) nRval = (double) numBGSample - 1;
                if (nGval == (double) numBGSample) nGval = (double) numBGSample - 1;

                // Check if it is a background pixel or a foreground pixel
                for (int k = 0; k < 3; k++)
                {
                    row = i * imgOutBkGnd->step;
                    col = j;
                    imgData = (uchar *)imgOutBkGnd->data + row;
                    if (fabs(nRval - BGMode[k * 3]) <= BGMode[k * 3 + 2] &&
                        fabs(nGval - BGMode[k * 3 + 1]) <= BGMode[k * 3 + 2])
                    {
                        //Background
                        imgData[col] = 0;
                        break;
                    }
                    else
                    {
                        // Not background
                        imgData[col] = 255;
                    }
                }
            }
        }

        // Morphological filtering
        IplConvKernel *radiusFilter = 0;
        int shapeFilter = CV_SHAPE_RECT, numIteration = 1;
        radiusFilter = cvCreateStructuringElementEx(sizeMorphFilter * 2 + 1, sizeMorphFilter * 2 + 1,
                                                    sizeMorphFilter, sizeMorphFilter, shapeFilter, 0);
        cvDilate(imgOutBkGnd, imgOutBkGnd, radiusFilter, numIteration);
        cvErode(imgOutBkGnd, imgOutBkGnd, radiusFilter, numIteration);
        cvReleaseStructuringElement(&radiusFilter);

        // Remove color information if the pixel is part of background
        for (int i = 0; i < imgHeight; i++)
        {
            for (int j = 0; j < imgWidth; j++)
            {
                if (((uchar *) (imgOutBkGnd->data + i * imgOutBkGnd->step))[j] == 0)
                {
                    int row = i * imgInp->step;
                    int col = j * imgInp->channels();
                    imgData = (uchar *)imgInp->data + row;
                    imgData[col + 0] = 0;
                    imgData[col + 1] = 0;
                    imgData[col + 2] = 0;
                }
            }
        }

        // Size thresholding for foreground
        //blobsFG = CBlobResult_cuda(imgOutBkGnd, NULL, 0, true);
        blobsFG.Filter(blobsFG, B_EXCLUDE, CBlobGetArea(), B_LESS, SIZE_BOAT_LOW);
        blobsFG.Filter(blobsFG, B_EXCLUDE, CBlobGetArea(), B_GREATER, SIZE_BOAT_HIGH);
        numForegroundBlobs = blobsFG.GetNumBlobs();

        // Size thresholding for background
        //blobsBG = CBlobResult_cuda(imgOutBkGnd, NULL, 0, true);
        numBackgroundBlobs = blobsBG.GetNumBlobs();

        // Image recreation with selected blobs, then copy to the original rgb image
        for (int k = 0; k < numBackgroundBlobs; k++)
        {
            currentBlob = blobsBG.GetBlob(k);
            if (currentBlob->Area() > proportionBG * SIZE_BOAT_HIGH)
            {
                currentBlob->FillBlob(imgInp, CV_RGB(0, 0, 0));
            }
        }

        return error;
    }





}