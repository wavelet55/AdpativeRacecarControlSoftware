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

// Blolb processing libraries
#include <cuda_runtime.h>

#include "CudaBlobTargetDetectorOpenCVSimple.h"
#include "BlobDetectorFixedParameters.h"
#include "../../CudaCommon/Utils/PixelColorValueCuda.h"
#include "../../CudaCommon/ImageColorThresholdKernel.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda_types.hpp>
#include <math.h>
#include <iostream>
#include <string>


using namespace std;
using namespace cv;
using namespace CudaImageProcLibsNS;


namespace CudaBlobTargetDetectorOpenCVNS
{


    CudaBlobTargetDetectorOpenCV::CudaBlobTargetDetectorOpenCV()
    {
        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);
        LOGINFO("Blob Target Detector (OpenCV Cuda) Created.")
    }

    CudaBlobTargetDetectorOpenCV::~CudaBlobTargetDetectorOpenCV()
    {
        Close();
    }

    void CudaBlobTargetDetectorOpenCV::Close()
    {
        releaseResources();
        LOGINFO("Blob Target Detector (OpenCV Cuda) Closed.")
    }

    void CudaBlobTargetDetectorOpenCV::releaseResources()
    {

        _cudaImgBWBlobMat.release();
        _cudaImgFiltedMat.release();
        _gaussianFilter.release();
        _blobDetector.release();
    }


    bool CudaBlobTargetDetectorOpenCV::Initialize()
    {
        bool error = false;

        if(!_blobDetector.empty())
        {
            _blobDetector.release();
        }

        _blobDetector = cv::SimpleBlobDetector::create(BlobDetectorParameters.BlobDetParams);
        BlobDetectorParameters.BlobDetParamChanged = false;
        if (_blobDetector.empty())
        {
            LOGERROR("CudaBlobTargetDetectorOpenCV: error creating the SimpleBlobDetector.")
            error = true;
        }

        LOGINFO("Blob Target Detector (OpenCV Cuda) Initialized.")
        return error;
    }


    //Target Detection Method
    //Handles the Image Processing for detecting targets
    //and returns the results in the provide message.
    //Returns the number of targets found
    //A number less than zero indicates an error.
    int CudaBlobTargetDetectorOpenCV::DetectTargets(cv::cuda::GpuMat *imgInpBGR,
                                          std::vector<BlobTargetInfo_t> *tgtResults)
    {
        int numberOfTargetsFound = 0;
        int imgHeight;
        int imgWidth;
        cv::cuda::GpuMat *imgInput = imgInpBGR;

        imgHeight = imgInpBGR->rows;
        imgWidth = imgInpBGR->cols;

        tgtResults->clear();
        if(_blobDetector.empty() || BlobDetectorParameters.BlobDetParamChanged)
        {
            releaseResources();
            Initialize();
            BlobDetectorParameters.BlobDetParamChanged = false;
        }

        //Filter the image
        if(BlobDetectorParameters.UseGausianFilter)
        {
            if(_gaussianFilter.empty() || BlobDetectorParameters.BlobDetParamChanged)
            {
                _gaussianFilter.release();
                _gaussianFilter = cv::cuda::createGaussianFilter(imgInpBGR->type(), imgInpBGR->type(),
                                            cv::Size(
                                                    BlobDetectorParameters.GausianFilterKernalSize,
                                                    BlobDetectorParameters.GausianFilterKernalSize),
                                            BlobDetectorParameters.GausianFilterSigma);
            }

            //Note:  We will not use cudaDeviceSynchronize()
            //Until after all the Cuda Size processing is done.
            //The non-blocking call allows the following code to execute
            //while the gpu is processing the filter.
            try
            {
                _gaussianFilter->apply(*imgInpBGR, _cudaImgFiltedMat);
                imgInput = &_cudaImgFiltedMat;
            }
            catch (std::exception &e)
            {
                LOGERROR("Cuda Target Detector Blob OpenCV Gaussian Filter: Exception: " << e.what());
            }
        }


        //Create a temporary cv::Mat for computation... only one is
        //needed and it is reused during the different phases of target detection.
        if( _cudaImgBWBlobMat.rows != imgInpBGR->rows
           || _cudaImgBWBlobMat.cols != imgInpBGR->cols )
        {
            _cudaImgBWBlobMat.release();
            _cudaImgBWBlobMat.create(imgHeight, imgWidth, CV_8U);
        }


        /************************
        if(BlobDetectorParameters.BackgroundModelingEnabled)
        {
            //Modify the image to remove background.
            ImageBackgroundRemovalProcess(imgInput, _imgBWBlobPtr);
        }
         ****************************/

        if(BlobDetectorParameters.DetectType1Targets )
                //&& _targetType1ParamsMsg->TargetTypeCode == 1
                //&& _targetType1ParamsMsg->TargetRGBColorCode != 0)
        {





            if( !ImageColorThreshold(*imgInput, _cudaImgBWBlobMat,
                                    BlobDetectorParameters.BlobMinColorValue,
                                    BlobDetectorParameters.BlobMaxColorValue,
                                    true) )
            {
                try
                {
                    //If Unified memory is working properly, we should be
                    //able to use the _cudaImgBWBlobMat data... this does a GpuMat.download behind the scenes
                    cv::Mat bwImg(_cudaImgBWBlobMat);

                    _blobDetector->detect(bwImg, _blobsLoc);
                    numberOfTargetsFound = _blobsLoc.size();
                }
                catch (std::exception &e)
                {
                    LOGERROR("Cuda Target Detector Blob OpenCV: Exception: " << e.what());
                    numberOfTargetsFound = 0;
                }

                if( numberOfTargetsFound > 0 )
                {
                    //CBlobGetOrientation blobOrientation;
                    BlobTargetInfo_t tgtInfo;
                    for(int i = 0; i < numberOfTargetsFound; i++)
                    {
                        tgtInfo.Clear();
                        cv::KeyPoint blobInfo = _blobsLoc[i];
                        tgtInfo.TgtCenterPixel_x = (int)blobInfo.pt.x;
                        tgtInfo.TgtCenterPixel_y = (int)blobInfo.pt.y;
                        tgtInfo.TgtOrientationAngleDeg = blobInfo.angle;
                        tgtInfo.TgtDiameterPixels = blobInfo.size;
                        //the keypoint size is the diameter... so assume a circle
                        //and compute a rough area:
                        tgtInfo.TgtAreaSqPixels = 0.25 * M_PI * (blobInfo.size * blobInfo.size);
                        //blobInfo.size is the approximate diameter of the blob
                        //assume a circle.
                        tgtInfo.TgtParimeterPixels = M_PI * blobInfo.size;
                        tgtResults->push_back(tgtInfo);
                    }

                }
            }
            else
            {
                LOGERROR("CudaBlobTargetDetector:  ImageColorThresholdKernalKernel failed");
                return -1;
            }
        }

        return numberOfTargetsFound;
    }


    //Modify the image to remove background.
    //ToDo:  Optimize for Cuda... it is ideal for optimization
    bool CudaBlobTargetDetectorOpenCV::ImageBackgroundRemovalProcess(cv::Mat *imgInp, cv::Mat *imgOutBkGnd)
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
                    if (fabs(nRval - BlobDetectorParameters.BGMode[k * 3]) <= BlobDetectorParameters.BGMode[k * 3 + 2] &&
                        fabs(nGval - BlobDetectorParameters.BGMode[k * 3 + 1]) <= BlobDetectorParameters.BGMode[k * 3 + 2])
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
        Mat mfilter = cv::getStructuringElement(cv::MORPH_RECT, Size(2*sizeMorphFilter + 1, 2*sizeMorphFilter + 1),
                                                Point(sizeMorphFilter, sizeMorphFilter));
        cv::dilate(*imgOutBkGnd, *imgOutBkGnd, mfilter);
        cv::erode(*imgOutBkGnd, *imgOutBkGnd, mfilter);

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

        /*****************************************************
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
         *****************************************************/

        return error;
    }


    bool CudaBlobTargetDetectorOpenCV::GetIntermediateImage(int imgNumber, cv::OutputArray outImg)
    {
        bool imgObtained = false;
        if(_cudaImgBWBlobMat.rows > 0 && _cudaImgBWBlobMat.cols > 0)
        {
            _cudaImgBWBlobMat.download(outImg);
            imgObtained = true;
        }
        return imgObtained;
    }


}
