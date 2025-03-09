/* ****************************************************************
 * Blob Target Detector
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Nov 2016
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
 * A Blob Target Detector based upon the OpenCV Blob Detection process.
 *
  *******************************************************************/

#include "StdBlobTargetDetectorOpenCVSimple.h"
#include "BlobDetectorFixedParameters.h"

#include <opencv2/core/types_c.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <math.h>

using namespace std;
using namespace cv;


namespace StdBlobTargetDetectorOpenCVNS
{

    StdBlobTargetDetectorOpenCVSimple::StdBlobTargetDetectorOpenCVSimple()
        : BlobDetectorParameters()
    {
        //_blobDetector = NULL;
        imgBWBlobPtr = nullptr;

        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);
        LOGINFO("Blob Target Detector (OpenCV Std) Created.")

        BlobDetectorParameters.setDefaultParameters();
    }

    StdBlobTargetDetectorOpenCVSimple::~StdBlobTargetDetectorOpenCVSimple()
    {
        Close();
    }


    bool StdBlobTargetDetectorOpenCVSimple::Initialize()
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
            LOGERROR("StdBlobTargetDetectorOpenCVSimple: error creating the SimpleBlobDetector.")
            error = true;
        }

        LOGINFO("Blob Target Detector (OpenCV Std) Initialized.")
        return error;
    }


    void StdBlobTargetDetectorOpenCVSimple::Close()
    {
        releaseResources();
        LOGINFO("Blob Target Detector (OpenCV Std) Closed.")
    }

    void StdBlobTargetDetectorOpenCVSimple::releaseResources()
    {
        if(imgBWBlobPtr != nullptr)
        {
            imgBWBlobPtr->release();
            delete(imgBWBlobPtr);
            imgBWBlobPtr = nullptr;
        }
        if(imgFiltedPtr != nullptr)
        {
            imgFiltedPtr->release();
            delete(imgFiltedPtr);
            imgFiltedPtr = nullptr;
        }
        if(!_blobDetector.empty())
            _blobDetector.release();

    }

    //Target Detection Method
    //Handles the Image Processing for detecting targets
    //and returns the results in the provide message.
    //Returns the number of targets found
    //A number less than zero indicates an error.
    int StdBlobTargetDetectorOpenCVSimple::DetectTargets(cv::Mat *imgInpBGR,
                                          std::vector<BlobTargetInfo_t> *tgtResults)
    {
        int numberOfTargetsFound = 0;
        int imgHeight;
        int imgWidth;
        NumberOfType1Blobs = 0;
        NumberOfType2Blobs = 0;
        NumberOfType3Blobs = 0;

        cv::Mat *imgInput = imgInpBGR;

        imgHeight = imgInpBGR->rows;
        imgWidth = imgInpBGR->cols;

        tgtResults->clear();
        if(_blobDetector.empty() || BlobDetectorParameters.BlobDetParamChanged)
        {
            releaseResources();
            Initialize();
            BlobDetectorParameters.BlobDetParamChanged = false;
        }

        //Filter the image  _imgFiltedPtr
        if(BlobDetectorParameters.UseGausianFilter)
        {
            if(imgFiltedPtr == nullptr
               || imgFiltedPtr->rows != imgInpBGR->rows
               || imgFiltedPtr->cols != imgInpBGR->cols )
            {
                if( imgFiltedPtr != nullptr )
                {
                    //This cv::Mat is the wrong size... delete
                    //this one and create a new one ove the correct size.
                    delete imgFiltedPtr;
                }
                imgFiltedPtr = new cv::Mat(imgHeight, imgWidth, CV_8UC3);
            }

            cv::GaussianBlur(*imgInpBGR, *imgFiltedPtr,
                             cv::Size(BlobDetectorParameters.GausianFilterKernalSize,
                                      BlobDetectorParameters.GausianFilterKernalSize),
                             BlobDetectorParameters.GausianFilterSigma );

            imgInput = imgFiltedPtr;
        }

        //Create a temporary cv::Mat for computation... only one is
        //needed and it is reused during the different phases of target detection.
        if(imgBWBlobPtr == nullptr
           || imgBWBlobPtr->rows != imgInpBGR->rows
           || imgBWBlobPtr->cols != imgInpBGR->cols )
        {
            if( imgBWBlobPtr != nullptr )
            {
                //This cv::Mat is the wrong size... delete
                //this one and create a new one ove the correct size.
                delete imgBWBlobPtr;
            }
            imgBWBlobPtr = new cv::Mat(imgHeight, imgWidth, CV_8U);
        }

        if(BlobDetectorParameters.BackgroundModelingEnabled)
        {
            //Modify the image to remove background.
            ImageBackgroundRemovalProcess(imgInput, imgBWBlobPtr);
        }

        if(BlobDetectorParameters.DetectType1Targets )
                //&& _targetType1ParamsMsg->TargetTypeCode == 1
                //&& _targetType1ParamsMsg->TargetRGBColorCode != 0)
        {

            //ToDo:  Sensor Type Input
            if( ImageColorThreshold(imgInput, imgBWBlobPtr, nullptr) )
            {
                LOGWARN("Blob Target Detector: Error doing ImageColorThreshold.")
            }
            else
            {
                try
                {
                    _blobDetector->detect(*imgBWBlobPtr, _blobsLoc);
                    numberOfTargetsFound = _blobsLoc.size();
                }
                catch (std::exception &e)
                {
                    LOGERROR("Target Detector Blob OpenCV: Exception: " << e.what());
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
        }

        return numberOfTargetsFound;
    }


    //Modify the image to remove background.
    //ToDo:  Optimize for Cuda... it is ideal for optimization
    bool StdBlobTargetDetectorOpenCVSimple::ImageBackgroundRemovalProcess(cv::Mat *imgInp, cv::Mat *imgOutBkGnd)
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

        /********************************************
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
         ************************************/

        return error;
    }


    //Generate a output image based on pixels that are within a color range.
    //The input image can be from an EO or IR sensor.
    //The output image matrix is assumed to be of the shape and size to
    //hold the output image results.
    //A part of the image can be masked by providing a mask rectangle,
    //that part of the image will be black.
    bool StdBlobTargetDetectorOpenCVSimple::ImageColorThreshold(const cv::Mat *imgInp,
                                                 cv::Mat *imgOut,
                                                 ImageMaskRect_t *maskImgRect)
    {
        bool error = false;
        unsigned char R, G, B;
        int CrVal, CbVal;
        unsigned char *imgInpData, *imgOutData;
        PixelColorValue_t pixelVal;

        int imgHeight = imgInp->rows;
        int imgWidth = imgInp->cols;
        if(imgOut->rows < imgHeight || imgOut->cols < imgWidth)
        {
           // LOGWARN("ImageColorThreshold: Output Image Matrix is too small to hold the generated image.")
            return true;
        }

        for(int i = 0; i < imgHeight; i++)
        {
            for(int j = 0; j < imgWidth; j++)
            {
                int row_inimg = i * imgInp->step;
                int col_inimg = j * imgInp->channels();
                imgInpData = (uchar *)imgInp->data + row_inimg;

                int row_outimg = i * imgOut->step;
                int col_outimg = j * imgOut->channels();
                imgOutData = (uchar *)imgOut->data + row_outimg;

                if(maskImgRect != nullptr
                        && j + 1 >= maskImgRect->MASK_PIXEL_X_MIN
                        && j < maskImgRect->MASK_PIXEL_X_MAX
                        &&   i + 1 >= maskImgRect->MASK_PIXEL_Y_MIN
                        && i < maskImgRect->MASK_PIXEL_Y_MAX)
                {
                    //Set to black
                    imgOutData[col_outimg] = 0;
                }
                else
                {
                    B = imgInpData[col_inimg + 0];
                    G = imgInpData[col_inimg + 1];
                    R = imgInpData[col_inimg + 2];
                    pixelVal.setRGBColor(R, G, B);
                    bool HSxFormat = false;
                    if(BlobDetectorParameters.BlobMinColorValue.colorFormat == ImageColorFormat_e::IPCF_HSV)
                    {
                        pixelVal.RGBToHSVFormat();
                        HSxFormat = true;
                    }
                    else if(BlobDetectorParameters.BlobMinColorValue.colorFormat == ImageColorFormat_e::IPCF_HSL)
                    {
                        pixelVal.RGBToHSLFormat();
                        HSxFormat = true;
                    }
                    else if(BlobDetectorParameters.BlobMinColorValue.colorFormat == ImageColorFormat_e::IPCF_HSI)
                    {
                        pixelVal.RGBToHSIFormat();
                        HSxFormat = true;
                    }
                    else if (BlobDetectorParameters.BlobMinColorValue.colorFormat == ImageColorFormat_e::IPCF_YCrCb)
                    {
                        pixelVal.RGBToYCrCbFormat();
                    }

                    bool inRange = true;
                    //Special Handling of Hue if HSx format is required
                    if(HSxFormat && BlobDetectorParameters.BlobMinColorValue.c0 >= BlobDetectorParameters.BlobMaxColorValue.c0)
                    {
                        //Hue wrap-around effect.
                        inRange &= pixelVal.c0 >= BlobDetectorParameters.BlobMinColorValue.c0
                                || pixelVal.c0 <= BlobDetectorParameters.BlobMaxColorValue.c0;
                    }
                    else
                    {
                        inRange &= pixelVal.c0 >= BlobDetectorParameters.BlobMinColorValue.c0
                                   && pixelVal.c0 <= BlobDetectorParameters.BlobMaxColorValue.c0;
                    }

                    if(HSxFormat && BlobDetectorParameters.BlobMinColorValue.c1 >= BlobDetectorParameters.BlobMaxColorValue.c1)
                    {
                        //There may be some cases where Saturation min and max are reversed.
                        inRange &= pixelVal.c1 >= BlobDetectorParameters.BlobMaxColorValue.c1
                                   && pixelVal.c1 <= BlobDetectorParameters.BlobMinColorValue.c1;
                    }
                    else
                    {
                        inRange &= pixelVal.c1 >= BlobDetectorParameters.BlobMinColorValue.c1
                                   && pixelVal.c1 <= BlobDetectorParameters.BlobMaxColorValue.c1;
                    }

                    inRange &= pixelVal.c2 >= BlobDetectorParameters.BlobMinColorValue.c2
                               && pixelVal.c2 <= BlobDetectorParameters.BlobMaxColorValue.c2;

                    if(inRange)
                    {
                        //Set to White
                        imgOutData[col_outimg] = 255;
                    }
                    else
                    {
                        //Set to black
                        imgOutData[col_outimg] = 0;
                    }
                }
            }
        }
        return error;
    }


    bool StdBlobTargetDetectorOpenCVSimple::GetIntermediateImage(int imgNumber, cv::OutputArray outImg)
    {
        bool imgObtained = false;
        if(imgBWBlobPtr != nullptr)
        {
            if(imgBWBlobPtr->rows > 0 && imgBWBlobPtr->cols > 0)
            {
                imgBWBlobPtr->copyTo(outImg);
                imgObtained = true;
            }
        }
        return imgObtained;
    }
}
