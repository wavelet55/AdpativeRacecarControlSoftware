/* ****************************************************************
 * Checker Board Target Detector
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: May 2017
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
 *
  *******************************************************************/

#include "CheckerBoardTargetDetector.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <math.h>

using namespace std;
using namespace cv;


namespace CheckerBoardTargetDetectorNS
{

    CheckerBoardTargetDetector::CheckerBoardTargetDetector()
    {
        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);
        LOGINFO("CheckerBoard Target Detector (OpenCV Std) Created.")
    }

    CheckerBoardTargetDetector::~CheckerBoardTargetDetector()
    {
        Close();
    }


    bool CheckerBoardTargetDetector::Initialize()
    {
        bool error = false;
        _imageTgtPts.clear();

        LOGINFO("CheckerBoard Target Detector Initialized.")
        return error;
    }

    void CheckerBoardTargetDetector::Close()
    {
        LOGINFO("CheckerBoard Target Detector (OpenCV Std) Closed.")
    }


    //Target Detection Method
    //Handles the Image Processing for detecting targets
    //and returns the results in the provide message.
    //Returns the number of targets found
    //A number less than zero indicates an error.
    int CheckerBoardTargetDetector::DetectTargets(cv::Mat *imgInpRGB,
                                          std::vector<BlobTargetInfo_t> *tgtResults)
    {
        int numberOfTargetsFound = 0;
        int imgHeight;
        int imgWidth;
        NumberOfCBCornders = 0;
        bool imgOk = false;

        imgHeight = imgInpRGB->rows;
        imgWidth = imgInpRGB->cols;
        Size board_size = Size(_numberOfObjects_X_Axis, _numberOfObjects_Y_Axis);

        _imageTgtPts.clear();
        tgtResults->clear();

        try
        {
            //Convert Image to Grey Scale.
            if(imgInpRGB->channels() > 1)
            {
                //ToDo: We may want to hand different color types.
                cv::cvtColor(*imgInpRGB, GrayScaleImg, cv::COLOR_BGR2GRAY);
            }
            else
            {
                //Assume the image is already gray scale.
                imgInpRGB->copyTo(GrayScaleImg);
            }

            imgOk = cv::findChessboardCorners(GrayScaleImg, board_size, _imageTgtPts, 
                                              cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FILTER_QUADS);
            if (imgOk)
            {
                cornerSubPix(GrayScaleImg, _imageTgtPts, cv::Size(5, 5), cv::Size(-1, -1),
                             TermCriteria( cv::TermCriteria::EPS| cv::TermCriteria::MAX_ITER, 30, 0.1));
                if(MarkTargetEnabled)
                {
                    drawChessboardCorners(GrayScaleImg, board_size, _imageTgtPts, imgOk);
                }
            }

             //_blobDetector->detect(*imgBWBlobPtr, _blobsLoc);
            numberOfTargetsFound = _imageTgtPts.size();
        }
        catch (std::exception &e)
        {
            LOGERROR("Target Detector Checker Board: Exception: " << e.what());
            numberOfTargetsFound = 0;
        }

        if( numberOfTargetsFound > 0 )
        {
            //CBlobGetOrientation blobOrientation;
            BlobTargetInfo_t tgtInfo;
            for(int i = 0; i < numberOfTargetsFound; i++)
            {
                tgtInfo.TgtCenterPixel_x = (int)_imageTgtPts[i].x;
                tgtInfo.TgtCenterPixel_y = (int)_imageTgtPts[i].y;
                tgtInfo.TgtOrientationAngleDeg = 0;  //No Orientation
                tgtInfo.TgtAreaSqPixels = 1.0;    //Single point
                tgtInfo.TgtParimeterPixels = 1.0;  //Single point
                tgtResults->push_back(tgtInfo);
            }

        }
         return numberOfTargetsFound;
    }







}
