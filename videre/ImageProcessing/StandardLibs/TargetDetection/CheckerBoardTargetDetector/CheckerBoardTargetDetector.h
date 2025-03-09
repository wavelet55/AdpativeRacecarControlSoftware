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

#ifndef STANDARD_CHECKERBOARDTARGETDETECTOR_H
#define STANDARD_CHECKERBOARDTARGETDETECTOR_H

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include "BlobDetectorFixedParameters.h"
#include "CommonImageProcTypesDefs.h"

using namespace ImageProcLibsNS;
using namespace CommonBlobTargetDetectorNS;

namespace CheckerBoardTargetDetectorNS
{


    //Checker Baoard Target Detector
    //Detects location of cross-corner locations on a checker board pattern.
    class CheckerBoardTargetDetector
    {
    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        std::vector< cv::Point2f > _imageTgtPts;

        int _numberOfObjects_X_Axis = 6;    //Number of Chess board corners x-axis
        int _numberOfObjects_Y_Axis = 7;
        double _chessBoardSquareSizeMilliMeters = 1.0;  //Standard lenghts are in meters.

    public:

        // Mark found targets in red on image if set
        bool MarkTargetEnabled = false;

        int NumberOfCBCornders = 0;

        //A Grey Scale image of the input image... may also
        //contain the marked up image.
        cv::Mat GrayScaleImg;

    public:
        CheckerBoardTargetDetector();

        ~CheckerBoardTargetDetector();

        //Create a Checker Board-detector and load with basic parameters
        bool Initialize();

        //Release all resources.
        void Close();

        //For a Chess/Checker Board, the number of x-axis objects is the number of
        //black corner crossings in the x direction.  A chess board with 7 squares will
        //have 6 black-black corner crossings.
        int Get_NumberOfObjects_X_Axis()
        {
            return _numberOfObjects_X_Axis;
        }

        //For a Chess/Checker Board, the number of x-axis objects is the number of
        //black corner crossings in the x direction.  A chess board with 7 squares will
        //have 6 black-black corner crossings.
        void Set_NumberOfObjects_X_Axis(int val)
        {
            _numberOfObjects_X_Axis = val < 1 ? 1 : val > 1000 ? 1000 : val;
        }

        //For a Chess/Checker Board, the number of y-axis objects is the number of
        //black corner crossings in the y direction.  A chess board with 8 squares will
        //have 7 black-black corner crossings.
        int Get_NumberOfObjects_Y_Axis()
        {
            return _numberOfObjects_Y_Axis;
        }

        //For a Chess/Checker Board, the number of y-axis objects is the number of
        //black corner crossings in the y direction.  A chess board with 8 squares will
        //have 7 black-black corner crossings.
        void Set_NumberOfObjects_Y_Axis(int val)
        {
            _numberOfObjects_Y_Axis = val < 1 ? 1 : val > 1000 ? 1000 : val;
        }

        //Target Detection Method
        //Handles the Image Processing for detecting targets
        //and returns the results in the provide message.
        //Returns the number of targets found
        //A number less than zero indicates an error.
        int DetectTargets(cv::Mat *imgInpRGB,
                          std::vector<BlobTargetInfo_t> *tgtResults);



    };

}
#endif //STANDARD_BLOBTARGETDETECTOR_OPENCV_H
