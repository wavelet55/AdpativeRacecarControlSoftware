/* ****************************************************************
 * Camera Calibration with 2D Objects
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Nov. 2016
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
 * The Camera Calibration routines have been taken from:
 * http://sourishghosh.com/2016/camera-calibration-cpp-opencv/
 * Source code:  https://github.com/sourishg/stereo-calibration
    *******************************************************************/

#ifndef VIDERE_DEV_CAMERACALIBRATIONWITH2DOBJECTS_H
#define VIDERE_DEV_CAMERACALIBRATIONWITH2DOBJECTS_H

#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <string>
#include <boost/filesystem.hpp>
#include "global_defines.h"
#include "logger.h"
#include "../Utilities/JpgFileHandling.h"
#include "CameraCalibrationData.h"

namespace CameraCalibrationNS
{


    class CameraCalibrationWith2DObjects
    {

    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        std::vector< std::vector< cv::Point3f > > _realWorldObjPtsVec;
        std::vector< std::vector< cv::Point2f > > _imageCalPtsVec;
        std::vector< cv::Point3f > _realWorldObjPts;
        std::vector< cv::Point2f > _imageCalPts;

        std::vector<cv::Mat> _cameraRotationMtxVec;
        std::vector<cv::Mat> _cameraTranslationVecVec;

        int _numberOfObjects_X_Axis = 6;    //Number of Chess board corners x-axis
        int _numberOfObjects_Y_Axis = 7;

        double _chessBoardSquareSizeMilliMeters = 1.0;  //Standard lenghts are in meters.

        double _totalCalProjectionError = 0;

        cv::Mat _calImage;
        cv::Size _calImageSize;

        VidereImageprocessing::JpgFileHandler _JpgFileHandler;

    public:

        videre::CameraCalibrationType_e CalType;

        cv::Mat CameraDistortionMat;
        cv::Mat CameraIntrinsicCalMat;


        double Get_TotalCalProjectionError()
        {
            return _totalCalProjectionError;
        }

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

        //get the Chess Board Square size in Milli-Meters.
        //Meters are the standard unit of length in the Vedere system
        //and must be used for camera calibration so that the calibrated
        //image on the ground is in meters.
        int Get_ChessBoardSquareSizeMilliMeters()
        {
            return _chessBoardSquareSizeMilliMeters;
        }

        //Set the Chess Board Square size in Milli-Meters.
        //Meters are the standard unit of length in the Vedere system
        //and must be used for camera calibration so that the calibrated
        //image on the ground is in meters.
        void Set_ChessBoardSquareSizeMilliMeters(double val)
        {
            _chessBoardSquareSizeMilliMeters = val < 1.e-3 ? 1e-3 ? val > 1e6 : 1e6 : val;
        }

        //The Calibration routine computes a gray-scale image of the input
        //color image.  It will also paint the chess or other pattern
        //on the gray-scale image for user verification.  The image is accessed
        //here.
        cv::Mat GrayScaleImg;


        CameraCalibrationWith2DObjects();

        //Clear Calibration before starting a new calibration.
        void Clear();

        //Generate the Real World Points based on the geometery of the
        //Chess board or calibration image type. This fills in the
        //object_points array.
        void GenerateRealWorldPoints();


        //Generate the Real World Points based on the geometery of the
        //Chess board.
        //This fills in the object_points array.
        void GenerateRealWorldPointsChessBd(int numObjXaxis, int numObjYaxis, double squareSize);


        //check an Image to verify that all the Chess board corner crossings
        //can be found.
        //Returns true if ok, false if Chess board corners cannot be found.
        // If outpImg != null, provide the checker board image
        //with the corner locations drawn/painted on to send to user for
        //verification.
        bool CheckChessBoardImage(const cv::Mat &inPImg, bool drawPtsOnImage = true);

        //Process a Calibration Image to generate
        bool ProcessCalImage(const cv::Mat &inpImg);

        //Do the font-end processing of the list of calibration images.
        //Return the number of valid images processed.
        int ProcessCalImages(std::vector<boost::filesystem::path> imageFilenames);

        //Run Calibration... asssumes front-end image processing
        //has been run.
        int RunCalibration();

        //Run Calibration on the list of images.
        //Returns the number of valid images processed.
        int RunCalibration(std::vector<boost::filesystem::path> imageFilenames);

        //May be run after a calibration process to compute total projection error.
        double computeReprojectionErrors();

        //Used for Test
        void ComputeUnDistoredImagePoints(std::vector<cv::Point2f> &imgPts,
                                          std::vector<cv::Point2f> &outpPts);


        void WriteCalToCameraCalData(ImageProcLibsNS::CameraCalibrationData &cameraCalData);
    };

}
#endif //VIDERE_DEV_CAMERACALIBRATIONWITH2DOBJECTS_H
