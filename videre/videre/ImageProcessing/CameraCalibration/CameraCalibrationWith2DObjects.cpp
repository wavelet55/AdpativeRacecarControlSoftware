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
 * http://docs.opencv.org/3.1.0/dc/dbb/tutorial_py_calibration.html
    *******************************************************************/


#include "CameraCalibrationWith2DObjects.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types_c.h>
#include <stdio.h>
#include <iostream>
//#include "popt_pp.h"

using namespace videre;
using namespace std;
using namespace cv;
using namespace ImageProcLibsNS;

namespace CameraCalibrationNS
{

    CameraCalibrationWith2DObjects::CameraCalibrationWith2DObjects()
    {
        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        CalType = CameraCalibrationType_e::CameraCal_2DPlaneCheckerBoard;
        _numberOfObjects_X_Axis = 6;    //Number of Chess board corners x-axis
        _numberOfObjects_Y_Axis = 7;
        _chessBoardSquareSizeMilliMeters = 0.03;  //Standard lenghts are in meters.
    }

    //Clear Calibration before starting a new calibration.
    void CameraCalibrationWith2DObjects::Clear()
    {
        _totalCalProjectionError = 0.0;
        _imageCalPts.clear();
        _realWorldObjPts.clear();
        _imageCalPtsVec.clear();
        _realWorldObjPtsVec.clear();
        CameraDistortionMat.release();
        CameraIntrinsicCalMat.release();
        _calImage.release();
        for(int i = 0; i < _cameraRotationMtxVec.size(); i++)
            _cameraRotationMtxVec[i].release();
        _cameraRotationMtxVec.clear();

        for(int i = 0; i < _cameraTranslationVecVec.size(); i++)
            _cameraTranslationVecVec[i].release();
        _cameraTranslationVecVec.clear();
    }


    //Generate the Real World Points based on the geometery of the
    //Chess board or calibration image type. This fills in the
    //object_points array.
    void CameraCalibrationWith2DObjects::GenerateRealWorldPoints()
    {
        _realWorldObjPts.clear();
        float sqSize = (float)_chessBoardSquareSizeMilliMeters;
        Point3f xyzPt;
        xyzPt.z = (float)0.0;
        for(int i = 0; i < _numberOfObjects_Y_Axis; i++)
        {
            xyzPt.y = (float)i * sqSize;
            for(int j = 0; j < _numberOfObjects_X_Axis; j++)
            {
                xyzPt.x = (float)j * sqSize;
                _realWorldObjPts.push_back(xyzPt);
            }
        }
     }


    //Generate the Real World Points based on the geometery of the
    //Chess board.
    //This fills in the object_points array.
    void CameraCalibrationWith2DObjects::GenerateRealWorldPointsChessBd(int numObjXaxis, int numObjYaxis, double squareSize)
    {
        Set_NumberOfObjects_X_Axis(numObjXaxis);
        Set_NumberOfObjects_Y_Axis(numObjYaxis);
        Set_ChessBoardSquareSizeMilliMeters(squareSize);
        GenerateRealWorldPoints();
    }


    //check an Image to verify that all the Chess board corner crossings
    //can be found.  If outpImg != null, provide the checker board image
    //with the corner locations drawn/painted on to send to user for
    //verification.
    bool CameraCalibrationWith2DObjects::CheckChessBoardImage(const cv::Mat &inpImg, bool drawPtsOnImage)
    {
        bool imgOk = false;
        Size board_size = Size(_numberOfObjects_X_Axis, _numberOfObjects_Y_Axis);
        int numCorners = _numberOfObjects_X_Axis * _numberOfObjects_Y_Axis;
        _imageCalPts.clear();
        try
        {
            if(inpImg.channels() > 1)
            {
                //ToDo: We may want to hand different color types.
                cv::cvtColor(inpImg, GrayScaleImg, cv::COLOR_BGR2GRAY);
            }
            else
            {
                //Assume the image is already gray scale.
                inpImg.copyTo(GrayScaleImg);
            }
            imgOk = cv::findChessboardCorners(GrayScaleImg, board_size, _imageCalPts,
                                              cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FILTER_QUADS);
            if (imgOk)
            {
                cornerSubPix(GrayScaleImg, _imageCalPts, cv::Size(5, 5), cv::Size(-1, -1),
                             TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.1));
                if(drawPtsOnImage)
                {
                    drawChessboardCorners(GrayScaleImg, board_size, _imageCalPts, imgOk);
                }
                if(_imageCalPts.size() <  numCorners)
                {
                    imgOk = false;
                    cout << "CheckChessBoardImage expected number of corners: " << numCorners
                         << " Number of corners found: " << _imageCalPts.size() << endl;
                }
            }
        }
        catch(Exception e)
        {
            LOGERROR("CameraCalibrationWith2DObjects::CheckChessBoardImage:  Exception" << e.what());
        }
        return imgOk;
    }

    bool CameraCalibrationWith2DObjects::ProcessCalImage(const cv::Mat &inpImg)
    {
        bool imgOk;
        imgOk = CheckChessBoardImage(inpImg, false);
        if(imgOk)
        {
            _realWorldObjPtsVec.push_back(_realWorldObjPts);
            _imageCalPtsVec.push_back(_imageCalPts);
        }
        return imgOk;
    }

    //Do the font-end processing of the list of calibration images.
    //Return the number of valid images processed.
    int CameraCalibrationWith2DObjects::ProcessCalImages(std::vector<boost::filesystem::path> imageFilenames)
    {
        int numImgsProcessed = 0;
        Clear();    //Ensure we are starting from a cleared point.
        GenerateRealWorldPoints();
        for(int i = 0; i < imageFilenames.size(); i++)
        {
            if(!_JpgFileHandler.ReadImageFromFile(_calImage, imageFilenames[i].c_str()))
            {
                if(ProcessCalImage(_calImage))
                {
                    _calImageSize = _calImage.size();
                    ++numImgsProcessed;
                }
            }
        }
        return numImgsProcessed;
    }

    int CameraCalibrationWith2DObjects::RunCalibration()
    {
        int numImgsProcessed = 0;
        int flag = 0;
        flag |= cv::CALIB_FIX_K4;
        flag |= cv::CALIB_FIX_K5;
        if(_realWorldObjPtsVec.size() == _imageCalPtsVec.size()
           && _imageCalPtsVec.size() > 3)
        {
            try
            {
                numImgsProcessed = _imageCalPtsVec.size();
                _totalCalProjectionError = calibrateCamera(_realWorldObjPtsVec,
                                                           _imageCalPtsVec,
                                                           _calImageSize,
                                                           CameraIntrinsicCalMat,
                                                           CameraDistortionMat,
                                                           _cameraRotationMtxVec,
                                                           _cameraTranslationVecVec,
                                                           flag);
            }
            catch(Exception e)
            {
                LOGERROR("CameraCalibrationWith2DObjects::RunCalibration exception: " << e.what());
                cout << "CameraCalibrationWith2DObjects::RunCalibration exception: " << e.what() << endl;
            }
        }

        return numImgsProcessed;
    }


    //Run Calibration on the list of images.
    //Returns the number of valid images processed.
    int CameraCalibrationWith2DObjects::RunCalibration(std::vector<boost::filesystem::path> imageFilenames)
    {
        int numImgsProcessed = 0;
        numImgsProcessed = ProcessCalImages(imageFilenames);
        if( numImgsProcessed > 0)
        {
            numImgsProcessed = RunCalibration();
        }
         return numImgsProcessed;
    }


    //May be run after a calibration process to compute total projection error.
    double CameraCalibrationWith2DObjects::computeReprojectionErrors()
    {
        vector<Point2f> imagePoints2;
        int i, totalPoints = 0;
        double totalErr = 0, err;
        vector<float> perViewErrors;
        if (_cameraRotationMtxVec.size() > 0 && _realWorldObjPtsVec.size() > 0)
        {
            try
            {
                perViewErrors.resize(_realWorldObjPtsVec.size());

                for (i = 0; i < (int) _realWorldObjPtsVec.size(); ++i)
                {
                    projectPoints(Mat(_realWorldObjPtsVec[i]),
                                  _cameraRotationMtxVec[i],
                                  _cameraTranslationVecVec[i],
                                  CameraIntrinsicCalMat,
                                  CameraDistortionMat,
                                  imagePoints2);
                    err = norm(Mat(_imageCalPtsVec[i]), Mat(imagePoints2), cv::NORM_L2);
                    int n = (int) _realWorldObjPtsVec[i].size();
                    perViewErrors[i] = (float) std::sqrt(err * err / n);
                    totalErr += err * err;
                    totalPoints += n;
                }
                std::vector<cv::Point2f> outpPts;

            }
            catch(Exception e)
            {
                LOGERROR("CameraCalibrationWith2DObjects::computeReprojectionErrors exception: " << e.what());
                cout << "CameraCalibrationWith2DObjects::computeReprojectionErrors exception: " << e.what() << endl;
            }
        }
        return std::sqrt(totalErr/totalPoints);
    }

    //Used for Test
    void CameraCalibrationWith2DObjects::ComputeUnDistoredImagePoints(std::vector<cv::Point2f> &imgPts,
                                                                      std::vector<cv::Point2f> &outpPts)
    {
        Point2f outpPt;
        cv::Mat pixInpM(1, 2, CV_64FC2);
        cv::Mat pixUnDistM(1, 2, CV_64FC2);

        outpPts.clear();
        for(int i = 0; i < imgPts.size(); i++)
        {
            pixInpM.at<double>(0) = imgPts[i].x;
            pixInpM.at<double>(1) = imgPts[i].y;

            cv::undistortPoints(pixInpM, pixUnDistM, CameraIntrinsicCalMat, CameraDistortionMat);
            outpPt.x = pixUnDistM.at<double>(0);
            outpPt.y = pixUnDistM.at<double>(1);
            outpPts.push_back(outpPt);

            cout << "Image Input: " << imgPts[i].x << ", " << imgPts[i].y << endl
                 << " Output: " << outpPt.x <<  ", " << outpPt.y << endl << endl;
        }
        cout << "Done" << endl;
    }

    void CameraCalibrationWith2DObjects::WriteCalToCameraCalData(CameraCalibrationData &cameraCalData)
    {
        cameraCalData.SetDistortionCalibrationData(CameraDistortionMat);
        cameraCalData.SetIntrinsicCalibrationData(CameraIntrinsicCalMat);
    }

}
