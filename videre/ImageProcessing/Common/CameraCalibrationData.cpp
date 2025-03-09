/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: April 2017
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
 *  Camera Calibration Data
  *******************************************************************/

#include "CameraCalibrationData.h"
#include "OpenCVMatUtils.h"

using namespace std;
using namespace cv;
using namespace ImageProcLibsNS;

namespace ImageProcLibsNS
{

    CameraCalibrationData::CameraCalibrationData()
            : cvIntrinsicCalM(3,3,CV_64F),
              cvDistortionCoeff(1,5,CV_64F),
              cvRotationCalM(3,3,CV_64F),
              cvTranslationCalM(3,1,CV_64F)
    {
        ClearAll();
        SetDefaults();
    }

    CameraCalibrationData::CameraCalibrationData(const CameraCalibrationData &ccd)
            : cvIntrinsicCalM(3,3,CV_64F),
              cvDistortionCoeff(1,5,CV_64F),
              cvRotationCalM(3,3,CV_64F),
              cvTranslationCalM(3,1,CV_64F)
    {
        ccd.cvIntrinsicCalM.copyTo(cvIntrinsicCalM);
        ccd.cvDistortionCoeff.copyTo(cvDistortionCoeff);
        ccd.cvRotationCalM.copyTo(cvRotationCalM);
        ccd.cvTranslationCalM.copyTo(cvTranslationCalM);
        SetCameraFocalLength( ccd.GetCameraFocalLength() );
        SetCalibrationScaleFactor( ccd.GetCalibrationScaleFactor());
    }


    CameraCalibrationData::~CameraCalibrationData()
    {
        cvIntrinsicCalM.release();
        cvDistortionCoeff.release();
        cvRotationCalM.release();
        cvTranslationCalM.release();
    }

    void CameraCalibrationData::ClearAll()
    {
        cvIntrinsicCalM = 0.0;
        cvDistortionCoeff = 0.0;
        cvRotationCalM = 0.0;
        cvTranslationCalM = 0.0;
        _cameraFocalLength = 1.0;
        _calibrationScaleFactor = 1.0;
        _calibrationScaleFactorInverse = 1.0;
        _YawCorrectionDegrees = 0;
        _PitchCorrectionDegrees = 0;
        _RollCorrectionDegrees = 0;
        _DelXCorrectionMeters = 0;
        _DelYCorrectionMeters = 0;
    }

    //Assumes a 640x480 camera with 65 degree FOV
    //
    void CameraCalibrationData::SetDefaults()
    {
        cvIntrinsicCalM.at<double>(0,0) = 500.0;
        cvIntrinsicCalM.at<double>(0,2) = 320.0;
        cvIntrinsicCalM.at<double>(1,1) = 500.0;
        cvIntrinsicCalM.at<double>(1,2) = 240.0;
        cvIntrinsicCalM.at<double>(2,2) = 1.0;

        //Set the Rotation matrix to be a Identity Matrix by default.
        //Swap x and y axis and mirror the y axis.... This gives normal
        //alignmed with a UAV and the camera pointing down.
        //x = -y and y = x
        cvRotationCalM.at<double>(0,1)  = -1.0;
        cvRotationCalM.at<double>(1,0)  = 1.0;
        cvRotationCalM.at<double>(2,2)  = 1.0;
    }

    //Copy the this CameraCalibrationData into outpMat;
    void CameraCalibrationData::CopyTo(CameraCalibrationData &outpCalData) const
    {
        cvIntrinsicCalM.copyTo(outpCalData.cvIntrinsicCalM);
        cvDistortionCoeff.copyTo(outpCalData.cvDistortionCoeff);
        cvRotationCalM.copyTo(outpCalData.cvRotationCalM);
        cvTranslationCalM.copyTo(outpCalData.cvTranslationCalM);
        outpCalData.SetCameraFocalLength( GetCameraFocalLength() );
        outpCalData.SetCalibrationScaleFactor( GetCalibrationScaleFactor());
    }

    void CameraCalibrationData::SetIntrinsicCalibrationData(double *intrinsicCal3x3Mtx)
    {
        cvIntrinsicCalM.at<double>(0, 0) = intrinsicCal3x3Mtx[0];
        cvIntrinsicCalM.at<double>(0, 1) = intrinsicCal3x3Mtx[1];
        cvIntrinsicCalM.at<double>(0, 2) = intrinsicCal3x3Mtx[2];
        cvIntrinsicCalM.at<double>(1, 0) = intrinsicCal3x3Mtx[3];
        cvIntrinsicCalM.at<double>(1, 1) = intrinsicCal3x3Mtx[4];
        cvIntrinsicCalM.at<double>(1, 2) = intrinsicCal3x3Mtx[5];
        cvIntrinsicCalM.at<double>(2, 0) = intrinsicCal3x3Mtx[6];
        cvIntrinsicCalM.at<double>(2, 1) = intrinsicCal3x3Mtx[7];
        cvIntrinsicCalM.at<double>(2, 2) = intrinsicCal3x3Mtx[8];
    }

    void CameraCalibrationData::SetIntrinsicCalibrationData(cv::Mat &intrinsicCal3x3Mtx)
    {
        intrinsicCal3x3Mtx.copyTo(cvIntrinsicCalM);
    }

    void CameraCalibrationData::SetDistortionCalibrationData(double *distortionCal5xVec)
    {
        //Note cvDistortionCoeff is a 1x5 matrix
        cvDistortionCoeff.at<double>(0,0) = distortionCal5xVec[0];
        cvDistortionCoeff.at<double>(0,1) = distortionCal5xVec[1];
        cvDistortionCoeff.at<double>(0,2) = distortionCal5xVec[2];
        cvDistortionCoeff.at<double>(0,3) = distortionCal5xVec[3];
        cvDistortionCoeff.at<double>(0,4) = distortionCal5xVec[4];
    }

    void CameraCalibrationData::SetDistortionCalibrationData(cv::Mat &distortionCal5xVec)
    {
        distortionCal5xVec.copyTo(cvDistortionCoeff);
    }


        //Rotation matrix from the camera coordinate frame to the uav coordinate frame
    void CameraCalibrationData::SetRotationCalibrationData(double *rotationCal3x3Mtx)
    {
        cvRotationCalM.at<double>(0, 0) = rotationCal3x3Mtx[0];
        cvRotationCalM.at<double>(0, 1) = rotationCal3x3Mtx[1];
        cvRotationCalM.at<double>(0, 2) = rotationCal3x3Mtx[2];
        cvRotationCalM.at<double>(1, 0) = rotationCal3x3Mtx[3];
        cvRotationCalM.at<double>(1, 1) = rotationCal3x3Mtx[4];
        cvRotationCalM.at<double>(1, 2) = rotationCal3x3Mtx[5];
        cvRotationCalM.at<double>(2, 0) = rotationCal3x3Mtx[6];
        cvRotationCalM.at<double>(2, 1) = rotationCal3x3Mtx[7];
        cvRotationCalM.at<double>(2, 2) = rotationCal3x3Mtx[8];
    }

    void CameraCalibrationData::SetRotationCalibrationData(cv::Mat &rotationCal3x3Mtx)
    {
        rotationCal3x3Mtx.copyTo(cvRotationCalM);
    }

        //Translation vector from the camera coordinate frame to the uav coordinate frame
    void CameraCalibrationData::SetTranslationCalibrationData(double *translationCal3xVec)
    {
        cvTranslationCalM.at<double>(0) = translationCal3xVec[0];
        cvTranslationCalM.at<double>(1) = translationCal3xVec[1];
        cvTranslationCalM.at<double>(2) = translationCal3xVec[2];
    }

    void CameraCalibrationData::SetTranslationCalibrationData(cv::Mat &translationCal3xVec)
    {
        translationCal3xVec.copyTo(cvTranslationCalM);
    }

    void CameraCalibrationData::GenerateRotationXlationCalFromCameraMountingCorrection()
    {
        double r1, r2;
        const double PI = boost::math::constants::pi<double>();
        cv::Mat rtMtx(3,3,CV_64F);
        rtMtx = 0.0;
        //Add 90 degrees to yaw to bring std camera orientation into
        //the x-axis aligning with the UAV.
        double yawDeg = 90.0 + _YawCorrectionDegrees;
        yawDeg = yawDeg > 180.0 ? yawDeg - 360.0 : yawDeg < -180.0 ? yawDeg + 360.0 : yawDeg;
        double angleRad = (PI / 180.0) * yawDeg;
        Generate_Yaw_ZAxis_RotationMtx(angleRad, cvRotationCalM);

        angleRad = (PI / 180.0) * _RollCorrectionDegrees;
        Generate_Pitch_YAxis_RotationMtx(angleRad, rtMtx);
        cvRotationCalM = cvRotationCalM * rtMtx;

        angleRad = (PI / 180.0) * _PitchCorrectionDegrees;
        Generate_Roll_XAxis_RotationMtx(angleRad, rtMtx);
        cvRotationCalM = cvRotationCalM * rtMtx;

        cvTranslationCalM.at<double>(0) = GetDelXCorrectionMeters();
        cvTranslationCalM.at<double>(1) = GetDelYCorrectionMeters();
        cvTranslationCalM.at<double>(2) = GetDelZCorrectionMeters();
    }

}