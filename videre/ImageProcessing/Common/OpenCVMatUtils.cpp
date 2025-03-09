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
  *******************************************************************/

#include "OpenCVMatUtils.h"

namespace ImageProcLibsNS
{


    //Generate a Roll-Axis (around x-axis) rotation matrix.
    //outpRot_3x3_Mtx is expected to be a 3x3 matrix of doubles.
    void Generate_Roll_XAxis_RotationMtx(double angleRad, cv::Mat &outpRot_3x3_Mtx)
    {
        outpRot_3x3_Mtx = 0.0;
        double a = cos(angleRad);
        double b = sin(angleRad);
        outpRot_3x3_Mtx.at<double>(0,0) = 1;
        outpRot_3x3_Mtx.at<double>(1,1) = a;
        outpRot_3x3_Mtx.at<double>(2,2) = a;
        outpRot_3x3_Mtx.at<double>(1,2) = -b;
        outpRot_3x3_Mtx.at<double>(2,1) = b;
    }


    //Generate a Pitch (around y-axis) rotation matrix.
    //outpRot_3x3_Mtx is expected to be a 3x3 matrix of doubles.
    void Generate_Pitch_YAxis_RotationMtx(double angleRad, cv::Mat &outpRot_3x3_Mtx)
    {
        outpRot_3x3_Mtx = 0.0;
        double a = cos(angleRad);
        double b = sin(angleRad);
        outpRot_3x3_Mtx.at<double>(1,1) = 1;
        outpRot_3x3_Mtx.at<double>(0,0) = a;
        outpRot_3x3_Mtx.at<double>(2,2) = a;
        outpRot_3x3_Mtx.at<double>(2,0) = -b;
        outpRot_3x3_Mtx.at<double>(0,2) = b;
    }

    //Generate a Yaw (around z-axis) rotation matrix.
    //outpRot_3x3_Mtx is expected to be a 3x3 matrix of doubles.
    void Generate_Yaw_ZAxis_RotationMtx(double angleRad, cv::Mat &outpRot_3x3_Mtx)
    {
        outpRot_3x3_Mtx = 0.0;
        double a = cos(angleRad);
        double b = sin(angleRad);
        outpRot_3x3_Mtx.at<double>(2,2) = 1;
        outpRot_3x3_Mtx.at<double>(0,0) = a;
        outpRot_3x3_Mtx.at<double>(1,1) = a;
        outpRot_3x3_Mtx.at<double>(0,1) = -b;
        outpRot_3x3_Mtx.at<double>(1,0) = b;
    }




}