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

#ifndef VIDERE_DEV_OPENCVMATUTILS_H
#define VIDERE_DEV_OPENCVMATUTILS_H

#include <opencv2/core.hpp>


namespace ImageProcLibsNS
{

    //Generate a Roll-Axis (around x-axis) rotation matrix.
    //outpRot_3x3_Mtx is expected to be a 3x3 matrix of doubles.
    void Generate_Roll_XAxis_RotationMtx(double angleRad, cv::Mat &outpRot_3x3_Mtx);

    //Generate a Pitch (around y-axis) rotation matrix.
    //outpRot_3x3_Mtx is expected to be a 3x3 matrix of doubles.
    void Generate_Pitch_YAxis_RotationMtx(double angleRad, cv::Mat &outpRot_3x3_Mtx);

    //Generate a Yaw (around z-axis) rotation matrix.
    //outpRot_3x3_Mtx is expected to be a 3x3 matrix of doubles.
    void Generate_Yaw_ZAxis_RotationMtx(double angleRad, cv::Mat &outpRot_3x3_Mtx);


    class OpenCVMatUtils
    {

    };

}


#endif //VIDERE_DEV_OPENCVMATUTILS_H
