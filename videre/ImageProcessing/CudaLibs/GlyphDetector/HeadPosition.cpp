/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athrs: Randy Direen PhD, Harry Direen PhD
 * Date: July 2018
  *******************************************************************/

#include "HeadPosition.h"
#include <algorithm>
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/cudalegacy.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "MathUtils.h"
#include "XYZCoord_t.h"
#include "Quaternion.h"

using namespace ImageProcLibsNS;
using namespace cv;
using namespace MathLibsNS;

namespace CudaImageProcLibsTrackHeadNS
{

    HeadPosition::HeadPosition()
    {
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

    }

    HeadPosition::~HeadPosition()
    {

    }

    /**
    * @brief
    *
    * @param frame
    */
    bool HeadPosition::init(ImageProcLibsNS::CameraCalibrationData &cameraCalData,
                            GlyphModel &gm)
    {
        bool error = false;
        IsDataValild_ = false;
        countSinceLastValidData_ = 0;
        lastValidHeadOrientationData_.Clear();
        try
        {
            setCameraCalibration(cameraCalData);
            model_ = gm.get_model();

            coordinates_.clear();
            coordinates_.push_back(cv::Point3f(0.0f, 0.0f, 0.0f));
            coordinates_.push_back(cv::Point3f(100.0f, 0.0f, 0.0f));
            coordinates_.push_back(cv::Point3f(0.0f, 100.0f, 0.0f));
            coordinates_.push_back(cv::Point3f(0.0f, 0.0f, 100.0f));

        }
        catch (std::exception &e)
        {
            LOGERROR("HeadPosition:init: Exception: " << e.what());
            error = true;
        }
        return error;
    }

    void HeadPosition::setCameraCalibration(ImageProcLibsNS::CameraCalibrationData &cameraCalData)
    {
        camera_matrix_ = cameraCalData.cvIntrinsicCalM;
        distortion_coeffs_ = cameraCalData.cvDistortionCoeff;
        // uncalibrated distortion coefficients
        //distortion_coeffs_ = cv::Mat::zeros(1, 5,cv::DataType<double>::type);

        //The cuda calibration matrix is in float values so copy across specifically:
        cam_mat_cuda_ = cv::Mat::zeros(3, 3, CV_32FC1);
        cam_mat_cuda_.at<float>(0, 0) = (float)cameraCalData.cvIntrinsicCalM.at<double>(0,0);
        cam_mat_cuda_.at<float>(0, 1) = (float)cameraCalData.cvIntrinsicCalM.at<double>(0,1);
        cam_mat_cuda_.at<float>(0, 2) = (float)cameraCalData.cvIntrinsicCalM.at<double>(0,2);
        cam_mat_cuda_.at<float>(1, 0) = (float)cameraCalData.cvIntrinsicCalM.at<double>(1,0);
        cam_mat_cuda_.at<float>(1, 1) = (float)cameraCalData.cvIntrinsicCalM.at<double>(1,1);
        cam_mat_cuda_.at<float>(1, 2) = (float)cameraCalData.cvIntrinsicCalM.at<double>(1,2);
        cam_mat_cuda_.at<float>(2, 0) = (float)cameraCalData.cvIntrinsicCalM.at<double>(2,0);
        cam_mat_cuda_.at<float>(2, 1) = (float)cameraCalData.cvIntrinsicCalM.at<double>(2,1);
        cam_mat_cuda_.at<float>(2, 2) = (float)cameraCalData.cvIntrinsicCalM.at<double>(2,2);

        distortion_coeffs_cuda_ = cv::Mat::zeros(1, 5, CV_32FC1);
        distortion_coeffs_cuda_.at<float>(0, 0) = (float)cameraCalData.cvDistortionCoeff.at<double>(0,0);
        distortion_coeffs_cuda_.at<float>(0, 1) = (float)cameraCalData.cvDistortionCoeff.at<double>(0,1);
        distortion_coeffs_cuda_.at<float>(0, 2) = (float)cameraCalData.cvDistortionCoeff.at<double>(0,2);
        distortion_coeffs_cuda_.at<float>(0, 3) = (float)cameraCalData.cvDistortionCoeff.at<double>(0,3);
        distortion_coeffs_cuda_.at<float>(0, 4) = (float)cameraCalData.cvDistortionCoeff.at<double>(0,4);
    }

    void HeadPosition::setHeadOrientationData(bool dataValid)
    {
        if(dataValid)
        {
            HeadOrientationData.HeadOrientationQuaternion.MakeQuaternionFromRodriguesVector(rvec_.at<double>(0),
                                                                                            rvec_.at<double>(1),
                                                                                            rvec_.at<double>(2));
            //Force the Quaterion to be in a consistant orientation (q and -q are identical orientations.
            //The Image processing can flip between the two which causes great pains when the image processing
            //head orientation quaternion is combined with the constant flipping of sign.
            //The positive orientation is consistant with the qyro measurement.
            //This check uses the fact that the x-axis is close to 1 or -1
            if(HeadOrientationData.HeadOrientationQuaternion.qVec.x < 0)
            {
                HeadOrientationData.HeadOrientationQuaternion = HeadOrientationData.HeadOrientationQuaternion * -1.0;
            }

            HeadOrientationData.HeadTranslationVec.x = tvec_.at<double>(0);
            HeadOrientationData.HeadTranslationVec.y = tvec_.at<double>(1);
            HeadOrientationData.HeadTranslationVec.z = tvec_.at<double>(2);
        }
        else
        {
            HeadOrientationData.Clear();
        }
        HeadOrientationData.IsDataValid = IsDataValild_;
    }

    void HeadPosition::setHeadOrientationDataCuda(bool dataValid)
    {
        if(dataValid)
        {
            HeadOrientationData.HeadOrientationQuaternion.MakeQuaternionFromRodriguesVector(rvec_.at<double>(0),
                                                                                            rvec_.at<double>(1),
                                                                                            rvec_.at<double>(2));
            //Force the Quaterion to be in a consistant orientation (q and -q are identical orientations.
            //The Image processing can flip between the two which causes great pains when the image processing
            //head orientation quaternion is combined with the constant flipping of sign.
            //The positive orientation is consistant with the qyro measurement.
            //This check uses the fact that the x-axis is close to 1 or -1
            if(HeadOrientationData.HeadOrientationQuaternion.qVec.x < 0)
            {
                HeadOrientationData.HeadOrientationQuaternion = HeadOrientationData.HeadOrientationQuaternion * -1.0;
            }

            HeadOrientationData.HeadTranslationVec.x = (double) tvec_.at<float>(0);
            HeadOrientationData.HeadTranslationVec.y = (double) tvec_.at<float>(1);
            HeadOrientationData.HeadTranslationVec.z = (double) tvec_.at<float>(2);
        }
        else
        {
            HeadOrientationData.Clear();
        }
        HeadOrientationData.IsDataValid = IsDataValild_;
    }

    //These checks are by imperical data at this point in time.
    //Keeps from having wild data being used for a measurement input.
    bool HeadPosition::validateOrientation()
    {
        if(HeadOrientationData.IsDataValid)
        {
            bool valid = !HeadOrientationData.HeadOrientationQuaternion.CheckNAN();
            valid &= HeadOrientationData.HeadOrientationQuaternion.qVec.x > 0.60;
            valid &= fabs(HeadOrientationData.HeadOrientationQuaternion.qVec.y) < 0.50;
            valid &= fabs(HeadOrientationData.HeadOrientationQuaternion.qVec.z) < 0.65;
            valid &= fabs(HeadOrientationData.HeadOrientationQuaternion.qScale) < 0.5;
            HeadOrientationData.IsDataValid = valid;
            if(valid)
            {
                lastValidHeadOrientationData_ = HeadOrientationData;
                countSinceLastValidData_ = 1;
            }
        }
        return HeadOrientationData.IsDataValid;
    }

    bool HeadPosition::checkAndSetupReasonableRansacStartPoint()
    {
        bool useExtrinsicGuess = false;
        XYZCoord_t rvec;
        if(lastValidHeadOrientationData_.IsDataValid && countSinceLastValidData_ > 0 &&  countSinceLastValidData_ < 25)
        {
            rvec = lastValidHeadOrientationData_.HeadOrientationQuaternion.MakeRodriguesVector();
            rvec_.at<double>(0) = rvec.x;
            rvec_.at<double>(1) = rvec.y;
            rvec_.at<double>(2) = rvec.z;

            tvec_.at<double>(0) = lastValidHeadOrientationData_.HeadTranslationVec.x;
            tvec_.at<double>(1) = lastValidHeadOrientationData_.HeadTranslationVec.y;
            tvec_.at<double>(2) = lastValidHeadOrientationData_.HeadTranslationVec.z;
            useExtrinsicGuess = true;
        }
        return useExtrinsicGuess;
    }

    /**
    * @brief
    *
    * @param indices
    * @param image_points
    */
    bool HeadPosition::update(std::vector<int>& indices, std::vector<cv::Point2f> image_points)
    {
        IsDataValild_ = false;
        try
        {
            list_3d_points_.clear();
            if(indices.size() >= 8)
            {
                for(size_t i = 0; i < indices.size(); ++i)
                {
                    list_3d_points_.push_back(model_[indices[i]]);
                }

                if(UseGPU)
                {
                    //The GPU version has issues.. don't use it at this time
                    //it does not speed up the process... it runs slower
                    //estimatePoseCuda(list_3d_points_, image_points);
                    estimatePose(list_3d_points_, image_points);
                }
                else
                {
                    estimatePose2(list_3d_points_, image_points);
                }

                if(DisplayType == HeadTrackingImageDisplayType_e::HTID_HeadOrientationVector)
                {

                    std::vector<cv::Point2f> pc = proj_coordinates_;
                    blank_.setTo(0);
                    if(pc.size() == 4)
                    {
                        line(blank_, pc[0], pc[1], cv::Scalar(0, 0, 255), 2);
                        line(blank_, pc[0], pc[2], cv::Scalar(0, 255, 0), 2);
                        line(blank_, pc[0], pc[3], cv::Scalar(255, 0, 0), 2);
                    }

                    for(size_t i = 0; i < dots_.size(); ++i)
                    {
                        circle(blank_, dots_[i], 2, cv::Scalar(255, 255, 255), 2, 7, 0);
                    }
                    cv::resize(blank_, blank60_, cv::Size(), .6, .6);
                    blank60_.copyTo(debugImgIn_(cv::Rect(cv::Point(400, 230), blank60_.size())));

                    cv::putText(debugImgIn_, cv::format("inliers: %d", num_inliers()),
                                cv::Point(400, 450),
                                cv::FONT_HERSHEY_COMPLEX, 0.5,
                                cv::Scalar(0, 255, 0), 1);
                }
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("HeadPosition:update: Exception: " << e.what());
            IsDataValild_ = false;
        }
        return IsDataValild_;
    }


    bool HeadPosition::estimatePose(std::vector<cv::Point3f>& p3d, std::vector<cv::Point2f>& p2d)
    {
        try
        {
            useExtrinsicGuess_ = checkAndSetupReasonableRansacStartPoint();
            cv::solvePnPRansac(p3d, p2d, camera_matrix_, distortion_coeffs_,
                               rvec_, tvec_, useExtrinsicGuess_, iterations_count_,
                               reprojection_error_, confidence_, inliers_, flags_);

            //We need atleast 4 corners on a glypyh to be valid... to insure
            //a reasonable orientation.
            IsDataValild_ = inliers_.rows >= 4;
            setHeadOrientationData(IsDataValild_);
            IsDataValild_ = validateOrientation();

            if(DisplayType == HeadTrackingImageDisplayType_e::HTID_HeadOrientationVector)
            {
                projectPoints(coordinates_, rvec_, tvec_, camera_matrix_, distortion_coeffs_,
                              proj_coordinates_);

                projectPoints(model_, rvec_, tvec_, camera_matrix_, distortion_coeffs_,
                              dots_);
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("HeadPosition:estimatePose: Exception: " << e.what());
            HeadOrientationData.Clear();
            IsDataValild_ = false;
        }
        return IsDataValild_;
    }

    bool HeadPosition::estimatePoseCuda(std::vector<cv::Point3f>& p3d, std::vector<cv::Point2f>& p2d)
    {
        try
        {
            cv::Mat p3dm = cv::Mat(p3d);
            cv::Mat p2dm = cv::Mat(p2d);

            cv::cuda::solvePnPRansac(p3dm.t(), p2dm.t(), cam_mat_cuda_, distortion_coeffs_cuda_,
                                     rvec_, tvec_, useExtrinsicGuess_, iterations_count_,
                                     reprojection_error_, 1 );

            IsDataValild_ = true;
            setHeadOrientationDataCuda(IsDataValild_);

            if(DisplayType == HeadTrackingImageDisplayType_e::HTID_HeadOrientationVector)
            {
                projectPoints(coordinates_, rvec_, tvec_, camera_matrix_, distortion_coeffs_,
                              proj_coordinates_);

                projectPoints(model_, rvec_, tvec_, camera_matrix_, distortion_coeffs_,
                              dots_);
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("HeadPosition:estimatePoseCuda: Exception: " << e.what());
            HeadOrientationData.Clear();
            IsDataValild_ = false;
        }
        return IsDataValild_;
    }

    bool HeadPosition::estimatePose2(std::vector<cv::Point3f>& p3d, std::vector<cv::Point2f>& p2d)
    {
        try
        {
            useExtrinsicGuess_ = checkAndSetupReasonableRansacStartPoint();
            cv::solvePnP(p3d, p2d, camera_matrix_, distortion_coeffs_,
                         rvec_, tvec_, useExtrinsicGuess_,
                         flags_);

            IsDataValild_ = true;
            setHeadOrientationData(IsDataValild_);
            IsDataValild_ = validateOrientation();

            if(DisplayType == HeadTrackingImageDisplayType_e::HTID_HeadOrientationVector)
            {
                projectPoints(coordinates_, rvec_, tvec_, camera_matrix_, distortion_coeffs_,
                              proj_coordinates_);

                projectPoints(model_, rvec_, tvec_, camera_matrix_, distortion_coeffs_,
                              dots_);
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("HeadPosition:estimatePose2: Exception: " << e.what());
            IsDataValild_ = false;
            HeadOrientationData.Clear();
        }
        return IsDataValild_;
    }


}
