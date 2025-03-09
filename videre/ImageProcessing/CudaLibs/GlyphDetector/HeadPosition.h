/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athrs: Randy Direen PhD, Harry Direen PhD
 * Date: July 2018
  *******************************************************************/
#ifndef VIDERE_DEV_HEADPOSITION_H
#define VIDERE_DEV_HEADPOSITION_H

#include <iostream>
#include <opencv2/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/calib3d/calib3d_c.h"
#include "CommonImageProcTypesDefs.h"
#include "CameraCalibrationData.h"
#include "GlyphModel.h"


namespace CudaImageProcLibsTrackHeadNS
{

    class HeadPosition
    {
    public:
        ImageProcLibsNS::TrackHeadOrientationData_t HeadOrientationData;

        HeadPosition();

        ~HeadPosition();

        /**
        * @brief
        *
        * @param frame
        */
        bool init(ImageProcLibsNS::CameraCalibrationData &cameraCalData,
                  GlyphModel &gm);

        void Clear()
        {
            IsDataValild_ = false;
            HeadOrientationData.Clear();
        }

        /**
        * @brief
        *
        * @param indices
        * @param image_points
        */
        bool update(std::vector<int>& indices, std::vector<cv::Point2f> image_points);

        /**
        * @brief
        *
        * @return
        */
        std::vector<cv::Point2f>& proj_coordinates() {return proj_coordinates_;}

        /**
        * @brief
        *
        * @return
        */
        std::vector<cv::Point2f>& dots(){return dots_;}

        /**
        * @brief
        *
        * @return
        */
        int num_inliers(){return inliers_.size().height;}

        /**
        * @brief
        *
        * @param debugImgIn
        */
        void setDebugInMat(cv::Mat debugImgIn)
        {
            debugImgIn_ = debugImgIn;
            blank_.copySize(debugImgIn);
        }

        void setInputImageSize(cv::Mat inputImage)
        {
            if( blank_.rows != inputImage.rows
                || blank_.cols != inputImage.cols
                    || blank_.data == nullptr)
            {
                blank_ = inputImage.clone();
            }
        }

        /**
        * @brief
        *
        * @return
        */
        cv::Mat debugMat(){ return debugImgIn_; }

        bool newData()
        {
            bool r = IsDataValild_;
            IsDataValild_ = false;
            return r;
        }
        /**
        * @brief Rotation vector (see Rodrigues function for description)
        *
        * @return
        */
        cv::Mat rvec(){ return rvec_; }

        /**
        * @brief Translation vector
        *
        * @return
        */
        cv::Mat tvec(){ return tvec_; }

        /**
        * @brief A parameter for tuning SolvePnPRansac
        *
        * @param val
        */
        void set_iterations_count(int val){iterations_count_ = val;}

        /**
        * @brief A parameter for tuning SolvePnPRansac
        *
        * @param val
        */
        void set_reprojection_error(double val){reprojection_error_ = (float)val;}

        /**
        * @brief A parameter for tuning SolvePnPRansac
        *
        * @param confidence
        */
        void set_confidence(double confidencePercent){confidence_ = (float)(0.01 * confidencePercent);}

        //The camera calibration must be setup at init time to ensure
        //the camera is properly calibrated.
        void setCameraCalibration(ImageProcLibsNS::CameraCalibrationData &cameraCalData);

    private:
        void setHeadOrientationData(bool dataValid);

        void setHeadOrientationDataCuda(bool dataValid);

        bool checkAndSetupReasonableRansacStartPoint();

        bool estimatePose(std::vector<cv::Point3f>& p3d, std::vector<cv::Point2f>& p2d);

        bool estimatePoseCuda(std::vector<cv::Point3f>& p3d, std::vector<cv::Point2f>& p2d);

        bool estimatePose2(std::vector<cv::Point3f>& p3d, std::vector<cv::Point2f>& p2d);

    public:
        bool UseGPU = false;
        ImageProcLibsNS::HeadTrackingImageDisplayType_e DisplayType;

        //Returns true if orientation is valid/reasonable... false otherwise.
        bool validateOrientation();

    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        std::vector<cv::Point3f> model_;
        cv::Mat camera_matrix_;
        cv::Mat cam_mat_cuda_;
        cv::Mat distortion_coeffs_;
        cv::Mat distortion_coeffs_cuda_;

        //Head-Orientation Rodrigues Rotation Vector
        cv::Mat rvec_;

        //Head Translation Vector( X, Y , Z)
        cv::Mat tvec_;

        bool IsDataValild_ = false;

        int countSinceLastValidData_ = 0;

        //Last Valid Head Orientation Data
        ImageProcLibsNS::TrackHeadOrientationData_t lastValidHeadOrientationData_;

        bool useExtrinsicGuess_ = false;
        int iterations_count_ = 100;
        float reprojection_error_ = 2.0;
        float confidence_ = 0.95;
        //int flags_ = CV_ITERATIVE;
        int flags_ = CV_EPNP;

        std::vector<cv::Point3f> coordinates_;
        std::vector<cv::Point2f> proj_coordinates_;
        std::vector<cv::Point2f> dots_;

        std::vector<cv::Point3f> list_3d_points_;
        cv::Mat inliers_;

        cv::Mat debugImgIn_;
        cv::Mat blank_;
        cv::Mat blank60_;
        cv::Mat debugImg_;

    };

}
#endif //VIDERE_DEV_HEADPOSITION_H
