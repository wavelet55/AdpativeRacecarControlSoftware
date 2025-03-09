/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Sept 2016
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
 *  Common Image Processing Libs Types and Definitions
  *******************************************************************/


#ifndef VIDERE_DEV_COMMONIMAGEPROCTYPESDEFS_H
#define VIDERE_DEV_COMMONIMAGEPROCTYPESDEFS_H

#include <log4cxx/logger.h>
#include <boost/math/constants/constants.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include "XYZCoord_t.h"
#include "Quaternion.h"


/* Macros are defined hear to make things cleaner.
 * detangle the dependency of the code on log4cxx a bit.
 * These assume "log4cpp_" within the scope of macro.
 */

#define LOGTRACE( message )\
        LOG4CXX_TRACE(log4cpp_, message );
#define LOGDEBUG( message )\
        LOG4CXX_DEBUG(log4cpp_, message );
#define LOGINFO( message )\
        LOG4CXX_INFO(log4cpp_, message );
#define LOGWARN( message )\
        LOG4CXX_WARN(log4cpp_, message );
#define LOGERROR( message )\
        LOG4CXX_ERROR(log4cpp_, message );
#define LOGFATAL( message )\
        LOG4CXX_FATAL(log4cpp_, message );


#define MATH_CNST_PI  (boost::math::constants::pi<double>())
#define MATH_CNST_TWOPI  (2.0 * boost::math::constants::pi<double>())
#define MATH_CNST_HALFPI  (0.5 * boost::math::constants::pi<double>())



namespace ImageProcLibsNS
{

    enum FeatureDetectorType_e
    {
        FDT_SIFT,
        FDT_ORB
    };

    //These must match the openCV type and order.
    enum FeatureMatcherType_e
    {
        FMT_FLANN = 1,
        FMT_BruteForce = 2,
        FMT_BruteForce_L1 = 3,
        FMT_BruteForce_Hamming1 = 4,
        FMT_BruteForce_Hamming2 = 5,
        FMT_BruteForce_SL2 = 6
    };


    enum FeatureMatchStrategy_e
    {
        FMS_BestMatch,
        FMS_K_NearestNeighbors,
        FMT_RadiusMatch
    };

    struct BlobTargetInfo_t
    {
        int TgtCenterPixel_x;
        int TgtCenterPixel_y;
        double TgtOrientationAngleDeg;   //OpenCV uses [0, 360)  -1 is undefinded.
        double TgtDiameterPixels;
        double TgtAreaSqPixels;
        double TgtParimeterPixels;
        bool IsTarget;
        int TargetType;

        BlobTargetInfo_t()
        {
            Clear();
        }

        BlobTargetInfo_t(const BlobTargetInfo_t &tgtInfo)
        {
            *this = tgtInfo;
        }

        void Clear()
        {
            TgtCenterPixel_x = 0;
            TgtCenterPixel_y = 0;
            TgtOrientationAngleDeg = 0;
            TgtDiameterPixels = 0;
            TgtAreaSqPixels = 0;
            TgtParimeterPixels = 0;
            IsTarget = false;
            TargetType = 0;
        }
    };


    struct TrackHeadOrientationData_t
    {
    public:
        //A Rodrigues Vector for Head Orientation
        MathLibsNS::Quaternion_t HeadOrientationQuaternion;

        MathLibsNS::XYZCoord_t HeadTranslationVec;

        bool IsDataValid = false;

        TrackHeadOrientationData_t()
        {
            Clear();
        }

        TrackHeadOrientationData_t(const TrackHeadOrientationData_t &data)
        {
            *this = data;
        }

        void Clear()
        {
            HeadOrientationQuaternion.Clear();
            HeadTranslationVec.Clear();
            IsDataValid = false;
        }

    };

    struct HeadOrientationCalData_t
    {
        MathLibsNS::Quaternion_t HeadToModelQ;

        MathLibsNS::Quaternion_t CameraToCarQ;

        MathLibsNS::Quaternion_t GyroToHeadQ;

        void Clear()
        {
            HeadToModelQ.MakeIdentity();
            CameraToCarQ.MakeIdentity();
            GyroToHeadQ.MakeIdentity();
        }

    };


    struct HeadTrackingParameters_t
    {
        int Canny_low;
        int Canny_high;
        int GlyphAreaPixels_min;
        int GlyphAreaPixels_max;

        int NumberOfIterations;
        double ReprojectionErrorDistance;
        double ConfidencePercent;           //[0, 100.0]

        HeadTrackingParameters_t()
        {
            SetDefaults();
        }

        HeadTrackingParameters_t(const HeadTrackingParameters_t &params)
        {
            *this = params;
        }

        void SetDefaults()
        {
            Canny_low = 50;
            Canny_high = 150;
            GlyphAreaPixels_min = 1000;
            GlyphAreaPixels_max = 8000;

            NumberOfIterations = 10;
            ReprojectionErrorDistance = 5.0;
            ConfidencePercent = 95.0;  //[0, 100.0]
        }
    };

    enum HeadTrackingImageDisplayType_e
    {
        HTID_None,
        HTID_HighLightGlyphs,
        HTID_HeadOrientationVector
    };

}

#endif //VIDERE_DEV_COMMONIMAGEPROCTYPESDEFS_H
