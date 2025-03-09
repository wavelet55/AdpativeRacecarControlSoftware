/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Dec. 2017
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
  *******************************************************************/

/*******************************************************************
 * ORB Feature Detector:
 *
    @param nfeatures The maximum number of features to retain.
    @param scaleFactor Pyramid decimation ratio, greater than 1. scaleFactor==2 means the classical
    pyramid, where each next level has 4x less pixels than the previous, but such a big scale factor
    will degrade feature matching scores dramatically. On the other hand, too close to 1 scale factor
    will mean that to cover certain scale range you will need more pyramid levels and so the speed
    will suffer.
    @param nlevels The number of pyramid levels. The smallest level will have linear size equal to
    input_image_linear_size/pow(scaleFactor, nlevels).
    @param edgeThreshold This is size of the border where the features are not detected. It should
    roughly match the patchSize parameter.
    @param firstLevel It should be 0 in the current implementation.
    @param WTA_K The number of points that produce each element of the oriented BRIEF descriptor. The
    default value 2 means the BRIEF where we take a random point pair and compare their brightnesses,
    so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3
    random points (of course, those point coordinates are random, but they are generated from the
    pre-defined seed, so each element of BRIEF descriptor is computed deterministically from the pixel
    rectangle), find point of maximum brightness and output index of the winner (0, 1 or 2). Such
    output will occupy 2 bits, and therefore it will need a special variant of Hamming distance,
    denoted as NORM_HAMMING2 (2 bits per bin). When WTA_K=4, we take 4 random points to compute each
    bin (that will also occupy 2 bits with possible values 0, 1, 2 or 3).
    @param scoreType The default HARRIS_SCORE means that Harris algorithm is used to rank features
    (the score is written to KeyPoint::score and is used to retain best nfeatures features);
    FAST_SCORE is alternative value of the parameter that produces slightly less stable keypoints,
    but it is a little faster to compute.
    @param patchSize size of the patch used by the oriented BRIEF descriptor. Of course, on smaller
    pyramid layers the perceived image area covered by a feature will be larger.
    @param fastThreshold

   CV_WRAP static Ptr<ORB> create(int nfeatures=500, float scaleFactor=1.2f, int nlevels=8, int edgeThreshold=31,
                               int firstLevel=0, int WTA_K=2, int scoreType=ORB::HARRIS_SCORE, int patchSize=31, int fastThreshold=20);
 *
 ********************************************************************/




#ifndef VIDERE_DEV_IMAGEKEYPOINTSANDFEATURES_H
#define VIDERE_DEV_IMAGEKEYPOINTSANDFEATURES_H

#include "CommonImageProcTypesDefs.h"
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

using namespace ImageProcLibsNS;

namespace CudaImageProcLibsNS
{

    class CudaImageKeypointsAndFeatures
    {
    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        cv::Feature2D *_featureDetectorPtr = nullptr;
        cv::DescriptorMatcher *_descriptorMatcherPtr = nullptr;
        //std::vector<cv::Point2f> object_bb;

        int _maxNumberOfFeatures = 1000;
        double _scaleFactor = 1.2;
        int _numberOfLevels = 8;
        int _edgeThreashold = 31;
        int _firstLevel = 0;
        int _wta_k = 2;
        cv::ORB::ScoreType _scoreType= cv::ORB::ScoreType::HARRIS_SCORE;
        int _patchSize=31;
        int _fastThreshold=20;


        FeatureDetectorType_e _featureDetectorType = FeatureDetectorType_e::FDT_ORB;

    public:

        //A vector of the Key Points
        std::vector<cv::KeyPoint> KeyPointsVec;

        //An openCV Matrix of Features
        cv::Mat FeaturesMat;

        FeatureDetectorType_e getFeatureDetectorType() { return _featureDetectorType; }

        int getNumberOfKeyPoints()
        {
            return (int)KeyPointsVec.size();
        }

        void setFeatureDetectorType(FeatureDetectorType_e fdt)
        {
            if(_featureDetectorType != fdt)
            {
                _featureDetectorType = fdt;
            }
        }

        int getMaxNumberOfFeatures() {return _maxNumberOfFeatures;}
        void setMaxNumberOfFeatures(int val)
        {
            _maxNumberOfFeatures = val < 10 ? 10 : val > 1000000 ? 1000000 : val;
            releaseFeatureDetector();   //force feature detector to be re-created with new value.
        }

        double getScaleFactor() {return _scaleFactor;}
        void setScaleFactor(double val)
        {
            _scaleFactor = val < 1.0 ? 1.00 : val > 2.0 ? 2.0 : val;
            releaseFeatureDetector();   //force feature detector to be re-created with new value.
        }


        int getNumberOfLevels() {return _numberOfLevels;}
        void setNumberOfLevels(int val)
        {
            _numberOfLevels = val < 1 ? 1 : val > 100 ? 100 : val;
            releaseFeatureDetector();   //force feature detector to be re-created with new value.
        }

        int getEdgeThreashold() {return _edgeThreashold;}
        void setEdgeThreashold(int val)
        {
            _edgeThreashold = val < 1 ? 1 : val > 1000 ? 1000 : val;
            releaseFeatureDetector();   //force feature detector to be re-created with new value.
        }

        int getFirstLevel() {return _firstLevel;}
        void setFirstLevel(int val)
        {
            _firstLevel = val < 0 ? 0 : val > 1 ? 1 : val;
            releaseFeatureDetector();   //force feature detector to be re-created with new value.
        }

        int getWTA_K() {return _wta_k;}
        void setWTA_K(int val)
        {
            _wta_k = val < 1 ? 1 : val > 10 ? 10 : val;
            releaseFeatureDetector();   //force feature detector to be re-created with new value.
        }

        int getPatchSize() {return _patchSize;}
        void setPatchSize(int val)
        {
            _patchSize = val < 1 ? 1 : val > 1000 ? 1000 : val;
            releaseFeatureDetector();   //force feature detector to be re-created with new value.
        }

        int getFastThreshold() {return _fastThreshold;}
        void setFastThreshold(int val)
        {
            _fastThreshold = val < 1 ? 1 : val > 1000 ? 1000 : val;
            releaseFeatureDetector();   //force feature detector to be re-created with new value.
        }

        cv::ORB::ScoreType getScoreType() {return _scoreType;}
        void setScoreType(cv::ORB::ScoreType val)
        {
            _scoreType = val;
            releaseFeatureDetector();   //force feature detector to be re-created with new value.
        }




        CudaImageKeypointsAndFeatures(FeatureDetectorType_e fdt = FeatureDetectorType_e::FDT_ORB);

        ~CudaImageKeypointsAndFeatures();

        //generate the Keypoints and Features for the Image
        //The first time this method is called, it will establish the feature
        //detector... subsequent calls will use the feature detector previously
        //established.
        //returns false if generated ok, true if there was an error
        //generating the keypoints and images.
        bool generateKeypointsAndFeatures(const cv::Mat &img);

        //establish/setup the feature detector
        //Returns false if set up ok, true if there is an error setting
        //up the feture detector
        bool setupFeatureDetector();

        void releaseFeatureDetector()
        {
            if(_featureDetectorPtr != nullptr)
            {
                _featureDetectorPtr = nullptr;
            }
        }


        //Close the feature detector and release memory objects.
        void releaseAllResources();

    };

}
#endif //VIDERE_DEV_IMAGEKEYPOINTSANDFEATURES_H
