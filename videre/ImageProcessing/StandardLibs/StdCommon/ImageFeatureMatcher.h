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

#ifndef VIDERE_DEV_IMAGEFEATUREMATCHER_H
#define VIDERE_DEV_IMAGEFEATUREMATCHER_H

#include "CommonImageProcTypesDefs.h"
#include "ImageKeypointsAndFeatures.h"
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

namespace ImageProcLibsNS
{
    class ImageFeatureMatcher
    {
    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        cv::Ptr<cv::DescriptorMatcher> _descriptorMatcherPtr;

        FeatureMatcherType_e _featureMatcherType = FeatureMatcherType_e::FMT_FLANN;


        int _k_numberOfMatches = 2;

        double _matchRadius = 1.0;

        double _mlDistanceThreadhold = 0.75;

        ImageKeypointsAndFeatures *_lastQueryInpKptsFeatures = nullptr;
        ImageKeypointsAndFeatures *_lastMatchSetInpKptsFeatures = nullptr;

    public:

        FeatureMatcherType_e getFeatureMatcherType() { return _featureMatcherType; }

        void setFeatureMatcherType(FeatureMatcherType_e fdt)
        {
            if(_featureMatcherType != fdt)
            {
                _featureMatcherType = fdt;
                releaseFeatureMatcher();
            }
        }

        int getKNumberOfMatches() {return _k_numberOfMatches;}
        void setKNumberOfMatches(int val)
        {
            _k_numberOfMatches = val < 1 ? 1 : val > 10 ? 10 : val;
        }

        double getScaleFactor() {return _matchRadius;}
        void setScaleFactor(double val)
        {
            _matchRadius = val < 0.001 ? 0.001 : val > 1000000 ? 1000000 : val;
        }

        double getMLDistanceThreadhold() {return _mlDistanceThreadhold;}
        void setMLDistanceThreadhold(double val)
        {
            _mlDistanceThreadhold = val < 0.01 ? 0.01 : val > 1000.0 ? 1000.0 : val;
        }

        //If true and K_NearestNeighbors matching was used and _k_numberOfMatches == 2
        //then automatically run the reduce matches.
        bool ReduceKnnMatches = true;

        FeatureMatchStrategy_e FeatureMatchStrategy = FeatureMatchStrategy_e::FMS_K_NearestNeighbors;

        std::vector<cv::DMatch> MatchesBestSL;
        std::vector< std::vector< cv::DMatch> > MatchesML;


        ImageFeatureMatcher(FeatureMatcherType_e fmt = FeatureMatcherType_e::FMT_FLANN);

        ~ImageFeatureMatcher();


        //establish/setup the feature detector
        //Returns false if set up ok, true if there is an error setting
        //up the feture detector
        bool setupFeatureMatcher();

        void releaseFeatureMatcher()
        {
            if(!_descriptorMatcherPtr.empty())
            {
                _descriptorMatcherPtr.release();
            }
        }

        //matchFeatures
        //The queryInpKptsFeatures are the new image's keypoints and feature.
        //The matchSetInpKptsFeatures is the "database" of features that the new image is being
        //compared to.
        //returns the number of features found. A negative number inticates and error
        //occurred during the match process.
        int matchFeatures(ImageKeypointsAndFeatures queryInpKptsFeatures, ImageKeypointsAndFeatures matchSetInpKptsFeatures);

        //Can only be run after ImageFeatureMatcher is run.
        int reduceKnnMatches();


        //Close the feature detector and release memory objects.
        void releaseAllResources();

        //Create an Image from two images showing the matched keypoints.
        bool mapFeaturesAcrossImages(cv::InputArray inpImg1,
                                        ImageKeypointsAndFeatures &img1Keypoints,
                                        cv::InputArray inpImg2,
                                        ImageKeypointsAndFeatures &img2Keypoints,
                                        cv::InputOutputArray outpImg);


    };

}
#endif //VIDERE_DEV_IMAGEFEATUREMATCHER_H
