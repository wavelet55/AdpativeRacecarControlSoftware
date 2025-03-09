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


#include "ImageFeatureMatcher.h"

namespace ImageProcLibsNS
{

    ImageFeatureMatcher::ImageFeatureMatcher(FeatureMatcherType_e fmt)
    {
        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        setFeatureMatcherType(fmt);
    }

    ImageFeatureMatcher::~ImageFeatureMatcher()
    {
        releaseAllResources();
    }



    //establish/setup the feature matcher
    //Returns false if set up ok, true if there is an error setting
    //up the feture detector
    bool ImageFeatureMatcher::setupFeatureMatcher()
    {
        bool error = false;
        if(_descriptorMatcherPtr.empty() )
        {
            try
            {
                //_descriptorMatcherPtr = cv::DescriptorMatcher::create((int)_featureMatcherType);
                //OpenCV 4
                _descriptorMatcherPtr = cv::FlannBasedMatcher::create();
                error = false;
                if(_descriptorMatcherPtr.empty())
                {
                    LOGERROR("Error creating an Feature matcher." );
                    error = true;
                }
            }
            catch (std::exception &e)
            {
                LOGERROR("Error creating a Feature Matcher: Exception: " << e.what());
                error = true;
            }
        }
        return error;
    }


    //matchFeatures
    //The queryInpKptsFeatures are the new image's keypoints and feature.
    //The matchSetInpKptsFeatures is the "database" of features that the new image is being
    //compared to.
    //returns the number of features found. A negative number inticates and error
    //occurred during the match process.
    int ImageFeatureMatcher::matchFeatures(ImageKeypointsAndFeatures queryInpKptsFeatures,
                      ImageKeypointsAndFeatures matchSetInpKptsFeatures)
    {
        int noMatches = -1;
        try
        {
            if(!setupFeatureMatcher())
            {
                _lastQueryInpKptsFeatures = &queryInpKptsFeatures;
                _lastMatchSetInpKptsFeatures = &matchSetInpKptsFeatures;
                MatchesBestSL.clear();
                MatchesML.clear();
                switch(FeatureMatchStrategy)
                {
                    case FeatureMatchStrategy_e::FMS_BestMatch:
                        _descriptorMatcherPtr->match(queryInpKptsFeatures.FeaturesMat,
                                                        matchSetInpKptsFeatures.FeaturesMat, MatchesBestSL );
                        MatchesML.push_back(MatchesBestSL);
                        noMatches = MatchesBestSL.size();
                        break;
                    case FeatureMatchStrategy_e::FMS_K_NearestNeighbors:
                        _descriptorMatcherPtr->knnMatch(queryInpKptsFeatures.FeaturesMat,
                                                        matchSetInpKptsFeatures.FeaturesMat, MatchesML,
                                                        _k_numberOfMatches);
                        if(ReduceKnnMatches)
                        {
                            noMatches = reduceKnnMatches();
                        }
                        else
                        {
                            noMatches = MatchesML.size();
                        }
                        break;
                    case FeatureMatchStrategy_e::FMT_RadiusMatch:
                        _descriptorMatcherPtr->radiusMatch(queryInpKptsFeatures.FeaturesMat,
                                                        matchSetInpKptsFeatures.FeaturesMat, MatchesML,
                                                           _matchRadius);
                        noMatches = MatchesML.size();
                        break;
                    default:
                        LOGERROR("Error unsupported Feature matcher strategy." );
                        noMatches = -1;
                        break;
                }
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("Error matching features: Exception: " << e.what());
            noMatches = -1;
        }

        return noMatches;
    }

    //Can only be run after ImageFeatureMatcher is run.
    int ImageFeatureMatcher::reduceKnnMatches()
    {
        int noMatches = -1;
        MatchesBestSL.clear();
        if(_lastQueryInpKptsFeatures != nullptr && MatchesML.size() > 1 && MatchesML[0].size() > 1)
        {
            int Nf = _lastQueryInpKptsFeatures->FeaturesMat.rows;
            int Nm = MatchesML.size();
            int N = Nf < Nm ? Nf : Nm;

            for(int k = 0; k < N; k++)
            {
                if(MatchesML[k][0].distance < _mlDistanceThreadhold * MatchesML[k][1].distance)
                {
                    // take the first result only if its distance is smaller than 0.6*second_best_dist
                    // that means this descriptor is ignored if the second distance is bigger or of similar
                    MatchesBestSL.push_back(MatchesML[k][0]);
                }
            }
            noMatches = MatchesBestSL.size();
        }

        return noMatches;
    }


    //Close the feature detector and release memory objects.
    void ImageFeatureMatcher::releaseAllResources()
    {
        releaseFeatureMatcher();
        MatchesBestSL.clear();
        MatchesBestSL.shrink_to_fit();
        MatchesML.clear();
        MatchesML.shrink_to_fit();
    }


    //Create an Image from two images showing the matched keypoints.
    //The first image and keypoints should be the query image and keypoints.
    //Ths second image and keypoints is the primary or dictionary of keypoints.
    bool ImageFeatureMatcher::mapFeaturesAcrossImages(cv::InputArray inpImg1,
                                                            ImageKeypointsAndFeatures &img1Keypoints,
                                                            cv::InputArray inpImg2,
                                                            ImageKeypointsAndFeatures &img2Keypoints,
                                                            cv::InputOutputArray outpImg)
    {
        bool error = true;
        try
        {
            if(MatchesBestSL.size() > 0) {
                cv::drawMatches(inpImg1, img1Keypoints.KeyPointsVec,
                                inpImg2, img2Keypoints.KeyPointsVec,
                                MatchesBestSL,
                                outpImg);
                error = false;
            }
            else if(MatchesML.size() > 0)
            {
                cv::drawMatches(inpImg1, img1Keypoints.KeyPointsVec,
                                inpImg2, img2Keypoints.KeyPointsVec,
                                MatchesML,
                                outpImg);
                error = false;
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("Error drawing matches on an image: Exception: " << e.what());
            error = true;
        }
        return error;
    }


}
