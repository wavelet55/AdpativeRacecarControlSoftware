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

#include "ImageKeypointsAndFeatures.h"
//#include <opencv2/xfeatures2d.hpp>
#ifdef CUDA
#include <opencv2/cudaimgproc.hpp>
#endif
#include <stdio.h>

using namespace cv;
using namespace std;
//using namespace cv::xfeatures2d;

namespace ImageProcLibsNS
{

    ImageKeypointsAndFeatures::ImageKeypointsAndFeatures(FeatureDetectorType_e fdt)
    {
        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        setFeatureDetectorType(fdt);

        //signal(SIGSEGV, segFaultHandler);
    }

    ImageKeypointsAndFeatures::~ImageKeypointsAndFeatures()
    {
        releaseAllResources();
    }


    //Close the feature detector and release memory objects.
    void ImageKeypointsAndFeatures::releaseAllResources()
    {
        releaseFeatureDetector();
        KeyPointsVec.clear();
        KeyPointsVec.shrink_to_fit();
        FeaturesMat.release();
        FeaturesAvailable = false;
    }

    //generate the Keypoints only... not the Features, for the Image
    //The first time this method is called, it will establish the feature
    //detector... subsequent calls will use the feature detector previously
    //established.
    //returns false if generated ok, true if there was an error
    //generating the keypoints and images.
    bool ImageKeypointsAndFeatures::generateKeypointsOnly(cv::InputArray img,
                                                          cv::InputArray mask)
    {
        bool error = true;
        FeaturesAvailable = false;
        if(!setupFeatureDetector())
        {
            try
            {
                KeyPointsVec.clear();
                _featureDetectorPtr->detect(img, KeyPointsVec, mask);
                error = false;
            }
            catch (std::exception &e)
            {
                LOGERROR("Error Generating Keypoints: Exception: " << e.what());
                error = true;
            }
        }
        return error;
    }


    //generate the Keypoints and Features for the Image
    //The first time this method is called, it will establish the feature
    //detector... subsequent calls will use the feature detector previously
    //established.
    //returns false if generated ok, true if there was an error
    //generating the keypoints and images.
    bool ImageKeypointsAndFeatures::generateKeypointsAndFeatures(cv::InputArray img,
                                                                 cv::InputArray mask)
    {
        bool error = true;
        FeaturesAvailable = false;
        if(!setupFeatureDetector())
        {
            try
            {
                KeyPointsVec.clear();
                _featureDetectorPtr->detectAndCompute(img, mask, KeyPointsVec, FeaturesMat);
                FeaturesAvailable = true;
                error = false;
            }
            catch (std::exception &e)
            {
                LOGERROR("Error Generating Keypoints And Features: Exception: " << e.what());
                error = true;
            }
        }
        return error;
    }

    //establish/setup the feature detector
    //Returns false if set up ok, true if there is an error setting
    //up the feture detector
    bool ImageKeypointsAndFeatures::setupFeatureDetector()
    {
        bool error = false;
        if(_featureDetectorPtr.empty())
        {
            try
            {
                if(_featureDetectorType == FeatureDetectorType_e::FDT_ORB)
                {
                     _featureDetectorPtr = ORB::create(_maxNumberOfFeatures,
                                                       (float)_scaleFactor,
                                                       _numberOfLevels,
                                                       _edgeThreashold,
                                                       _firstLevel,
                                                       _wta_k,
                                                       _scoreType,
                                                       _patchSize,
                                                       _fastThreshold );
                    if(_featureDetectorPtr.empty())
                    {
                        LOGERROR("Error creating an ORB Feature Detector." );
                        error = true;
                    }
                }
                else if(_featureDetectorType == FeatureDetectorType_e::FDT_SIFT)
                {
                    if(_featureDetectorPtr.empty())
                    {
                        LOGERROR("Error creating a SIFT Feature Detector... Currently Not supported" );
                        error = true;
                    }
                }
                else
                {
                    LOGERROR("Error unsupported feature detector type" );
                    error = true;
                }
            }
            catch (std::exception &e)
            {
                LOGERROR("Error creating a Feature Detector: Exception: " << e.what());
                error = true;
            }
        }
        return error;
    }

    //Modify the image by adding the keypoint locations to the image.
    bool ImageKeypointsAndFeatures::markImageWithKeypoints(cv::InputOutputArray img, bool drawRichKeypts)
    {
        bool error = false;
        try
        {
            cv::DrawMatchesFlags flags = cv::DrawMatchesFlags::DRAW_OVER_OUTIMG;
            if(drawRichKeypts) {
                flags |= cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS;
            }
            cv::drawKeypoints(img, KeyPointsVec, img, -1, flags);
        }
        catch (std::exception &e)
        {
            LOGERROR("Error drawing keypoints on an image: Exception: " << e.what());
            error = true;
        }
        return error;
    }

}
