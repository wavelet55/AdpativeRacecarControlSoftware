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

#include "CudaImageKeypointsAndFeatures.h"
#include <opencv2/features2d.hpp>
#include <opencv2/cudaimgproc.hpp>

using namespace cv;
using namespace std;
//using namespace cv::xfeatures2d;

namespace CudaImageProcLibsNS
{

    CudaImageKeypointsAndFeatures::CudaImageKeypointsAndFeatures(FeatureDetectorType_e fdt)
    {
        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        setFeatureDetectorType(fdt);
    }

    CudaImageKeypointsAndFeatures::~CudaImageKeypointsAndFeatures()
    {
        releaseAllResources();
    }

    //Close the feature detector and release memory objects.
    void CudaImageKeypointsAndFeatures::releaseAllResources()
    {
        releaseFeatureDetector();
        KeyPointsVec.clear();
        KeyPointsVec.shrink_to_fit();
        FeaturesMat.release();
    }

    //generate the Keypoints and Features for the Image
    //The first time this method is called, it will establish the feature
    //detector... subsequent calls will use the feature detector previously
    //established.
    //returns false if generated ok, true if there was an error
    //generating the keypoints and images.
    bool CudaImageKeypointsAndFeatures::generateKeypointsAndFeatures(const cv::Mat &img)
    {
        bool error = true;
        //cuda::

        if(!setupFeatureDetector())
        {
            try
            {
                KeyPointsVec.clear();
                _featureDetectorPtr->detectAndCompute(img, noArray(), KeyPointsVec, FeaturesMat);
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
    bool CudaImageKeypointsAndFeatures::setupFeatureDetector()
    {
        bool error = false;
        if(_featureDetectorPtr == nullptr)
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

                    if(_featureDetectorPtr == NULL)
                    {
                        LOGERROR("Error creating an ORB Feature Detector." );
                        error = true;
                    }
                }
                else if(_featureDetectorType == FeatureDetectorType_e::FDT_SIFT)
                {
                    _featureDetectorPtr = nullptr;
                    if(_featureDetectorPtr == nullptr)
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





}