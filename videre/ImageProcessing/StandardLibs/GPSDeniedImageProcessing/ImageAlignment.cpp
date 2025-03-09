/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Jan 2017
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
 * Routines taken from Dr. Dan Connor's work.
 * Original routines were written in Python
 *******************************************************************/

#include "ImageAlignment.h"
#include <iostream>


using namespace std;
using namespace cv;

namespace StdGPSDeniedImageProcessingNS
{

    ImageAlignment::ImageAlignment(int maxNumberOfFeatures)
    {
        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        _maxNumberOfFeatures = maxNumberOfFeatures;
        _featureDetector = ORB::create(maxNumberOfFeatures);

        _descriptorMatcher = BFMatcher::create(cv::NORM_HAMMING);
    }

    ImageAlignment::~ImageAlignment()
    {

    }

    void ImageAlignment::clearOldMatches()
    {
        for(size_t i = 0; i < _matches.capacity(); i++)
        {
            _matches[i].clear();
        }
        _matches.clear();
    }

    //Image Alignment Using ORB (Oriented FAST and Rotated BRIEF)
    //http://docs.opencv.org/2.4/modules/features2d/doc/feature_detection_and_description.html
    //img1:  input image 1 is the reference image.
    //img2:  input image 2
    //imgOut:  The ouptput image... a transformed image 2 which can be aligned with image 1.
    //          The user must provide a cv::Mat that the results can be put into.
    //maxNumberOfFeatures:  the maximum number of ORB keypoints used in the allignment.
    //returns false if ok... true if error in processing.
    bool ImageAlignment::ImageAlignmentUsingORB(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &imgOut)
    {
        bool error = true;
        cv::DMatch tmpM;
        try
        {
            _img1KeyPts.clear();
            _img2KeyPts.clear();
            clearOldMatches();
            _filteredMatches.clear();

            _featureDetector->detectAndCompute(img1, cv::noArray(), _img1KeyPts, _img1Descriptor);
            _featureDetector->detectAndCompute(img2, cv::noArray(), _img2KeyPts, _img2Descriptor);

            _descriptorMatcher->clear();
            _descriptorMatcher->knnMatch(_img1Descriptor, _img2Descriptor, _matches, 2);

            for(int i = 0; i < _matches.size(); i++)
            {
                if(_matches[i].size() == 2
                   && _matches[i][0].distance < 0.75 * _matches[i][1].distance)
                {
                    tmpM = _matches[i][0];
                    _filteredMatches.push_back(tmpM);
                }
            }


        }
        catch(std::exception &e)
        {
            LOGERROR("ImageAlignmentUsingORB: Exception: " << e.what());
            error = true;
        }

        return error;
    }





}