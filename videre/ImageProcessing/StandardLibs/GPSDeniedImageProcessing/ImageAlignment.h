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


#ifndef VIDERE_DEV_IMAGEALIGNMENT_H
#define VIDERE_DEV_IMAGEALIGNMENT_H

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include "CommonImageProcTypesDefs.h"
#include <vector>

using namespace ImageProcLibsNS;

namespace StdGPSDeniedImageProcessingNS
{




    class ImageAlignment
    {
    protected:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        std::vector<cv::KeyPoint> _img1KeyPts, _img2KeyPts;
        cv::Mat _img1Descriptor, _img2Descriptor;
        cv::Ptr<cv::Feature2D> _featureDetector;
        cv::Ptr<cv::DescriptorMatcher> _descriptorMatcher;
        std::vector< std::vector<cv::DMatch> > _matches;
        std::vector<cv::DMatch> _filteredMatches;

        int _maxNumberOfFeatures = 1000;

    public:
        ImageAlignment(int maxNumberOfFeatures = 1000);

        ~ImageAlignment();



        //Image Alignment Using ORB (Oriented FAST and Rotated BRIEF)
        //http://docs.opencv.org/2.4/modules/features2d/doc/feature_detection_and_description.html
        //img1:  input image 1 is the reference image.
        //img2:  input image 2
        //imgOut:  The ouptput image... a transformed image 2 which can be aligned with image 1.
        //          The user must provide a cv::Mat that the results can be put into.
        //maxNumberOfFeatures:  the maximum number of ORB keypoints used in the allignment.
        //returns false if ok... true if error in processing.
        bool ImageAlignmentUsingORB(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &imgOut);


    private:
        void clearOldMatches();

    };

}
#endif //VIDERE_DEV_IMAGEALIGNMENT_H
