/* ****************************************************************
 * GPS Denied Image Processor
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
 *******************************************************************/


#ifndef VIDERE_DEV_GPSDENIEDIMAGEPROCESSOR_H
#define VIDERE_DEV_GPSDENIEDIMAGEPROCESSOR_H

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include "CommonImageProcTypesDefs.h"

using namespace ImageProcLibsNS;

namespace StdGPSDeniedImageProcessingNS
{

    class GPSDeniedImageProcessor
    {
    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        double _estimatedLatitudeRad;
        double _estimatedLongitudeRad;


    public:

        GPSDeniedImageProcessor();

        ~GPSDeniedImageProcessor();

        //Create a blob-detector and load with basic parameters
        bool Initialize();

        //Release all resources.
        void Close();

        //Clear / Reset process.
        void ClearReset();

        double GetEstimatedLatitudeRadians()
        {
            return _estimatedLatitudeRad;
        }

        double GetEstimatedLongitudeRadians()
        {
            return _estimatedLongitudeRad;
        }

        double GetEstimatedLatitudeDegrees()
        {
            return _estimatedLatitudeRad * (180.0 / MATH_CNST_PI);
        }

        double GetEstimatedLongitudeDegrees()
        {
            return _estimatedLongitudeRad * (180.0 / MATH_CNST_PI);
        }



    };

}

#endif //VIDERE_DEV_GPSDENIEDIMAGEPROCESSOR_H
