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


#include "GPSDeniedImageProcessor.h"

namespace StdGPSDeniedImageProcessingNS
{

    GPSDeniedImageProcessor::GPSDeniedImageProcessor()
    {
        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);
        LOGINFO("GPS Denied Image Processor Created.")

        ClearReset();
    }

    GPSDeniedImageProcessor::~GPSDeniedImageProcessor()
    {

    }

    //Create a blob-detector and load with basic parameters
    bool GPSDeniedImageProcessor::Initialize()
    {
        bool error = true;

        return error;
    }

    //Release all resources.
    void GPSDeniedImageProcessor::Close()
    {

    }

    //Clear / Reset process.
    void GPSDeniedImageProcessor::ClearReset()
    {
        _estimatedLatitudeRad = 0;
        _estimatedLongitudeRad = 0;

    }


}