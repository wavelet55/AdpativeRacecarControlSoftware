/* ****************************************************************
 * Standard Target Detector Vision Processor
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Feb. 2017
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
 * Notes:  The Image Sensor is a right-handed system with the x-axis
 * pointed to the front of the vehicle, the y-axis to the left (pilot/driver's
 * view) and z-axis is pointed down. The vehicle coordinate frame is
 * a left-handed system with the x-axis pointed to the front of the vehicle,
 * the y-axis to the right (pilot/driver's view) and z-axis is pointed down.
 *
 * The Intrinsic calibration matrix values fx and fy focal lengths,
 * along with the ofset values are typically given in pixels.
  *******************************************************************/

#ifndef VIDERE_DEV_CAMERACALIBRATIONPROCESSOR_H
#define VIDERE_DEV_CAMERACALIBRATIONPROCESSOR_H

#include "VisionProcessResources.h"
#include "VisionProcessorAbstract.h"
#include "TargetDetectorProcessControl.h"
#include "Utilities/JpgFileHandling.h"
#include "CameraCalibration/CameraCalibrationWith2DObjects.h"
#include "logger.h"
#include "CameraCalibrationData.h"

namespace videre
{

    class CameraCalibrationProcessor : public VisionProcessorAbstract
    {
    private:

        CameraCalibrationType_e _CalType;

        CameraCalibrationState_e _CalState;

        std::string _calImageDirectoryName;
        std::string _calImageBaseName;
        std::string _calDataDirectoryName;
        std::string _calDataFileName;
        ImageFormatType_e _imageFormatType;

        int _numberOfCalImages = 0;
        std::vector<boost::filesystem::path> _listOfCalImageFiles;

        bool _capturedImageOk = false;

        bool _capturedImageSentForView = false;

        int _numberSendCapturedImageTries = 0;

        ImagePlusMetadataMessage *_currImgMsgPtr = nullptr;

        VidereImageprocessing::JpgFileHandler _JpgFileHandler;

        CameraCalibrationNS::CameraCalibrationWith2DObjects _cameraCal2DObjects;

        ImageProcLibsNS::CameraCalibrationData  _cameraCalData;

    public:
        CameraCalibrationProcessor(VisionProcessResources* vpResources);

        ~CameraCalibrationProcessor();

        std::string CreateImageFilename(int fileIdx);

        void ReadCameraMountCorrectionParameters(const shared_ptr<CameraCalCommandMessage> calDataMsg);

        void WriteCalDataToFile();

        //Reset the Vision Processor.
        //Reset must return all resources obtained during operation...
        //such a image messages, target result messages and the like.
        virtual void Reset();

        //This is the primary call for running the Vision Processor
        //each time through the Manager's loop.
        virtual VPProcessResult_e ExecuteUnitOfWork();

    };

}
#endif //VIDERE_DEV_CAMERACALIBRATIONPROCESSOR_H
