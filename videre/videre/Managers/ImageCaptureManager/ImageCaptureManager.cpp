/* ****************************************************************
 * Image Manager
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Randy Direen PhD
 * Date: May 2016
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
  *******************************************************************/

#include "ImageCaptureManager.h"
#include <exception>
#include <iostream>
#include <RabitWorkspace.h>
/* Log4Cxx is used for all of my logging */
#include <log4cxx/logger.h>

using namespace std;
using namespace cv;
using namespace Rabit;

namespace videre
{

    //The Image Manager is responsible for capturing images from the camera.
    //Metadata associated with the images are captured and metadata structures
    //filled in with the data.
    //The images plus metadata are piped over queues to image processing and
    //the StreamRecordManager.
    ImageCaptureManager::ImageCaptureManager(std::string name, std::shared_ptr<ConfigData> config)
    : ImageCaptureManagerWSRMgr(name)
    {
        this->SetWakeupTimeDelayMSec(250);
        _config_sptr = config;

        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        //Messages
        _imageCaptureControlMsg_sptr = std::make_shared<ImageCaptureControlMessage>();
        this->AddPublishSubscribeMessage("ImageCaptureControlMessage", _imageCaptureControlMsg_sptr);
        _imageCaptureControlMsg_sptr->Register_SomethingPublished(boost::bind(&ImageCaptureManager::WakeUpManagerEH, this));

        _imageCaptureControlStatusMsg_sptr = std::make_shared<ImageCaptureControlMessage>();
        this->AddPublishSubscribeMessage("ImageCaptureControlStatusMessage", _imageCaptureControlStatusMsg_sptr);

        _imageCaptureStatusMsg_sptr = std::make_shared<ImageCaptureStatusMessage>();
        this->AddPublishSubscribeMessage("ImageCaptureStatusMessage", _imageCaptureStatusMsg_sptr);

        _imageProcessControlMsg_sptr = std::make_shared<ImageProcessControlMessage>();
        this->AddPublishSubscribeMessage("ImageProcessControlMessage", _imageProcessControlMsg_sptr);
        _imageProcessControlMsg_sptr->Register_SomethingPublished(boost::bind(&ImageCaptureManager::WakeUpManagerEH, this));

        _streamRecordControlMsg_sptr = std::make_shared<StreamRecordImageControlMesssage>();
        this->AddPublishSubscribeMessage("StreamRecordImageControlMesssage", _streamRecordControlMsg_sptr);
        _streamRecordControlMsg_sptr->Register_SomethingPublished(boost::bind(&ImageCaptureManager::WakeUpManagerEH, this));

        _imageLoggingControlMsg = std::make_shared<ImageLoggingControlMessage>();
        this->AddPublishSubscribeMessage("ImageLoggingControlMessage", _imageLoggingControlMsg);
        _imageLoggingControlMsg->Register_SomethingPublished(boost::bind(&ImageCaptureManager::WakeUpManagerEH, this));

        _vehicleInertialStatesMsg_sptr = std::make_shared<VehicleInertialStatesMessage>();
        AddPublishSubscribeMessage("VehicleInertialStatesMessage", _vehicleInertialStatesMsg_sptr);

        _cameraOrientationMsg_sptr = std::make_shared<CameraOrientationMessage>();
        AddPublishSubscribeMessage("CameraOrientationMessage", _cameraOrientationMsg_sptr);

        _CameraParametersSetupMsg_sptr = std::make_shared<CameraParametersSetupMessage>();
        AddPublishSubscribeMessage("CameraParametersSetupMessage", _CameraParametersSetupMsg_sptr);

        _ImagePixelToRealWorldTranslator_sptr = std::make_shared<ImagePixelLocationToRealWorldLocation>();

        //Create the Image Capture Objects
        _openCVWebcamImageCapture_sptr = std::make_shared<OpenCVWebcamImageCapture>(config,
                                                                                    _imageCaptureControlMsg_sptr,
                                                                                    _CameraParametersSetupMsg_sptr);
        _ipmDataFileImageCapture_sptr = std::make_shared<ImagePlusMetaDataFileReader>(config, _imageCaptureControlMsg_sptr);
        _compressedDataFileImageCapture_sptr = std::make_shared<CompressedImageFileReader>(config, _imageCaptureControlMsg_sptr);
        _sensorayImageCapture_sptr = std::make_shared<Sensoray2253ImageCapture>(config, _imageCaptureControlMsg_sptr);

        _nvidiaCSIImageCapture_sptr = std::make_shared<CSI_Camera>(config,
                                                                     _imageCaptureControlMsg_sptr,
                                                                     _CameraParametersSetupMsg_sptr);
        _videoFileCapture_sptr = std::make_shared<VideoFileReader>(config, _imageCaptureControlMsg_sptr);

        //Set a Default Image Capture Object
        _currentImageCaptureObj_sptr = _openCVWebcamImageCapture_sptr;
    }

    void ImageCaptureManager::Initialize()
    {
        string cfgVal;
        LOGINFO("ImageCaptureManager: Initialization Started")

        _imageCaptureEnabled = false;
        _imageCaptureError = ImageCaptureError_e::ImageCaptureError_None;
        _imageCaptureSouceError = false;
        _endOfImages = false;

        _imageCaptureControlStatusMsg_sptr->Clear();

        _imageSensorType = SensorType_e::EO;
        cfgVal = _config_sptr->GetConfigStringLowerCaseValue("ImageSensor.SensorType", "eo");
        if(cfgVal == "ir")
        {
            _imageSensorType = SensorType_e::IR;
        }

        _imageSensorMode = ImageSensorMode_e::ISM_RGB;
        cfgVal = _config_sptr->GetConfigStringLowerCaseValue("ImageSensor.ImageSensorMode", "yuv");
        if(cfgVal == "yuv")
        {
            _imageSensorMode = ImageSensorMode_e::ISM_YUV;
        }

        _postProcessImages = _config_sptr->GetConfigBoolValue("flags.PostProcessImages", "true");

        //Initialize Image Capture Objects... Each object that is enabled is initialized with
        //the configed parameters...
        _imageCaptureSource = ImageCaptureSource_e::ImageCaptureSource_NoChange;
        //Ensure atleast one Capture object is set.
        _currentImageCaptureObj_sptr = _openCVWebcamImageCapture_sptr;

        bool ipmFileReaderEnabled = _config_sptr->GetConfigBoolValue("ImagePlusMetadataReader.IPMReaderEnabled", "false");
        if(ipmFileReaderEnabled)
        {
            if(!_ipmDataFileImageCapture_sptr->Initialize())
            {
                _currentImageCaptureObj_sptr = _ipmDataFileImageCapture_sptr;
                _imageCaptureSource = ImageCaptureSource_e::ImageCaptureSource_IPMFiles;
            }
            else
            {
                _imageCaptureSouceError = true;
                _imageCaptureError = ImageCaptureError_e::ImageCaptureError_SourceConfig;
            }
        }

        bool compressedFileReaderEnabled = _config_sptr->GetConfigBoolValue("CompressedImageFileReader.CImagesReaderEnabled", "false");
        if(compressedFileReaderEnabled)
        {
            if( !_compressedDataFileImageCapture_sptr->Initialize() )
            {
                _currentImageCaptureObj_sptr = _compressedDataFileImageCapture_sptr;
                _imageCaptureSource = ImageCaptureSource_e::ImageCaptureSource_CImageFiles;
                _imageCaptureSouceError = false;
            }
            else
            {
                _imageCaptureSouceError = true;
                _imageCaptureError = ImageCaptureError_e::ImageCaptureError_SourceConfig;
            }
        }

        bool sensoray2253Enabled = _config_sptr->GetConfigBoolValue("Sensoray.Sensoray2253Enabled", "false");
        if(sensoray2253Enabled)
        {
            if(!_sensorayImageCapture_sptr->Initialize())
            {
                _currentImageCaptureObj_sptr = _sensorayImageCapture_sptr;
                _imageCaptureSource = ImageCaptureSource_e::ImageCaptureSource_Sensoray2253;
                _imageCaptureSouceError = false;
            }
            else
            {
                _imageCaptureSouceError = true;
                _imageCaptureError = ImageCaptureError_e::ImageCaptureError_SourceConfig;
            }
        }

        bool nvidiaCSIcamEnabled = _config_sptr->GetConfigBoolValue("NVidiaCSIcam.NVidiaCSIcamEnabled", "false");
        if(nvidiaCSIcamEnabled)
        {
            if(!_nvidiaCSIImageCapture_sptr->Initialize())
            {
                _currentImageCaptureObj_sptr = _nvidiaCSIImageCapture_sptr;
                _imageCaptureSource = ImageCaptureSource_e::ImageCaptureSource_NVidiaCSI;
                _imageCaptureSouceError = false;
            }
            else
            {
                _imageCaptureSouceError = true;
                _imageCaptureError = ImageCaptureError_e::ImageCaptureError_SourceConfig;
            }
        }

        //Note:  the last Image Capture source in the config process will have the highest priority
        //and will be in effect when the system starts up.  This would normally be used by HOPS
        bool webcamEnabled = _config_sptr->GetConfigBoolValue("OpenCVWebcam.webcamEnabled", "true");
        if(webcamEnabled)
        {
            if(!_openCVWebcamImageCapture_sptr->Initialize())
            {
                _currentImageCaptureObj_sptr = _openCVWebcamImageCapture_sptr;
                _imageCaptureSource = ImageCaptureSource_e::ImageCaptureSource_OpenCVWebCam;
                _imageCaptureSouceError = false;
            }
            else
            {
                _imageCaptureSouceError = true;
                _imageCaptureError = ImageCaptureError_e::ImageCaptureError_SourceConfig;
            }
        }

        if(_imageCaptureSource == ImageCaptureSource_e::ImageCaptureSource_NoChange)
        {
            _imageCaptureSouceError = true;
            _imageCaptureError = ImageCaptureError_e::ImageCaptureError_SourceConfig;
            LOGERROR("ImageCaptureManger: No Image Capture Source Established.")
            cout << "ImageCaptureManger: No Image Capture Source Established" << endl;
        }

        _imageCaptureControlStatusMsg_sptr->CopyMessage(&_currentImageCaptureObj_sptr->ImageCaptureControlStatusMsg);
        _imageCaptureControlStatusMsg_sptr->PostMessage();

        try
        {
            //Setup the _ImagePixelToRealWorldTranslator with calibration factors
            _cameraCalDir = _config_sptr->GetConfigStringValue("CameraCal.CameraCalDataDirectory", "CameraCalData");
            _cameraCalFilename = _config_sptr->GetConfigStringValue("CameraCal.DefaultCameraCalFilename", "CameraCalData");
            bool readErr = _ImagePixelToRealWorldTranslator_sptr->ReadCameraCalDataFromFile(_cameraCalDir, _cameraCalFilename);
            if(readErr)
            {
                //Try reading from the

            }

            //The Image/Vision Processing Manager Queue
            _VissionProcMgrImagePlusMetadataQueue_sptr = this->GetManagerMessageQueue<std::shared_ptr<RabitMsgPtrSPSCQueue>>(
                    "VissionProcMgrRxImagePlusMetadataMsgQueue");
            _VissionProcMgrEmptyImagePlusMetadataQueue_sptr = this->GetManagerMessageQueue<std::shared_ptr<RabitMsgPtrSPSCQueue>>(
                    "VissionProcMgrEmptyImagePlusMetadataMsgQueue");
            RabitWorkspace::GetWorkspace()->Register_EnqueuedEvent<RabitMessage*>(
                    "VissionProcMgrEmptyImagePlusMetadataMsgQueue",
                    boost::bind(&ImageCaptureManager::WakeUpManagerEH, this));
            _VisionProcessMgrAvailable = true;
        }
        catch (MessageNotRegisteredException &e)
        {
            LOGWARN("ImageCaptureManager: The VissionProcMgrImagePlusMetadataQueue does not exist.");
            cout << "ImageCaptureManager: The VissionProcMgrImagePlusMetadataQueue does not exist." << endl;
            _VissionProcMgrImagePlusMetadataQueue_sptr = nullptr;
            _VissionProcMgrEmptyImagePlusMetadataQueue_sptr = nullptr;
            _VisionProcessMgrAvailable = false;
        }

        try
        {
            //The StreamRecordManager Queue
            _StreamMgrImagePlusMetadataQueue_sptr = this->GetManagerMessageQueue<std::shared_ptr<RabitMsgPtrSPSCQueue>>(
                    "StreamRecordRxIPMDMsgQueue");
            _StreamMgrEmptyImagePlusMetadataQueue_sptr = this->GetManagerMessageQueue<std::shared_ptr<RabitMsgPtrSPSCQueue>>(
                    "StreamRecordEmptyIPMDMsgQueue");
            RabitWorkspace::GetWorkspace()->Register_EnqueuedEvent<RabitMessage*>(
                    "StreamRecordEmptyIPMDMsgQueue",
                    boost::bind(&ImageCaptureManager::WakeUpManagerEH, this));
            _StreamMgrAvailable = true;
        }
        catch (MessageNotRegisteredException &e)
        {
            LOGWARN("ImageCaptureManager: The StreamMgrRecordImageQueue message queue does not exist.");
            cout << "ImageCaptureManager: The StreamMgrRecordImageQueue message queue does not exist." << endl;
            _StreamMgrImagePlusMetadataQueue_sptr = nullptr;
            _StreamMgrEmptyImagePlusMetadataQueue_sptr = nullptr;
            _StreamMgrAvailable = false;
        }

        LOGINFO("ImageCaptureManager: Initialization Complete")
        std::cout << "ImageCaptureManager: Initialization Complete" << std::endl;
    }

    bool ImageCaptureManager::SetupImageCaptureSource(std::shared_ptr<ImageCaptureControlMessage> imageCaptureControlMsg)
    {
        bool error;
        if(imageCaptureControlMsg->ImageCaptureSource != ImageCaptureSource_e::ImageCaptureSource_NoChange)
        {
            _imageCaptureSouceError = true;
            //First close the current source.
            switch (_imageCaptureSource)
            {
                case ImageCaptureSource_e::ImageCaptureSource_OpenCVWebCam:
                    _openCVWebcamImageCapture_sptr->Close();
                    break;
                case ImageCaptureSource_e::ImageCaptureSource_IPMFiles:
                    _ipmDataFileImageCapture_sptr->Close();
                    break;
                case ImageCaptureSource_e::ImageCaptureSource_CImageFiles:
                    _compressedDataFileImageCapture_sptr->Close();
                    break;
                case ImageCaptureSource_e::ImageCaptureSource_Sensoray2253:
                    _sensorayImageCapture_sptr->Close();
                case ImageCaptureSource_e::ImageCaptureSource_NVidiaCSI:
                    _nvidiaCSIImageCapture_sptr->Close();
                    break;
                case ImageCaptureSource_e::ImageCaptureSource_VideoFile:
                    _videoFileCapture_sptr->Close();
                    break;

            }
            //Now setup the new source
            switch (imageCaptureControlMsg->ImageCaptureSource)
            {
                case ImageCaptureSource_e::ImageCaptureSource_OpenCVWebCam:
                    if(imageCaptureControlMsg->ImageCaptureSourceConfigPri.size() > 0)
                    {
                        error = _openCVWebcamImageCapture_sptr->Initialize(*imageCaptureControlMsg.get());
                     }
                    else
                    {
                        //Use Config File Parameters.
                        error = _openCVWebcamImageCapture_sptr->Initialize();
                    }
                    if(!error)
                    {
                        _currentImageCaptureObj_sptr = _openCVWebcamImageCapture_sptr;
                        _imageCaptureSource = ImageCaptureSource_e::ImageCaptureSource_OpenCVWebCam;
                        _imageCaptureSouceError = false;

                    }
                    _imageCaptureControlStatusMsg_sptr->CopyMessage(&_openCVWebcamImageCapture_sptr->ImageCaptureControlStatusMsg);
                    break;
                case ImageCaptureSource_e::ImageCaptureSource_NVidiaCSI:
                    if(imageCaptureControlMsg->ImageCaptureSourceConfigPri.size() > 0)
                    {
                        error = _nvidiaCSIImageCapture_sptr->Initialize(*imageCaptureControlMsg.get());
                    }
                    else
                    {
                        //Use Config File Parameters.
                        error = _nvidiaCSIImageCapture_sptr->Initialize();
                    }
                    if(!error)
                    {
                        _currentImageCaptureObj_sptr = _nvidiaCSIImageCapture_sptr;
                        _imageCaptureSource = ImageCaptureSource_e::ImageCaptureSource_NVidiaCSI;
                        _imageCaptureSouceError = false;

                    }
                    _imageCaptureControlStatusMsg_sptr->CopyMessage(&_nvidiaCSIImageCapture_sptr->ImageCaptureControlStatusMsg);
                    break;
                case ImageCaptureSource_e::ImageCaptureSource_IPMFiles:
                    if(imageCaptureControlMsg->ImageCaptureSourceConfigPri.size() > 0)
                    {
                        error = _ipmDataFileImageCapture_sptr->Initialize(*imageCaptureControlMsg.get());
                    }
                    else
                    {
                        //Use Config File Parameters.
                        _ipmDataFileImageCapture_sptr->Initialize();
                    }
                    if(!error)
                    {
                        _currentImageCaptureObj_sptr = _ipmDataFileImageCapture_sptr;
                        _imageCaptureSource = ImageCaptureSource_e::ImageCaptureSource_IPMFiles;
                        _imageCaptureSouceError = false;
                    }
                    _imageCaptureControlStatusMsg_sptr->CopyMessage(&_ipmDataFileImageCapture_sptr->ImageCaptureControlStatusMsg);
                    break;
                case ImageCaptureSource_e::ImageCaptureSource_CImageFiles:
                    if(imageCaptureControlMsg->ImageCaptureSourceConfigPri.size() > 0)
                    {
                        error = _compressedDataFileImageCapture_sptr->Initialize(*imageCaptureControlMsg.get());
                    }
                    else
                    {
                        //Use Config File Parameters.
                        _compressedDataFileImageCapture_sptr->Initialize();
                    }
                    if(!error)
                    {
                        _currentImageCaptureObj_sptr = _compressedDataFileImageCapture_sptr;
                        _imageCaptureSource = ImageCaptureSource_e::ImageCaptureSource_CImageFiles;
                        _imageCaptureSouceError = false;
                    }
                    _imageCaptureControlStatusMsg_sptr->CopyMessage(&_compressedDataFileImageCapture_sptr->ImageCaptureControlStatusMsg);
                    break;
                case ImageCaptureSource_e::ImageCaptureSource_Sensoray2253:
                    if(imageCaptureControlMsg->ImageCaptureSourceConfigPri.size() > 0)
                    {
                        error = _sensorayImageCapture_sptr->Initialize(*imageCaptureControlMsg.get());
                    }
                    else
                    {
                        //Use Config File Parameters.
                        error = _sensorayImageCapture_sptr->Initialize();
                    }
                    if(!error)
                    {
                        _currentImageCaptureObj_sptr = _sensorayImageCapture_sptr;
                        _imageCaptureSource = ImageCaptureSource_e::ImageCaptureSource_Sensoray2253;
                        _imageCaptureSouceError = false;
                    }
                    _imageCaptureControlStatusMsg_sptr->CopyMessage(&_sensorayImageCapture_sptr->ImageCaptureControlStatusMsg);
                     break;

                case ImageCaptureSource_e::ImageCaptureSource_VideoFile:
                    if(imageCaptureControlMsg->ImageCaptureSourceConfigPri.size() > 0
                            && imageCaptureControlMsg->ImageCaptureSourceConfigSec.size() > 0)
                    {
                        error = _videoFileCapture_sptr->Initialize(*imageCaptureControlMsg.get());
                    }
                    else
                    {
                        //Use Config File Parameters.
                        error = _videoFileCapture_sptr->Initialize();
                    }
                    if(!error)
                    {
                        _currentImageCaptureObj_sptr = _videoFileCapture_sptr;
                        _imageCaptureSource = ImageCaptureSource_e::ImageCaptureSource_VideoFile;
                        _imageCaptureSouceError = false;
                    }
                    _imageCaptureControlStatusMsg_sptr->CopyMessage(&_compressedDataFileImageCapture_sptr->ImageCaptureControlStatusMsg);
                    break;

            }
            _imageCaptureControlStatusMsg_sptr->PostMessage();
            this->SetWakeupTimeDelayMSec(_currentImageCaptureObj_sptr->GetDesiredImgCaptureMgrLoopTimeMSec());
        }
        else
        {
            LOGWARN("ImageCaptureManager::SetupImageCaptureSource: attempted setup with a No Change Source.")
        }
        if(_imageCaptureSouceError)
        {
            _imageCaptureError = ImageCaptureError_e::ImageCaptureError_SourceConfig;
        }
        return _imageCaptureSouceError;
    }

    unsigned int ImageCaptureManager::GetNumberOfImagesCaptured()
    {
        unsigned int N = 0;
        try
        {
            N = _currentImageCaptureObj_sptr->GetNumberOfImagesCaptured();
        }
        catch (std::exception &e)
        {
            LOGERROR("GetNumberOfImagesCaptured Exception: " << e.what());
        }

        return N;
    }

    void ImageCaptureManager::ClearNumberOfImagesCaptured()
    {
        try
        {
            _currentImageCaptureObj_sptr->ClearNumberOfImagesCaptured();
        }
        catch (std::exception &e)
        {
            LOGERROR("ClearNumberOfImagesCaptured Exception: " << e.what());
        }
    }

    bool ImageCaptureManager::PreCaptureImage()
    {
        bool imageCaptured = false;
        try
        {
            imageCaptured = _currentImageCaptureObj_sptr->PreCaptureImage();
        }
        catch (std::exception &e)
        {
            LOGERROR("Pre-Capture Image Exception: " << e.what());
            imageCaptured = false;
        }
        return imageCaptured;
    }

    //Get a new image and fill in the metadata.
    //Return true if image and metadata obtained ok,
    //otherwise return false.
    bool ImageCaptureManager::GetNewImage(ImagePlusMetadataMessage *imgPMetadataMsgPtr)
    {
        bool imageCaptured = false;
        ImageCaptureReturnType_e icrType;
        double deltaTimeSec = 0;
        try
        {

            icrType = _currentImageCaptureObj_sptr->GetNextImage(imgPMetadataMsgPtr);
            switch(icrType)
            {
                case ImageCaptureReturnType_e::ICRT_Error:
                    imageCaptured = false;
                    break;
                case ImageCaptureReturnType_e::ICRT_NoImageCaptured:
                    imageCaptured = false;
                    break;
                case ImageCaptureReturnType_e::IRCT_EndOfImages:
                    imageCaptured = false;
                    _endOfImages = true;
                    _imageCaptureStatusMsg_sptr->EndOfImages = true;
                    break;
                case ImageCaptureReturnType_e::IRCT_ImagePlusMetadata:
                    imageCaptured = true;
                    _endOfImages = false;
                    _imageCaptureStatusMsg_sptr->EndOfImages = false;
                    break;
                case ImageCaptureReturnType_e::IRCT_ImageOnly:
                    imageCaptured = true;
                    _endOfImages = false;
                    _imageCaptureStatusMsg_sptr->EndOfImages = false;

                    //ToDo:  Fill in the image metadata
                    _vehicleInertialStatesMsg_sptr->FetchMessage();
                    //ToDo:  get timestamp and move states forward in time;
                    deltaTimeSec = imgPMetadataMsgPtr->ImageCaptureTimeStampSec - _vehicleInertialStatesMsg_sptr->GpsTimeStampSec;
                    if( fabs(deltaTimeSec) < 1.5 )
                    {
                        //Only adjust the vehicle states if the time stamps are close in time...
                        //otherwise something is wrong with the time stamps and this would likely
                        //result in poor if not completely wrong advance in time.
                        _vehicleInertialStatesMsg_sptr->MoveStatesForwardInTime(deltaTimeSec);
                    }
                    imgPMetadataMsgPtr->VehicleInertialStates = *_vehicleInertialStatesMsg_sptr;

                    _cameraOrientationMsg_sptr->FetchMessage();

                    imgPMetadataMsgPtr->CameraOrientation = *_cameraOrientationMsg_sptr;

                    if (_imageProcessControlMsg_sptr->GPUProcessingEnabled)
                    {
                        //Copy Image into the Cuda memory space... this only works if
                        //the computer supports NVidia Cuda
                        imgPMetadataMsgPtr->CopyImageToCudaMat();
                    }
                    break;
                case ImageCaptureReturnType_e::IRCT_MetadataOnly:
                    imageCaptured = false;
                    _endOfImages = false;
                    _imageCaptureStatusMsg_sptr->EndOfImages = false;
                    break;
            }
            _imageCaptureStatusMsg_sptr->TotalNumberOfImagesCaptured = _currentImageCaptureObj_sptr->GetTotalNumberOfImagesCaptured();
            _imageCaptureStatusMsg_sptr->CurrentNumberOfImagesCaptured = _currentImageCaptureObj_sptr->GetNumberOfImagesCaptured();

        }
        catch (std::exception &e)
        {
            LOGERROR("Capture Image Exception: " << e.what());
            imageCaptured = false;
            //Set the Image number to zero which indicates the image is
            //invalid.  This can be used by other processes to ignore this
            //image message.
            imgPMetadataMsgPtr->ImageNumber = 0;
        }
        return imageCaptured;
    }

    bool ImageCaptureManager::SetupPixelToRealWorldTranslatorCalImageCorners(ImagePlusMetadataMessage *imgPMetadataMsgPtr,
                                                                        bool useLocalTranslator)
    {
        bool error = false;
        AzimuthElevation_t aeAngles;
        std::shared_ptr<ImagePixelLocationToRealWorldLocation> pixelToRWTrx;
        try
        {
            if( useLocalTranslator )
            {
                pixelToRWTrx = _ImagePixelToRealWorldTranslator_sptr;
            }
            else
            {
                if( imgPMetadataMsgPtr->PixelToRWTranslator == nullptr)
                {
                    //If the ImagePlusMetadataMessage does not have a translator... make one
                    imgPMetadataMsgPtr->PixelToRWTranslator = std::make_shared<ImagePixelLocationToRealWorldLocation>();
                }
                pixelToRWTrx = imgPMetadataMsgPtr->PixelToRWTranslator;
            }
            //ToDo: we need to check for other changes that require
            //Calibration parameters to be resetup... such as a change in file name,
            //or other parameters.
            if(!pixelToRWTrx->CalibrationIsSetup )
            {
                bool readErr = pixelToRWTrx->ReadCameraCalDataFromFile(_cameraCalDir, _cameraCalFilename);
            }

            XYZCoord_t uavLoc(imgPMetadataMsgPtr->VehicleInertialStates.XYZCoordinates);
            uavLoc.z = imgPMetadataMsgPtr->VehicleInertialStates.HeightAGL;
            pixelToRWTrx->SetVehicleAndCameraLocation(uavLoc,
                                                      imgPMetadataMsgPtr->VehicleInertialStates.RollPitchYaw,
                                                      imgPMetadataMsgPtr->CameraOrientation.CameraAzimuthElevationAngles);

            //Calculate Image Corner Locations.
            pixelToRWTrx->CalculateRealWorldLocation(0,
                                                     0,
                                                     &(imgPMetadataMsgPtr->ImageCorners[0]),
                                                     &aeAngles);
            pixelToRWTrx->CalculateRealWorldLocation(imgPMetadataMsgPtr->ImageNoPixelsWide - 1,
                                                     0,
                                                     &(imgPMetadataMsgPtr->ImageCorners[1]),
                                                     &aeAngles);
            pixelToRWTrx->CalculateRealWorldLocation(imgPMetadataMsgPtr->ImageNoPixelsWide - 1,
                                                     imgPMetadataMsgPtr->ImageNoPixelsHigh - 1,
                                                     &(imgPMetadataMsgPtr->ImageCorners[2]),
                                                     &aeAngles);
            pixelToRWTrx->CalculateRealWorldLocation(0,
                                                     imgPMetadataMsgPtr->ImageNoPixelsHigh - 1,
                                                     &(imgPMetadataMsgPtr->ImageCorners[3]),
                                                     &aeAngles);

        }
        catch (std::exception &e)
        {
            LOGERROR("SetupPixelToRealWorldTranslatorCalImageCorners Exception: " << e.what());
            error = true;
        }
        return error;
    }


    void ImageCaptureManager::ExecuteUnitOfWork()
    {
        Rabit::RabitMessage *tmpMsgPtr;
        ImagePlusMetadataMessage tempImageMsg;
        ImagePlusMetadataMessage *imgPMetadataMsg1Ptr = nullptr;
        ImagePlusMetadataMessage *imgPMetadataMsg2Ptr = nullptr;
        _imageProcessControlMsg_sptr->FetchMessage();
        _streamRecordControlMsg_sptr->FetchMessage();
        _imageLoggingControlMsg->FetchMessage();

        bool cmdChanged = _imageCaptureControlMsg_sptr->FetchMessage();
        if(cmdChanged)
        {
            if(_imageCaptureControlMsg_sptr->ImageCaptureEnabled)
            {
                //No updates to Image capture configureations can change
                //when image capture is enabled.
                //Don't Enable if there is a current un-resolved error.
                if(!_imageCaptureSouceError)
                {
                    if (_imageCaptureControlMsg_sptr->ImageCaptureEnabled != _imageCaptureEnabled)
                    {
                        //Changed from disabled to enabled.
                        _imageCaptureEnabled = true;
                        //Rest the Image Capture Counter.
                        ClearNumberOfImagesCaptured();
                    }
                }
            }
            else
            {
                _imageCaptureEnabled = false;
                if(_imageCaptureControlMsg_sptr->ImageCaptureSource != ImageCaptureSource_e::ImageCaptureSource_NoChange)
                {
                    //Change the Image Capture Source.
                    //Clear the current errors.
                    _endOfImages = false;
                    _imageCaptureSouceError = false;
                    _imageCaptureError = ImageCaptureError_e::ImageCaptureError_None;
                    SetupImageCaptureSource(_imageCaptureControlMsg_sptr);
                 }
            }
        }

        try
        {
            //Used to keep webcam buffer from filling up when images are
            //not being processed or processed fast enough.
            PreCaptureImage();

            //Update the
            _imageCaptureStatusMsg_sptr->ImageCaptureEnabled = _imageCaptureEnabled;
            _imageCaptureStatusMsg_sptr->ImageCaptureSource = _imageCaptureSource;

            _imageCaptureComplete = _imageCaptureControlMsg_sptr->NumberOfImagesToCapture > 0
                                    &&  (GetNumberOfImagesCaptured() >= _imageCaptureControlMsg_sptr->NumberOfImagesToCapture);
            _imageCaptureStatusMsg_sptr->ImageCaptureComplete = _imageCaptureComplete;

            if( _imageCaptureEnabled  && !_imageCaptureSouceError && !_imageCaptureComplete && !_endOfImages)
            {

                if (_VisionProcessMgrAvailable &&
                        (_imageProcessControlMsg_sptr->TargetImageProcessingEnabled
                        || _imageProcessControlMsg_sptr->GPSDeniedProcessingEnabled))
                {
                    //This is the primary path if the Vision Process Manager is available and enabled.
                    //The image capture rate will be established by the Vision Processor.
                    //It is assumed that the Record/Stream process is keeping up or is ahead of the
                    //Vision Processor.  More logic may be required later to wait for the slowest manager.
                    if (_VissionProcMgrEmptyImagePlusMetadataQueue_sptr->GetMessage(tmpMsgPtr))
                    {
                        imgPMetadataMsg1Ptr = static_cast<ImagePlusMetadataMessage *>(tmpMsgPtr);
                        if (GetNewImage(imgPMetadataMsg1Ptr))
                        {
                            SetupPixelToRealWorldTranslatorCalImageCorners(imgPMetadataMsg1Ptr, false);
                            if (_StreamMgrAvailable && !_postProcessImages
                                && (_streamRecordControlMsg_sptr->StreamImagesEnabled
                                    || _imageLoggingControlMsg->EnableLogging))
                            {
                                //Check to see if there is an available ImagePlusMetadataMessage from the Stream Mgr.
                                if (_StreamMgrEmptyImagePlusMetadataQueue_sptr->GetMessage(tmpMsgPtr))
                                {
                                    imgPMetadataMsg2Ptr = static_cast<ImagePlusMetadataMessage *>(tmpMsgPtr);
                                    //Make a Copy of the Image message
                                    imgPMetadataMsg2Ptr->CopyMessage(imgPMetadataMsg1Ptr);
                                    _StreamMgrImagePlusMetadataQueue_sptr->AddMessage(imgPMetadataMsg2Ptr);
                                }
                            }

                            //Send the image to the Vision Process Manager.
                            //Note:  don't send the image message until the above copy is done.
                            _VissionProcMgrImagePlusMetadataQueue_sptr->AddMessage(imgPMetadataMsg1Ptr);
                        }
                        else
                        {
                            //We have a invalid image... but it is imperative that we send the
                            //Image message back to the Vision Manager... or we will run out of
                            //image messages.
                            _VissionProcMgrImagePlusMetadataQueue_sptr->AddMessage(imgPMetadataMsg1Ptr);
                        }
                    }
                }
                else if (_StreamMgrAvailable && !_postProcessImages
                        && (_streamRecordControlMsg_sptr->StreamImagesEnabled
                            || _imageLoggingControlMsg->EnableLogging))
                {
                    if (_StreamMgrEmptyImagePlusMetadataQueue_sptr->GetMessage(tmpMsgPtr))
                    {
                        imgPMetadataMsg2Ptr = static_cast<ImagePlusMetadataMessage *>(tmpMsgPtr);
                        if (GetNewImage(imgPMetadataMsg2Ptr))
                        {
                            SetupPixelToRealWorldTranslatorCalImageCorners(imgPMetadataMsg2Ptr, true);
                        }
                        //Send the message back to the Stream Manager even if it is bad,
                        //otherwise we will run out of the messages.
                        _StreamMgrImagePlusMetadataQueue_sptr->AddMessage(imgPMetadataMsg2Ptr);
                    }
                    else
                    {
                        //ToDo: Test only to check Image Capture Rate  (Remove after Tests)
                        //Simply get and image and toss it to see what the image capture rate is.
                        GetNewImage(&tempImageMsg);
                        if (tempImageMsg.ImageNumber % 30 == 0)
                        {
                            std::cout << "Image Capture Number: " << tempImageMsg.ImageNumber << std::endl;
                        }
                    }
                }
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("ImageCaptureManager: Exception: " << e.what());
        }
        _imageCaptureStatusMsg_sptr->ErrorCode = _imageCaptureError;
        _imageCaptureStatusMsg_sptr->PostMessage();
    }


}
