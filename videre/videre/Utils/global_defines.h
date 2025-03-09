/* ****************************************************************
 * Global Definitions used across Videre
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: July 2016
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
  *******************************************************************/

#ifndef VIDERE_DEV_GLOBAL_DEFINES_H_H
#define VIDERE_DEV_GLOBAL_DEFINES_H_H

#include <string>
#include <RabitManager.h>
#include <RabitNonBlockingSPSCQueue.h>
//#include <opencv2/core/core.hpp>

namespace videre
{
    //Define a Rabit Message Queue which contains shared pointers to messages.
    typedef Rabit::RabitNonBlockingSPSCQueue<std::shared_ptr<Rabit::RabitMessage>> RabitMsgSharedPtrSPSCQueue;

    //Define a Rabit Message Queue which contains pointers to messages.
    typedef Rabit::RabitNonBlockingSPSCQueue<Rabit::RabitMessage *> RabitMsgPtrSPSCQueue;

    //Define a Message Queue that can be used for Video/Image Stream Data.
    typedef Rabit::RabitNonBlockingSPSCQueue<std::shared_ptr<std::vector<unsigned char>>> RabitVideoStreamSPSCQueue;

    typedef Rabit::RabitMessageQueue<std::shared_ptr<Rabit::RabitMessage>> RabitMsgQueue;

//Manager Names and Enums
    enum VisionSystemManagers_e
    {
        VS_UnknownMgr = 0,
        VS_VidereSystemControlManager = 1,
        VS_HeadOrientationManager = 2,
        VS_VehicleStateManager = 3,
        VS_CommsManager = 4,
        VS_ImageCaptureManager = 5,
        VS_VisionProcessManager = 6,
        VS_StreamRecordManager = 7,
        VS_IMUCommManager = 8,
        VS_GPSManager = 9,
        VS_SipnPuffManager = 10,
        VS_VehicleActuatorInterfaceManager = 11,
        VS_RemoteControlManager = 12,
        VS_RobotArmManager = 13,
        VS_SystemInfoManager = 14,
        VS_DTX_IMU_InterfaceManager = 15,
        VS_EndOfManagers = 16
    };

    const int VisionSystemNumberOfManagers = 17;

    extern std::string VisionSystemManagerNames[];

    inline std::string GetVisionSystemManagerName(VisionSystemManagers_e mgr)
    {
        return VisionSystemManagerNames[(int) mgr];
    }

// Sensor Type
    enum SensorType_e
    {
        EO = 0,
        IR = 1
    };

    enum ImageSensorMode_e
    {
        ISM_RGB,
        ISM_YUV
    };


    //Top-Level Vision Processing Mode of Operation.
    enum VisionProcessingMode_e
    {
        VisionProcMode_None = 0,
        VisionProcMode_Target = 1,        //Target Processing... see TargetProcessingMode_e for specifics
        VisionProcMode_GPSDenied = 2,     //GPS Denied Image Processing
        VisionProcMode_CameraCalibration = 3,  //Camera Calibration
        VisionProcMode_FeatureMatchProc = 4,   //Feature matching Process.
        VisionProcMode_HeadTrackingProc = 5,   //Head Tracking Process.
        VisionProcMode_HeadOrientationCalProc = 6,   //Head Orientation Calibration Process.
    };

    //There are multiple different was to calibrate
    //the camera... this enum is the type of calibrations
    //supported.
    enum CameraCalibrationType_e
    {
       CameraCal_2DPlaneCheckerBoard = 0,   //Default
    };

#define CAMERA_CAL_FILE_EXT "ini"

    enum CameraCalibrationState_e
    {
        CCalState_Reset,
        CCalState_WaitForStart,
        CCalState_StreamImages,
        CCalState_ImageValidate,
        CCalState_ImageCapturedWait,
        CCalState_CalProcess,
        CCalState_CalComplete,
        CCalState_CalError
    };

    enum CameraCalCmd_e
    {
        CCalCmd_NullCmd = 0,            //Remain in current state
        CCalCmd_Reset = 1,              //Go To reset State
        CCalCmd_ClearImageSet = 2,      //Go to Reset after clearing directory of Images
        CCalCmd_StreamImages = 3,       //Stream Images and wait for Capture Image
        CCalCmd_CaptureImage = 4,       //Capture and verify image
        CCalCmd_SetImageOk = 5,         //Causes image to be stored... goto StreamImages
        CCalCmd_RejectImage = 6,        //Reject image and goto StreamImages
        CCalCmd_RunCalProcess = 7,       //Run Calibration against Image Set.
    };

    enum FeatureMatchingState_e
    {
        FMPState_Reset,
        FMPState_WaitForStart,
        FMPState_StreamImages,
        FMPState_ImageValidate,
        FMPState_ImageCapturedWait,
        FMPState_FMProcess,
        FMPState_FMComplete,
        FMPState_FMError
    };

    enum FeatureMatchingProcCmd_e
    {
        FMPCmd_NullCmd = 0,            //Remain in current state
        FMPCmd_Reset = 1,              //Go To reset State
        FMPCmd_ClearImageSet = 2,      //Go to Reset after clearing directory of Images
        FMPCmd_StreamImages = 3,       //Stream Images and wait for Capture Image
        FMPCmd_CaptureImage = 4,       //Capture and verify image
        FMPCmd_SetImageOk = 5,         //Causes image to be stored... goto StreamImages
        FMPCmd_RejectImage = 6,        //Reject image and goto StreamImages
        FMPCmd_RunImageProcess = 7,    //Run Image Prcess against Feature Set.
    };

    //A range of different feature extraction type routings can be
    //selected.
    enum FeatureExtractionTypeRoutine_e
    {
        FETR_ORB,
        FETR_SIFT,
        FETR_SURF,
    };

    //A range of different feature extraction type routings can be
    //selExtractionected.
    enum FeatureMatchTypeRoutine_e
    {
        FMTR_BruteForce,
        FMTR_FLANN,
    };

    enum FMImagePostProcessMethod_e
    {
        FMIPPM_None,
        FMIPPM_GenFeatureMap,    //Generate an image showing mapping between Obj Image and Test image
        FMIPPM_MarkObjectFoundRect,
        FMIPPM_MarkObjectFoundCircle,
        FMIPPM_GenFeatureMapAndMarkObjRect,
        FMIPPM_GenFeatureMapAndMarkObjCircle,

    };


    //Note: this matches the enum in the
    //protobuf messages:  CPImageCaptureFormat_e
    enum ImageCaptureFormat_e
    {
        Unknown = 0,
        Grey8 = 1,
        Grey16 = 2,
        RGB24 = 3,
        BGR24 = 4,
        MJPEG = 5,
        YUV422 = 6,
    };

    //These are the different types of target procesing
    //supported by the system
    enum TargetProcessingMode_e
    {
        TgtProcMode_None = 0,
        TgtProcMode_Std = 1,    //At this time == Blob using OpenCV
        TgtProcMode_Blob = 2,   //Blob using Inspecta S.L. Blob Lib.
        TgtProcMode_CheckerBoard = 3,
    };

    enum GPSDeniedProcessingMode_e
    {
        GpsDeniedMode_None = 0,
        GpsDeniedMode_Std = 1,
    };


    //Image Compression or Format Type
    enum ImageFormatType_e
    {
        ImgFType_Raw,
        ImgFType_JPEG,
    };

    enum ImageCaptureSource_e
    {
        ImageCaptureSource_NoChange,        //Don't change current source... use config value
        ImageCaptureSource_OpenCVWebCam,    //OpenCV Driver for WebCam
        ImageCaptureSource_IPMFiles,        //ImagePlusMetadata Files (.ipm)
        ImageCaptureSource_CImageFiles,     //Compressed Image Files (.jpg)
        ImageCaptureSource_Sensoray2253,
        ImageCaptureSource_NVidiaCSI,
        ImageCaptureSource_VideoFile,
    };


    enum ImageCaptureError_e
    {
        ImageCaptureError_None,
        ImageCaptureError_SourceConfig,
        ImageCaptureError_SourceCapture
    };

    /// <summary>
    /// Vision logging type.
    /// The specific types of logging Falcon Vision Handles.
    /// </summary>
    enum VisionLoggingType_e
    {
        LogMetaDataOnly = 0,
        LogRawImages = 1,
        LogCompressedImages = 2
    };

    extern bool AddDateTimeToDataLogDirectory;
    extern std::string DataLogDirectory;

    //Make a Manager Plus string name in the format:
    //  MgrName:strName
    std::string MakeManagerPlusStringName(VisionSystemManagers_e mgr, const std::string& strName);

    //Convert a manager name (case insensitive) to the enum value for the manager name.
    VisionSystemManagers_e GetVisionSystemManagerEnumFromName(const std::string& mgrName);

    enum LinearActuatorFunction_e
    {
        LA_Default,
        LA_Brake,
        LA_Accelerator,
    };

    enum LinearActuatorPullPushType_e
    {
        LAPP_Push,
        LAPP_Pull
    };

    inline LinearActuatorPullPushType_e GetLinearActuatorPullPushType(LinearActuatorFunction_e)
    {
        //Currently both acuators are pull type... start at positive 3 inches and pull
        //down to lower position values.
        return LinearActuatorPullPushType_e::LAPP_Pull;
    }

    enum SteeringTorqueMap_e
    {
        STM_Disable,    //Disable auto steering control
        STM_L1,
        STM_L2,
        STM_L3,
        STM_L4,
        STM_L5,
    };

    enum VidereSystemStates_e
    {
        VSS_Init,
        VSS_RemoteControl,
        VSS_ExternalMonitorControl,
        VSS_ManualDriverControl,
        VSS_HeadOrientationCal,
        VSS_HeadOrientationControl,
    };

    enum VidereSystemStatus_e
    {
        VSX_Ok,
        VSX_Error
    };

#define MAXNUMBERGLYPHMODELS 5

}

#endif //VIDERE_DEV_GLOBAL_DEFINES_H_H
