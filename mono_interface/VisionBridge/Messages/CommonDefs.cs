/* ****************************************************************
 * Vision Bridge
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Aug. 2015
 * 
 * Developed under contract for:
 * Academy Center for UAS Research 
 * Department of Electrical and Computer Engineering 
 * HQ USAFA/DFEC 
 * 2354 Fairchild Drive 
 * USAF Academy, CO 80840-6236 
 *******************************************************************/
using System;
using ProtoBuf;

namespace VisionBridge.Messages
{
    /// <summary>
    /// ImageCaptureSource
    /// Must Match Proto-Buf and Videre definitions
    /// </summary>
    public enum ImageCaptureSource_e
    {
        NoChange,       //Don't change current config,
        OpenCVWebCam,    //WebCam from OpenCV Driver
        ImagePlusMetadataFiles,
        CompressedImages,    //A Directory of Compressed Images.
        Sensoray2253,        //Sensoray 2253
        NVidiaCSIcam,
        VideoFile
    }

    /// <summary>
    /// Image Capture Format
    /// These are WebCam dependent in terms of the webcam's 
    /// capabilities.
    /// </summary>
    public enum CPImageCaptureFormat_e
    {
        Unknown = 0,
        Grey8,
        Grey16,
        RGB24,
        BGR24,
        MJPEG,
        YUV422
    }


    public enum ImageCaptureError_e
    {
        None,       
        SourceConfig,    //Image Capture Source Config/Setup Error
        SourceCapture
    }

    public enum TargetProcessingMode_e
    {
        TgtProcMode_None = 0,
        TgtProcMode_Std = 1,        //Blob Target Detection using the OpenCV Blob Detector
        TgtProcMode_Blob = 2,       //Blob Target Detection using the Huntsman/JHART Blob Detector
        TgtProcMode_CheckerBoard = 3,
    }

    public enum
        VisionProcessingMode_e
    {
        VPM_None = 0,
        VPM_Target = 1,        //Target Processing... see TargetProcessingMode_e for specifics
        VPM_GPSDenied = 2,     //GPS Denied Image Processing
        VPM_CameraCalibration = 3,  //Camera Calibration
        VPM_FeatureMatchProc = 4,
        VPM_HeadTrackingProc = 5,
        VPM_HeadOrientationCalProc = 6,
    }

    public enum CameraCalibrationType_e
    {
        CCT_2DPlaneCheckerBoard = 0
    }

}
