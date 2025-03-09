/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Oct 2016
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
  *******************************************************************/
#ifndef VIDERE_DEV_OPENCVWEBCAMIMAGECAPTURE_H
#define VIDERE_DEV_OPENCVWEBCAMIMAGECAPTURE_H

#include "ImageCaptureAClass.h"
#include <opencv2/core.hpp>

/*******************************
 * OpenCV Set Parameters:
 *
 * Sets a property in the VideoCapture.

   set(propID, double value)

Parameters
propId	Property identifier. It can be one of the following:
CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
CAP_PROP_POS_AVI_RATIO Relative position of the video file: 0 - start of the film, 1 - end of the film.
CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
CAP_PROP_FPS Frame rate.
CAP_PROP_FOURCC 4-character code of codec.
CAP_PROP_FRAME_COUNT Number of frames in the video file.
CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
CAP_PROP_MODE Backend-specific value indicating the current capture mode.
CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
CAP_PROP_CONTRAST Contrast of the image (only for cameras).
CAP_PROP_SATURATION Saturation of the image (only for cameras).
CAP_PROP_HUE Hue of the image (only for cameras).
CAP_PROP_GAIN Gain of the image (only for cameras).
CAP_PROP_EXPOSURE Exposure (only for cameras).
CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
CAP_PROP_WHITE_BALANCE Currently unsupported
CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)
value	Value of the property.
 *
 *
 *
 ************************************************************************/


#include "CameraParametersSetupMessage.h"


namespace videre
{

//Capture images from a webcam using the OpenCV
//library.
    class OpenCVWebcamImageCapture : public ImageCaptureAClass
    {
        std::unique_ptr<cv::VideoCapture> _capture_sptr; /* pull images from this */
        std::string _device;

        //Messages
        std::shared_ptr<CameraParametersSetupMessage> _CameraParametersSetupMsg_sptr;

        bool _imageCaptured = false;

    public:
        OpenCVWebcamImageCapture(std::shared_ptr<ConfigData> config,
                                 std::shared_ptr<ImageCaptureControlMessage> imageCaptureControlMsg,
                                 std::shared_ptr<CameraParametersSetupMessage> cameraParametersSetupMsg)
            : ImageCaptureAClass(config)
        {
            _CameraParametersSetupMsg_sptr = cameraParametersSetupMsg;
        }

        ~OpenCVWebcamImageCapture()
        {
            Close();
        }

        std::string GetWebcamDevice()
        {
            return _device;
        }

        //Initalize the Image Capture process.
        //Returns true if error, false if ok.
        bool Initialize(const std::string &webcamDevice,
                        int desiredImageWidth,
                        int desiredImageHeight,
                        ImageCaptureFormat_e imgFmt,
                        bool autofocusEnable);

        //Initalize the Image Capture process.
        //Returns true if error, false if ok.
        virtual bool Initialize();

        //Initalize the Image Capture process.
        //Returns true if error, false if ok.
        virtual bool Initialize(ImageCaptureControlMessage &imgCapCtrlMsg);

        uint32_t ToImageFormat(ImageCaptureFormat_e imgFmt);

        virtual bool PreCaptureImage();

        virtual ImageCaptureReturnType_e GetNextImage(ImagePlusMetadataMessage *imgPMetadataMsgPtr);

        //Close the Image Capture Process.
        virtual void Close();

    };

}
#endif //VIDERE_DEV_OPENCVWEBCAMIMAGECAPTURE_H

/*********************
NVidia TX2 Camera setup:
https://stackoverflow.com/questions/36659151/how-to-receive-images-from-jetson-tx1-embedded-camera
 cap = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)I420,
 framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)I420 ! videoconvert ! video/x-raw,
 format=(string)BGR ! appsink")

 http://petermoran.org/csi-cameras-on-tx2/

 *************************/
