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
#ifndef VIDERE_DEV_CSI_CAMERA_H
#define VIDERE_DEV_CSI_CAMERA_H

#include "ImageCaptureAClass.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
// #include <opencv2/imagecodecs/imgcodecs.hpp> //OpenCV 3.0 only
//#include <opencv2/videoio/videoio.hpp> //OpenCV 3.0 only
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "CameraParametersSetupMessage.h"


namespace videre
{

/***************************************************
//Capture images from a CSI using the OpenCV library.
Ref: http://petermoran.org/csi-cameras-on-tx2/
Example code for displaying gstreamer video from the CSI port of the Nvidia Jetson in OpenCV.
Created by Peter Moran on 7/29/17.
https://gist.github.com/peter-moran/742998d893cd013edf6d0c86cc86ff7f
***********************************************************/

    class CSI_Camera : public ImageCaptureAClass
    {
        std::unique_ptr<cv::VideoCapture> _capture_sptr; /* pull images from this */
        std::string _device;

        //Messages
        std::shared_ptr<CameraParametersSetupMessage> _CameraParametersSetupMsg_sptr;

        bool _imageCaptured = false;

    public:
        CSI_Camera(std::shared_ptr<ConfigData> config,
                                 std::shared_ptr<ImageCaptureControlMessage> imageCaptureControlMsg,
                                 std::shared_ptr<CameraParametersSetupMessage> cameraParametersSetupMsg)
                : ImageCaptureAClass(config)
        {
            _CameraParametersSetupMsg_sptr = cameraParametersSetupMsg;
        }

        ~CSI_Camera()
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

        //Generate the Format String required to open the NVidia CSI Camera.
        std::string get_tegra_pipeline(int width, int height, int fps)
        {
            return "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(width) + ", height=(int)" +
                   std::to_string(height) + ", format=(string)I420, framerate=(fraction)" + std::to_string(fps) +
                    "/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
        }

        uint32_t ToImageFormat(ImageCaptureFormat_e imgFmt);

        virtual bool PreCaptureImage();

        virtual ImageCaptureReturnType_e GetNextImage(ImagePlusMetadataMessage *imgPMetadataMsgPtr);

        //Close the Image Capture Process.
        virtual void Close();

    };

}

#endif //VIDERE_DEV_CSI_CAMERA_H
