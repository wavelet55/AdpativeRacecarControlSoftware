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

#include "CSI_Camera.h"
#include <linux/videodev2.h>
#include <time.h>                   // needed for struct tm
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace chrono;

namespace videre {

    //Initalize the Image Capture process.
    //Returns true if error, false if ok.
    bool CSI_Camera::Initialize() {
        ImageCaptureControlMessage imgCapCtrlMsg;
        imgCapCtrlMsg.ImageCaptureSource = ImageCaptureSource_e::ImageCaptureSource_NVidiaCSI;
        bool error = true;
        try {
            //OpenCV is used to capture images from the camera.
            //These calls setup opencv to attatch to the camera.


            _device = _config_sptr->GetConfigStringValue("NVidiaCSIcam.device", "0");
            _capture_width = _config_sptr->GetConfigIntValue("NVidiaCSIcam.width", 640);
            _capture_height = _config_sptr->GetConfigIntValue("NVidiaCSIcam.height", 480);
            DesiredFrameRateFPS = _config_sptr->GetConfigIntValue("NVidiaCSIcam.FramesPerSec", 30);
            string imgFmtStr = _config_sptr->GetConfigStringValue("NVidiaCSIcam.ImageFormat", "RGB");
            ImageCaptureFormat_e imgFmt = ImageCaptureFormat_e::RGB24;

            if (imgFmtStr == "YUV422")
                imgFmt = ImageCaptureFormat_e::YUV422;
            else if (imgFmtStr == "MJPEG")
                imgFmt = ImageCaptureFormat_e::MJPEG;
            else if (imgFmtStr == "BGR24")
                imgFmt = ImageCaptureFormat_e::BGR24;
            else if (imgFmtStr == "RGB24")
                imgFmt = ImageCaptureFormat_e::RGB24;


            _imageCaptured = false;
            imgCapCtrlMsg.ImageCaptureSourceConfigPri = _device;
            imgCapCtrlMsg.DesiredImageWidth = _capture_width;
            imgCapCtrlMsg.DesiredImageHeight = _capture_height;
            imgCapCtrlMsg.DesiredFramesPerSecond = DesiredFrameRateFPS;
            imgCapCtrlMsg.ImageCaptureFormat = imgFmt;
            imgCapCtrlMsg.AutoFocusEnable = false;

            error = Initialize(imgCapCtrlMsg);
        }
        catch (std::exception &e) {
            LOGERROR("CSI_Camera:Initialize Exception: " << e.what());
            error = true;
        }
        return error;
    }


    bool CSI_Camera::Initialize(ImageCaptureControlMessage &imgCapCtrlMsg) {
        bool error = true;
        _device = imgCapCtrlMsg.ImageCaptureSourceConfigPri;
        _numberOfImagesCaptured = 0;
        _imageCaptured = false;
        uint32_t videoVal;

        //Start with copying the control message over to the status message...
        //the status message will be updated as needed by actual values.
        ImageCaptureControlStatusMsg.CopyMessage(&imgCapCtrlMsg);
        ImageCaptureControlStatusMsg.ImageCaptureSource = ImageCaptureSource_e::ImageCaptureSource_NVidiaCSI;
        //Use the ImageCaptureEnabled flag to indicate the webcam is
        //setup ok.
        ImageCaptureControlStatusMsg.ImageCaptureEnabled = false;
        ImageCaptureControlStatusMsg.SetTimeNow();

        //Close the camera and start new... allows parameters to be changed.
        Close();

        if (!_IsImageCaptureInitialized) {
            LOGINFO("Initializing the CSI_Camera");
            try {
                string fmtStr = get_tegra_pipeline(imgCapCtrlMsg.DesiredImageWidth,
                                                   imgCapCtrlMsg.DesiredImageHeight,
                                                   (int)imgCapCtrlMsg.DesiredFramesPerSecond);

                //fmtStr = "nvcamerasrc ! 'video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1, format=NV12' ! nvvidconv flip-method=2 ! nvegltransform ! nveglglessink -e";

                //const char* gst =  "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)I420, framerate=(fraction)120/1 ! \
			//nvvidconv flip-method=6 ! video/x-raw, format=(string)BGRx ! \
			//videoconvert ! video/x-raw, format=(string)BGR ! \
			//appsink";

                //OpenCV is used to capture images from the camera.
                //These calls setup opencv to attatch to the camera.
                _capture_sptr = unique_ptr<cv::VideoCapture>(new cv::VideoCapture(fmtStr.c_str(), cv::CAP_GSTREAMER));

                 if (!_capture_sptr->isOpened()) {
                    /* This is FATAL, throw the error back up to the top*/
                    LOGERROR("Can't get a NVidiaCSI VideoCapture object.")
                    error = true;
                } else {

                    //Verify we can Grab a single image from the camera.
                    //The image is simply tossed.
                    cv::Mat frame;
                    //*_capture_sptr >> frame;
                    _capture_sptr->read(frame);
                    if (frame.data != NULL
                        && frame.rows > 0
                        && frame.cols > 0) {

                        //Set and then verify the image properties.
                        //_capture_sptr->set(CV_CAP_PROP_FPS, _capture_fps);   //Not Necessary and does'nt work most the time.
                        //_capture_sptr->set(CV_CAP_PROP_FRAME_HEIGHT, imgCapCtrlMsg.DesiredImageHeight);
                        //_capture_sptr->set(CV_CAP_PROP_FRAME_WIDTH, imgCapCtrlMsg.DesiredImageWidth);


                        //videoVal = imgCapCtrlMsg.AutoFocusEnable ? 1 : 0;
                        _capture_sptr->set(cv::CAP_PROP_AUTOFOCUS, videoVal);
//
                        //Get the actual Image Width and Height supported by the Webcam.
                        _capture_height = _capture_sptr->get(cv::CAP_PROP_FRAME_HEIGHT);
                        _capture_width = _capture_sptr->get(cv::CAP_PROP_FRAME_WIDTH);
                        ImageCaptureControlStatusMsg.DesiredImageHeight = _capture_height;
                        ImageCaptureControlStatusMsg.DesiredImageWidth = _capture_width;

                        videoVal = _capture_sptr->get(cv::CAP_PROP_FPS);
                        LOGINFO("OpenCVWebcam FPS: " << videoVal);
                        ImageCaptureControlStatusMsg.DesiredFramesPerSecond = videoVal;

                        videoVal = _capture_sptr->get(cv::CAP_PROP_MODE);
                        LOGINFO("OpenCVWebcam MODE: " << videoVal);
                        videoVal = _capture_sptr->get(cv::CAP_PROP_CONVERT_RGB);
                        LOGINFO("OpenCVWebcam CONVERT_RGB: " << videoVal);
                        videoVal = _capture_sptr->get(cv::CAP_PROP_AUTO_EXPOSURE);
                        LOGINFO("OpenCVWebcam AUTO_EXPOSURE: " << videoVal);

                        LOGINFO("CSI_Camera Image Width: " << _capture_width)
                        LOGINFO("CSI_Camera Image Height: " << _capture_height)
                        LOGINFO("CSI_Camera is open.");

                        //Run loop as fast as possible to keep images from building up
                        //in the frame buffer.
                        _desiredImgCaptureMgrLoopTimeMSec = 10.0;

                        _IsImageCaptureInitialized = true;

                        //Use the ImageCaptureEnabled flag to indicate the webcam is
                        //setup ok.
                        ImageCaptureControlStatusMsg.ImageCaptureEnabled = true;
                        error = false;
                    } else {
                        LOGERROR("Can't capture an image from the camera.")
                        error = true;
                    }
                }
            }
            catch (std::exception &e) {
                LOGERROR("CSI_Camera:Initialize Exception: " << e.what());
                _IsImageCaptureInitialized = false;
                error = true;
            }

        }

        return error;
    }


    uint32_t CSI_Camera::ToImageFormat(ImageCaptureFormat_e imgFmt) {
        uint32_t pixFormat = V4L2_PIX_FMT_MJPEG;
        switch (imgFmt) {
            case ImageCaptureFormat_e::MJPEG:
                pixFormat = V4L2_PIX_FMT_MJPEG;
                break;
            case ImageCaptureFormat_e::YUV422:
                pixFormat = V4L2_PIX_FMT_YUYV;
                break;
            case ImageCaptureFormat_e::BGR24:
                pixFormat = V4L2_PIX_FMT_BGR24;
                break;
            default:
                pixFormat = V4L2_PIX_FMT_MJPEG;
        }
        return pixFormat;
    }


    //PreCaptureImage is used primarily by webcam and
    //simular devices that have a capture buffer that can be
    //filled faster than the images are being used by image processing.
    //A precapture can be called to capture an image.. if image processing
    //is not ready, the image will be thrown away.
    //PreCaptureImage must be called by the Image Capture manager before
    //GetNextImage.
    bool CSI_Camera::PreCaptureImage() {
        _imageCaptured = false;
        //ToDo:  See if there is a way to create an atomic operation
        //that locks getting a timestamp to the capture of the image.
        //Image capture can take a period of time... a significant portion
        //just downloading the image from the camera... so assume for now
        //that we capture the time then fire-off the process of capturing the image.
        if (_IsImageCaptureInitialized && _capture_sptr != nullptr) {
            SetImageCaptureTimeToNow();
            _imageCaptured = _capture_sptr->grab();
        }
        return _imageCaptured;
    }


    ImageCaptureReturnType_e CSI_Camera::GetNextImage(ImagePlusMetadataMessage *imgPMetadataMsgPtr) {
        ImageCaptureReturnType_e rtn = ImageCaptureReturnType_e::ICRT_Error;
        try {
            //A Webcam buffers images.  To prevent a build-up of images in the buffer... we need to
            //pull images from the buffer even if they are not being used so that when a image is to be used,
            //a fresh image will be retrieve.  The Image capture manager must run at least as fast as the
            //webcam frame rate to ensure the buffer does not fill with old images.

            if (!_imageCaptured) {
                PreCaptureImage();
            }

            if (_imageCaptured && imgPMetadataMsgPtr != NULL) {
                if (MaxNumberOfImagesToCapture == 0 || _numberOfImagesCaptured < MaxNumberOfImagesToCapture) {
                    //Retrieve the image grabbed/captured above;
                    _capture_sptr->retrieve(imgPMetadataMsgPtr->ImageFrame);
                    _imageCaptured = false;
                    if (imgPMetadataMsgPtr->ImageFrame.data != NULL
                        && imgPMetadataMsgPtr->ImageFrame.rows > 10
                        && imgPMetadataMsgPtr->ImageFrame.cols > 10) {
                        ++_imageNumberCounter;
                        ++_numberOfImagesCaptured;
                        imgPMetadataMsgPtr->ImageCaptureTimeStampSec = GetImageCaptureTime();
                        imgPMetadataMsgPtr->SetTimeNow();
                        imgPMetadataMsgPtr->ImageNumber = _imageNumberCounter;
                        imgPMetadataMsgPtr->ImageNoPixelsWide = imgPMetadataMsgPtr->ImageFrame.cols;
                        imgPMetadataMsgPtr->ImageNoPixelsHigh = imgPMetadataMsgPtr->ImageFrame.rows;
                        rtn = ImageCaptureReturnType_e::IRCT_ImageOnly;
                    } else {
                        imgPMetadataMsgPtr->ImageCaptureTimeStampSec = GetImageCaptureTime();
                        imgPMetadataMsgPtr->SetTimeNow();
                        imgPMetadataMsgPtr->ImageNumber = 0;
                        imgPMetadataMsgPtr->ImageNoPixelsWide = 0;
                        imgPMetadataMsgPtr->ImageNoPixelsHigh = 0;
                        rtn = ImageCaptureReturnType_e::ICRT_Error;
                    }
                } else {
                    rtn = ImageCaptureReturnType_e::IRCT_EndOfImages;
                }
            } else {
                rtn = ImageCaptureReturnType_e::ICRT_NoImageCaptured;
            }
        }
        catch (std::exception &e) {
            LOGERROR("CSI_Camera:GetNextImage Exception: " << e.what());
            rtn = ImageCaptureReturnType_e::ICRT_Error;
        }
        return rtn;
    }

    //Close the Image Capture Process.
    void CSI_Camera::Close() {
        try {
            if (_capture_sptr != nullptr) {
                _capture_sptr->release();
                _IsImageCaptureInitialized = false;
                ImageCaptureControlStatusMsg.ImageCaptureEnabled = false;
            }
        }
        catch (std::exception &e) {
            LOGERROR("CSI_Camera:Close Exception: " << e.what());
        }
    }

}




