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

#include "OpenCVWebcamImageCapture.h"
#include <linux/videodev2.h>
#include <time.h>                   // needed for struct tm
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc.hpp>
//#include <opencv2/imgcodecs.hpp> //OpenCV 3.0 only
//#include <opencv2/videoio.hpp> //OpenCV 3.0 only
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace chrono;
using namespace cv;

namespace videre
{

    //Initalize the Image Capture process.
    //Returns true if error, false if ok.
    bool OpenCVWebcamImageCapture::Initialize()
    {
        ImageCaptureControlMessage imgCapCtrlMsg;
        imgCapCtrlMsg.ImageCaptureSource = ImageCaptureSource_e ::ImageCaptureSource_OpenCVWebCam;
        bool error = true;
        try
        {
            //OpenCV is used to capture images from the camera.
            //These calls setup opencv to attatch to the camera.
            _device = _config_sptr->GetConfigStringValue("OpenCVWebcam.device", "0");
            _capture_width = _config_sptr->GetConfigIntValue("OpenCVWebcam.width", 640);
            _capture_height = _config_sptr->GetConfigIntValue("OpenCVWebcam.height", 480);
            _capture_framerate = _config_sptr->GetConfigIntValue("OpenCVWebcam.fps", 30);
            string imgFmtStr = _config_sptr->GetConfigStringValue("OpenCVWebcam.ImageFormat", "YUV422");
            ImageCaptureFormat_e imgFmt = ImageCaptureFormat_e::YUV422;

            if(imgFmtStr == "YUV422")
                imgFmt = ImageCaptureFormat_e::YUV422;
            if(imgFmtStr == "MJPEG")
                imgFmt = ImageCaptureFormat_e::MJPEG;
            if(imgFmtStr == "BGR24")
                imgFmt = ImageCaptureFormat_e::BGR24;

            bool autofocusEnable = _config_sptr->GetConfigBoolValue("OpenCVWebcam.AutoFocusEnable", false);

            _imageCaptured = false;
            imgCapCtrlMsg.ImageCaptureSourceConfigPri = _device;
            imgCapCtrlMsg.DesiredImageWidth = _capture_width;
            imgCapCtrlMsg.DesiredImageHeight = _capture_height;
            imgCapCtrlMsg.DesiredFramesPerSecond = _capture_framerate;
            imgCapCtrlMsg.ImageCaptureFormat = imgFmt;
            imgCapCtrlMsg.AutoFocusEnable = autofocusEnable;

            error = Initialize(imgCapCtrlMsg);
        }
        catch(std::exception &e)
        {
            LOGERROR("OpenCVWebcamImageCapture:Initialize Exception: " << e.what());
            error = true;
        }
        return error;
    }


    bool OpenCVWebcamImageCapture::Initialize(ImageCaptureControlMessage &imgCapCtrlMsg)
    {
        bool error = true;
        _device = imgCapCtrlMsg.ImageCaptureSourceConfigPri;
        _numberOfImagesCaptured = 0;
        _imageCaptured = false;
        uint32_t  videoVal;

        //Start with copying the control message over to the status message...
        //the status message will be updated as needed by actual values.
        ImageCaptureControlStatusMsg.CopyMessage(&imgCapCtrlMsg);
        ImageCaptureControlStatusMsg.ImageCaptureSource = ImageCaptureSource_e::ImageCaptureSource_OpenCVWebCam;
        //Use the ImageCaptureEnabled flag to indicate the webcam is
        //setup ok.
        ImageCaptureControlStatusMsg.ImageCaptureEnabled = false;
        ImageCaptureControlStatusMsg.SetTimeNow();

        //Close the webcam and start new... allows parameters to be changed.
        Close();

        if( !_IsImageCaptureInitialized)
        {
            LOGINFO("Initializing the OpenCVWebcamImageCapture");
            try
            {
                //OpenCV is used to capture images from the camera.
                //These calls setup opencv to attatch to the camera.
                _capture_sptr = unique_ptr<cv::VideoCapture>(new cv::VideoCapture(_device));

                if (!_capture_sptr->isOpened())
                    _capture_sptr->open(atoi(_device.c_str()));
                if (!_capture_sptr->isOpened())
                {
                    /* This is FATAL, throw the error back up to the top*/
                    LOGERROR("Can't get a VideoCapture object.")
                    error = true;
                }
                else
                {

                    //Verify we can Grab a single image from the camera.
                    //The image is simply tossed.
                    cv::Mat frame;
                    //*_capture_sptr >> frame;
                    _capture_sptr->read(frame);
                    if(frame.data != NULL
                            && frame.rows > 0
                            && frame.cols > 0) {

                        //Set and then verify the image properties.
                        int fps = 120;
                        _capture_sptr->set(cv::CAP_PROP_FPS, imgCapCtrlMsg.DesiredFramesPerSecond);   //Does'nt work most the time.
                        _capture_sptr->set(cv::CAP_PROP_FRAME_HEIGHT, imgCapCtrlMsg.DesiredImageHeight);
                        _capture_sptr->set(cv::CAP_PROP_FRAME_WIDTH, imgCapCtrlMsg.DesiredImageWidth);

                        videoVal = ToImageFormat(imgCapCtrlMsg.ImageCaptureFormat);
                        _capture_sptr->set(cv::CAP_PROP_FOURCC, videoVal);
                        //_capture_sptr->set(CV_CAP_PROP_MODE, videoVal);

                        videoVal = imgCapCtrlMsg.AutoFocusEnable ? 1 : 0;
                        _capture_sptr->set(cv::CAP_PROP_AUTOFOCUS, videoVal);

                        //Get the actual Image Width and Height supported by the Webcam.
                        _capture_height = _capture_sptr->get(cv::CAP_PROP_FRAME_HEIGHT);
                        _capture_width = _capture_sptr->get(cv::CAP_PROP_FRAME_WIDTH);
                        ImageCaptureControlStatusMsg.DesiredImageHeight = _capture_height;
                        ImageCaptureControlStatusMsg.DesiredImageWidth = _capture_width;

                        _capture_framerate = _capture_sptr->get(cv::CAP_PROP_FPS);
                        LOGINFO("OpenCVWebcam FPS: {0}" << _capture_framerate);
                        ImageCaptureControlStatusMsg.DesiredFramesPerSecond = _capture_framerate;

                        videoVal = _capture_sptr->get(cv::CAP_PROP_FOURCC);
                        LOGINFO("OpenCVWebcam FOURCC: " << videoVal)
                        videoVal = _capture_sptr->get(cv::CAP_PROP_FORMAT);
                        LOGINFO("OpenCVWebcam FORMAT: " << videoVal);

                        videoVal = _capture_sptr->get(cv::CAP_PROP_MODE);
                        LOGINFO("OpenCVWebcam MODE: " << videoVal);
                        videoVal = _capture_sptr->get(cv::CAP_PROP_CONVERT_RGB);
                        LOGINFO("OpenCVWebcam CONVERT_RGB: " << videoVal)
                        videoVal = _capture_sptr->get(cv::CAP_PROP_AUTO_EXPOSURE);
                        LOGINFO("OpenCVWebcam AUTO_EXPOSURE: " << videoVal);
                        //videoVal = _capture_sptr->get(CV_CAP_PROP_ZOOM);
                        //LOGINFO("OpenCVWebcam ZOOM: {0}" << videoVal)
                        videoVal = _capture_sptr->get(cv::CAP_PROP_FOCUS);
                        LOGINFO("OpenCVWebcam FOCUS: " << videoVal);
                        videoVal = _capture_sptr->get(cv::CAP_PROP_AUTOFOCUS);
                        LOGINFO("OpenCVWebcam AUTO FOCUS: " << videoVal);
                        ImageCaptureControlStatusMsg.AutoFocusEnable = videoVal == 0 ? false : true;

                        LOGINFO("OpenCVWebcam Image Width: " << _capture_width);
                        LOGINFO("OpenCVWebcam Image Height: " << _capture_height);
                        LOGINFO("OpenCVWebcamImageCapture is open.");

                        //Run loop as fast as possible to keep images from building up
                        //in the frame buffer.
                        _desiredImgCaptureMgrLoopTimeMSec = 10.0;

                        _IsImageCaptureInitialized = true;

                        //Use the ImageCaptureEnabled flag to indicate the webcam is
                        //setup ok.
                        ImageCaptureControlStatusMsg.ImageCaptureEnabled = true;
                        error = false;
                    }
                    else
                    {
                        LOGERROR("Can't capture an image from the camera.");
                        error = true;
                    }
                }
            }
            catch (std::exception &e)
            {
                LOGERROR("OpenCVWebcamImageCapture:Initialize Exception: " << e.what());
                _IsImageCaptureInitialized = false;
                error = true;
            }

        }

        return error;
    }


    uint32_t OpenCVWebcamImageCapture::ToImageFormat(ImageCaptureFormat_e imgFmt)
    {
        uint32_t pixFormat = V4L2_PIX_FMT_MJPEG;
        switch(imgFmt)
        {
            case ImageCaptureFormat_e::MJPEG:
                pixFormat = V4L2_PIX_FMT_MJPEG;
                break;
            case ImageCaptureFormat_e::YUV422:
                pixFormat = V4L2_PIX_FMT_YUYV;
                break;
            case ImageCaptureFormat_e::BGR24:
                pixFormat = V4L2_PIX_FMT_BGR24;
                break;
            case ImageCaptureFormat_e::RGB24:
                pixFormat = V4L2_PIX_FMT_RGB24;
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
    bool OpenCVWebcamImageCapture::PreCaptureImage()
    {
        _imageCaptured = false;
        //ToDo:  See if there is a way to create an atomic operation
        //that locks getting a timestamp to the capture of the image.
        //Image capture can take a period of time... a significant portion
        //just downloading the image from the camera... so assume for now
        //that we capture the time then fire-off the process of capturing the image.
        if(_IsImageCaptureInitialized && _capture_sptr != nullptr)
        {
            SetImageCaptureTimeToNow();
            _imageCaptured = _capture_sptr->grab();
        }
        return _imageCaptured;
    }


    ImageCaptureReturnType_e OpenCVWebcamImageCapture::GetNextImage(ImagePlusMetadataMessage *imgPMetadataMsgPtr)
    {
        ImageCaptureReturnType_e rtn = ImageCaptureReturnType_e::ICRT_Error;
        try
        {
            //A Webcam buffers images.  To prevent a build-up of images in the buffer... we need to
            //pull images from the buffer even if they are not being used so that when a image is to be used,
            //a fresh image will be retrieve.  The Image capture manager must run at least as fast as the
            //webcam frame rate to ensure the buffer does not fill with old images.

            if(!_imageCaptured)
            {
                PreCaptureImage();
            }

            if(_imageCaptured && imgPMetadataMsgPtr != NULL )
            {
                if (MaxNumberOfImagesToCapture == 0 || _numberOfImagesCaptured < MaxNumberOfImagesToCapture)
                {
                    //Retrieve the image grabbed/captured above;
                    _capture_sptr->retrieve(imgPMetadataMsgPtr->ImageFrame);
                    _imageCaptured = false;
                    if (imgPMetadataMsgPtr->ImageFrame.data != NULL
                        && imgPMetadataMsgPtr->ImageFrame.rows > 10
                        && imgPMetadataMsgPtr->ImageFrame.cols > 10)
                    {
                        ++_imageNumberCounter;
                        ++_numberOfImagesCaptured;
                        imgPMetadataMsgPtr->ImageCaptureTimeStampSec = GetImageCaptureTime();
                        imgPMetadataMsgPtr->SetTimeNow();
                        imgPMetadataMsgPtr->ImageNumber = _imageNumberCounter;
                        imgPMetadataMsgPtr->ImageNoPixelsWide = imgPMetadataMsgPtr->ImageFrame.cols;
                        imgPMetadataMsgPtr->ImageNoPixelsHigh = imgPMetadataMsgPtr->ImageFrame.rows;
                        rtn = ImageCaptureReturnType_e::IRCT_ImageOnly;
                    }
                    else
                    {
                        imgPMetadataMsgPtr->ImageCaptureTimeStampSec = GetImageCaptureTime();
                        imgPMetadataMsgPtr->SetTimeNow();
                        imgPMetadataMsgPtr->ImageNumber = 0;
                        imgPMetadataMsgPtr->ImageNoPixelsWide = 0;
                        imgPMetadataMsgPtr->ImageNoPixelsHigh = 0;
                        rtn = ImageCaptureReturnType_e::ICRT_Error;
                    }
                }
                else
                {
                    rtn = ImageCaptureReturnType_e::IRCT_EndOfImages;
                }
            }
            else
            {
                rtn = ImageCaptureReturnType_e::ICRT_NoImageCaptured;
            }
        }
        catch(std::exception &e)
        {
            LOGERROR("OpenCVWebcamImageCapture:GetNextImage Exception: " << e.what());
            rtn = ImageCaptureReturnType_e::ICRT_Error;
        }
        return rtn;
    }

    //Close the Image Capture Process.
    void OpenCVWebcamImageCapture::Close()
    {
        try
        {
            if(_capture_sptr != nullptr)
            {
                _capture_sptr->release();
                _IsImageCaptureInitialized = false;
                ImageCaptureControlStatusMsg.ImageCaptureEnabled = false;
            }
        }
        catch(std::exception &e)
        {
            LOGERROR("OpenCVWebcamImageCapture:Close Exception: " << e.what());
        }
    }




}