/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: April 2017
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
  *******************************************************************/

#include "Sensoray2253ImageCapture.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <asm/types.h>          /* for videodev2.h */
#include <linux/videodev2.h>
#include "Sensoray/s2253ioctl.h"
#include "FileUtils.h"

using namespace std;

namespace videre
{

    /****************************************************************************
    //Notes from .NET HOPS
    //Image Type:  STYPE_UYVY   (YUV)
    //Image Size:  640 x 480  Primary  Channel
    //ImageBufferSize = 614400
    //Half frame size:  320 x 240
    //  ConvertYUV_To_HalfSizeRGB=true
    //Secondary Channel:
    // ImageSensorMode = MJPEG
    // Image Size:  320 x 240
    // Compression:  75
    // ImageBufferSize>153600
    //
    // S2253.SetRecordMode(S2253.MID2253_RECMODE.MID2253_RECMODE_VIDEO, 0, 0);
    // S2253.EnableSnapshot(1, 0, 0);
     // S2253.StartSnapshot(0, 0);

    ********************************************************************************/

    bool Sensoray2253ImageCapture::ReadSenorayDeviceConfig()
    {
        bool error = false;
        std::string strVal;
        int intVal;
        try
        {
            //Video4Linux is used to capture images from the camera.
            //These calls setup opencv to attatch to the camera.
            _device_1_enabled = _config_sptr->GetConfigBoolValue("Sensoray.Device_1_Enabled", true);
            _device_1_connectionStr = _config_sptr->GetConfigStringValue("Sensoray.Device_1_Name", "/dev/video0");
            strVal = _config_sptr->GetConfigStringValue("Sensoray.Device_1_ImageType", "YUYV");
            Device_1_ImageType = ImageCompressionTypeStringToEnum(strVal);
            strVal = _config_sptr->GetConfigStringValue("Sensoray.Device_1_VideoSystem", "NTSC");
            _device_1_VideoSystem = VideosystemStringToEnum(strVal);
            intVal = _config_sptr->GetConfigIntValue("Sensoray.Device_1_CompressionLevel", 100);
            SetDevice1CompressionLevel(intVal);
            intVal = _config_sptr->GetConfigIntValue("Sensoray.Device_1_ImageHeight", 480);
            SetDevice1ImgageHeight(intVal);
            intVal = _config_sptr->GetConfigIntValue("Sensoray.Device_1_ImageWidth", 640);
            SetDevice1ImgageWidth(intVal);

            _device_2_enabled = _config_sptr->GetConfigBoolValue("Sensoray.Device_2_Enabled", false);
            _device_2_connectionStr = _config_sptr->GetConfigStringValue("Sensoray.Device_2_Name", "/dev/video1");
            strVal = _config_sptr->GetConfigStringValue("Sensoray.Device_1_ImageType", "JPEG");
            Device_2_ImageType = ImageCompressionTypeStringToEnum(strVal);
            strVal = _config_sptr->GetConfigStringValue("Sensoray.Device_2_VideoSystem", "NTSC");
            _device_2_VideoSystem = VideosystemStringToEnum(strVal);
            intVal = _config_sptr->GetConfigIntValue("Sensoray.Device_2_CompressionLevel", 40);
            SetDevice2CompressionLevel(intVal);
            intVal = _config_sptr->GetConfigIntValue("Sensoray.Device_2_ImageHeight", 480);
            SetDevice2ImgageHeight(intVal);
            intVal = _config_sptr->GetConfigIntValue("Sensoray.Device_2_ImageWidth", 640);
            SetDevice2ImgageWidth(intVal);

        }
        catch(std::exception &e)
        {
            LOGERROR("Sensoray2253ImageCapture:Read Config Exception: " << e.what());
            error = true;
        }
        return error;
    }



    //Initalize the Image Capture process.
    //Returns true if error, false if ok.
    bool Sensoray2253ImageCapture::Initialize()
    {
        bool error = true;
        std::string strVal;
        int intVal;
        try
        {
            //Video4Linux is used to capture images from the camera.
            if( !ReadSenorayDeviceConfig())
            {
                error = InitializedSensorayDevice( 1, _device_1_connectionStr,
                                                   Device_1_ImageType,
                                                   _device_1_VideoSystem,
                                                   _device_1_ImageWidth,
                                                   _device_1_ImageHeight,
                                                   _device_1_CompressionLevel);
                if(error)
                {
                    LOGERROR("Sensoray2253ImageCapture: Failed to initialize device 1")
                }
                else
                {
                    LOGINFO("Sensoray2253ImageCapture: Device 1 opened and ready for image capture.")
                }
            }
            //error = Initialize(_device, _capture_width, _capture_height);
        }
        catch(std::exception &e)
        {
            LOGERROR("Sensoray2253ImageCapture:Initialize Exception: " << e.what());
            error = true;
        }
        return error;
    }



    int Sensoray2253ImageCapture::xioctrl(int fd, int request, void *arg)
    {
        int r = -1;
        try
        {
            do
            {
                r = ioctl(fd, request, arg);
            } while (r == -1 && errno == EINTR);
        }
        catch(std::exception &e)
        {
            LOGERROR("Sensoray2253ImageCapture:xioctrl: " << e.what());
            r = -1;
        }

        return r;
    }

    //ToDo: The Sensorray Device is not currently being used...
    //needs work.
    bool Sensoray2253ImageCapture::Initialize(ImageCaptureControlMessage &imgCapCtrlMsg)
    {
        //Start with copying the control message over to the status message...
        //the status message will be updated as needed by actual values.
        ImageCaptureControlStatusMsg.CopyMessage(&imgCapCtrlMsg);
        ImageCaptureControlStatusMsg.ImageCaptureSource = ImageCaptureSource_e ::ImageCaptureSource_OpenCVWebCam;
        //Use the ImageCaptureEnabled flag to indicate the webcam is
        //setup ok.
        ImageCaptureControlStatusMsg.ImageCaptureEnabled = false;
        ImageCaptureControlStatusMsg.SetTimeNow();

        return true;
    }

    bool Sensoray2253ImageCapture::InitializedSensorayDevice(  int deviceNumber,
                                                               std::string connectionString,
                                                               SensorayImageType_e imgType,
                                                               SensorayVideosystem_e videoSystem,
                                                               int imgWidth, int imgHeight,
                                                               int compressionLvl )
    {
        bool error = true;
        int errval = 0;
        int *deviceHandlePtr = nullptr;
        //+std::ostringstream msgBuf;

        ImageCaptureControlStatusMsg.ImageCaptureSource = ImageCaptureSource_e ::ImageCaptureSource_OpenCVWebCam;
        //Use the ImageCaptureEnabled flag to indicate the webcam is
        //setup ok.
        ImageCaptureControlStatusMsg.ImageCaptureEnabled = false;
        ImageCaptureControlStatusMsg.SetTimeNow();


        struct v4l2_capability cap;
        struct v4l2_format fmt;
        struct s2253p_encoder encSettings;
        if(deviceNumber == 1)
            deviceHandlePtr = &_device_1_handle;
        else if(deviceNumber == 2)
            deviceHandlePtr = &_device_2_handle;
        if(deviceHandlePtr != nullptr )
        {
            if( *deviceHandlePtr > 0 )
            {
                CloseSensorayDevice(deviceNumber);
            }
            try
            {
                *deviceHandlePtr = open(connectionString.c_str(), O_RDWR /* required */ | O_NONBLOCK, 0);
                if( *deviceHandlePtr >= 0)
                {
                    //Setup device.
                    //Check device capablities and log for later info.
                    errval = ioctl(*deviceHandlePtr, VIDIOC_QUERYCAP, &cap);
                    if ( errval >=0 )
                    {
                        LOGINFO("Sensoray Device [" << deviceNumber << "] opened.  Card type: "
                        << (char *)(cap.card) << " Capibilities Value: " << cap.capabilities);
                    }
                    else
                    {
                        LOGERROR("InitializedSensorayDevice failed to open device: " << deviceNumber);
                        CloseSensorayDevice(deviceNumber);
                        return true;
                    }

                    v4l2_std_id vidSys = videoSystem == SRVIDEOSYS_PAL ? V4L2_STD_PAL : V4L2_STD_NTSC;
                    errval = ioctl(*deviceHandlePtr, VIDIOC_S_STD, &vidSys);
                    //errval = 0;
                    if( errval < 0 )
                    {
                        LOGERROR("InitializedSensorayDevice failed set Video system (NTSC/PAL): " << deviceNumber);
                    }

                    memset (&(fmt), 0, sizeof (v4l2_format));
                    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                    fmt.fmt.pix.height = imgHeight;
                    fmt.fmt.pix.width = imgWidth;
                    fmt.fmt.pix.pixelformat = ToPixelFormat(imgType);
                    //fmt.fmt.pix.field = videoSystem == SRVIDEOSYS_PAL ? V4L2_FIELD_NONE : V4L2_FIELD_INTERLACED;;
                    fmt.fmt.pix.field = V4L2_FIELD_NONE;

                    errval = ioctl(*deviceHandlePtr, VIDIOC_S_FMT, &fmt);
                    //errval = 0;
                    if( errval < 0 )
                    {
                        LOGERROR("InitializedSensorayDevice Video Capture Paramters: " << deviceNumber);
                        CloseSensorayDevice(deviceNumber);
                        return true;
                    }

                    //Set Capture to Async.
                    if( deviceNumber == 1)
                    {
                        encSettings.chan_mask = 0x01;
                        encSettings.enable_mask = 0x01;
                    }
                    else
                    {
                        encSettings.chan_mask = 0x02;
                        encSettings.enable_mask = 0x02;
                    }
                    errval = ioctl(*deviceHandlePtr, S2253P_VIDIOC_S_ENC_ASYNCEN, &encSettings);
                    //errval = 0;
                    if( errval < 0 )
                    {
                        LOGERROR("InitializedSensorayDevice Video Async set failed: " << deviceNumber);
                    }

                    if(imgType == SRCTYPE_JPEG)
                    {
                        struct v4l2_jpegcompression G_jc;	/* jpeg compression */
                        memset (&(G_jc), 0, sizeof (v4l2_jpegcompression));

                        errval = ioctl(*deviceHandlePtr, VIDIOC_S_JPEGCOMP, &G_jc);
                        G_jc.quality = compressionLvl < 10 ? 10 : compressionLvl > 90 ? 90 : compressionLvl;
                        if( errval < 0 )
                        {
                            LOGERROR("InitializedSensorayDevice failed set JPEG compression level: " << deviceNumber);
                        }
                    }

                    //Compute Buffer/frame sizes and update info based upon device number
                    //For compressed images the actual image size will be smaller than the
                    //frame size.
                    size_t frameSize = 3 * imgWidth * imgHeight;
                    if(deviceNumber == 1)
                    {
                        _device_1_ImageWidth = imgWidth;
                        _device_1_ImageHeight = imgHeight;
                        _capture_width = imgWidth;
                        _capture_height = imgHeight;
                        _device_1_frameSize = frameSize;
                    }
                    else if(deviceNumber == 2)
                    {
                        _device_2_ImageWidth = imgWidth;
                        _device_2_ImageHeight = imgHeight;
                        _device_2_frameSize = frameSize;
                    }

                    error = false;
                    ImageCaptureControlStatusMsg.ImageCaptureEnabled = true;
                }
                else
                {
                   //There was an error opening the device.
                    LOGERROR("Error opening Sensoray Device: " << connectionString);
                }
            }
            catch (std::exception &e)
            {
                LOGERROR("Sensoray2253ImageCapture:InitializedSensorayDevice Exception: " << e.what());
                error = true;
            }
        }
        return error;
    }

    void Sensoray2253ImageCapture::CloseSensorayDevice(int deviceNumber)
    {
        int *deviceHandlePtr = nullptr;
        if(deviceNumber == 1)
            deviceHandlePtr = &_device_1_handle;
        else if(deviceNumber == 1)
            deviceHandlePtr = &_device_1_handle;
        if(deviceHandlePtr != nullptr && *deviceHandlePtr > 0)
        {
            try
            {
                close(*deviceHandlePtr);
                *deviceHandlePtr = 0;
            }
            catch (std::exception &e)
            {
                LOGERROR("Sensoray2253ImageCapture:CloseSensorayDevice Exception: " << e.what());
            }
        }
        ImageCaptureControlStatusMsg.ImageCaptureEnabled = false;
    }

    uint32_t Sensoray2253ImageCapture::ToPixelFormat(SensorayImageType_e imgType)
    {
        uint32_t pixFormat = V4L2_PIX_FMT_MJPEG;
        switch(imgType)
        {
            case SRCTYPE_JPEG:
                pixFormat = V4L2_PIX_FMT_MJPEG;
                break;
            case SRCTYPE_YUYV:
                pixFormat = V4L2_PIX_FMT_YUYV;
                break;
            case SRCTYPE_UYVY:
                pixFormat = V4L2_PIX_FMT_UYVY;
                break;
            case SRCTYPE_BGR24:
                pixFormat = V4L2_PIX_FMT_BGR24;
                break;
            default:
                pixFormat = V4L2_PIX_FMT_MJPEG;
        }
        return pixFormat;
    }

    SensorayImageType_e Sensoray2253ImageCapture::ImageCompressionTypeStringToEnum(const std::string imageType)
    {
        SensorayImageType_e imgType = SRCTYPE_JPEG;
        if(VidereFileUtils::CompareStringCaseInsensitive(imageType, "jpeg") == 0)
            imgType = SRCTYPE_JPEG;
        else if(VidereFileUtils::CompareStringCaseInsensitive(imageType, "yuyv") == 0)
            imgType = SRCTYPE_YUYV;
        else if(VidereFileUtils::CompareStringCaseInsensitive(imageType, "uyvy") == 0)
            imgType = SRCTYPE_UYVY;
        else if(VidereFileUtils::CompareStringCaseInsensitive(imageType, "bgr24") == 0)
            imgType = SRCTYPE_BGR24;
        else
            LOGWARN("Sensoray2253ImageCapture:ImageCompressionTypeStringToEnum invalid type: " << imageType);

        return imgType;
    }

    SensorayVideosystem_e Sensoray2253ImageCapture::VideosystemStringToEnum(const std::string videoSys)
    {
        SensorayVideosystem_e vidSysType = SRVIDEOSYS_NTSC;
        if(VidereFileUtils::CompareStringCaseInsensitive(videoSys, "pal") == 0)
            vidSysType = SRVIDEOSYS_PAL;

        return vidSysType;
    }


    //Capture an image from Device 1
    ImageCaptureReturnType_e Sensoray2253ImageCapture::GetNextImage(ImagePlusMetadataMessage *imgPMetadataMsgPtr)
    {
        ImageCaptureReturnType_e rtn = ImageCaptureReturnType_e::ICRT_Error;
        struct v4l2_buffer imageBuf;
        unsigned char* bufPtr;
        try
        {
            if( _device_1_handle > 0
                && (MaxNumberOfImagesToCapture == 0 || _numberOfImagesCaptured < MaxNumberOfImagesToCapture) )
            {
                //memset (&(imageBuf), 0, sizeof (v4l2_buffer));
                //imageBuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                //imageBuf.memory = V4L2_MEMORY_MMAP;

                //Capture an image
                //cv::Mat frame;
                //ToDo:  There is a clock in the sensoray that could be set to the
                //corrected HOPS time... this might be used to set the image capture time
                //This needs to be looked into.
                SetImageCaptureTimeToNow();

                if( imgPMetadataMsgPtr->ImageFrame.cols < _device_1_ImageWidth
                        || imgPMetadataMsgPtr->ImageFrame.rows < _device_1_ImageHeight)
                {
                    if(imgPMetadataMsgPtr->ImageFrame.cols == 0 || imgPMetadataMsgPtr->ImageFrame.rows == 0)
                    {
                        imgPMetadataMsgPtr->ImageFrame = cv::Mat(_device_1_ImageHeight, _device_1_ImageWidth, CV_8UC3);
                    }
                    else
                    {
                        cv::resize(imgPMetadataMsgPtr->ImageFrame,
                                   imgPMetadataMsgPtr->ImageFrame,
                                   cv::Size(_device_1_ImageWidth, _device_1_ImageHeight));
                    }
                }

                read(_device_1_handle, imgPMetadataMsgPtr->ImageFrame.data, _device_1_frameSize);

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
                rtn = ImageCaptureReturnType_e::IRCT_EndOfImages;
            }
        }
        catch(std::exception &e)
        {
            LOGERROR("Sensoray2253ImageCapture:GetNextImage Exception: " << e.what());
            rtn = ImageCaptureReturnType_e::ICRT_Error;
        }
        return rtn;
    }

    //Close the Image Capture Process.
    void Sensoray2253ImageCapture::Close()
    {
        try
        {
            CloseSensorayDevice(1);
            CloseSensorayDevice(2);
            ImageCaptureControlStatusMsg.ImageCaptureEnabled = false;
        }
        catch(std::exception &e)
        {
            LOGERROR("Sensoray2253ImageCapture:Close Exception: " << e.what());
        }
    }


}