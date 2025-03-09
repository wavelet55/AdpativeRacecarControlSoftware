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

#ifndef VIDERE_DEV_SENSORAY2253IMAGECAPTURE_H
#define VIDERE_DEV_SENSORAY2253IMAGECAPTURE_H

#include "ImageCaptureAClass.h"
#include <opencv2/core.hpp>

namespace videre
{
    typedef enum {
        SRCTYPE_JPEG,
        SRCTYPE_MJPEG,
        SRCTYPE_MPEG1,
        SRCTYPE_MPEG2,
        SRCTYPE_MPEG4,
        SRCTYPE_H264,
        SRCTYPE_YUYV,
        SRCTYPE_UYVY,
        SRCTYPE_Y8,
        SRCTYPE_NV12,
        SRCTYPE_MP42,
        SRCTYPE_MPEGTS,
        SRCTYPE_MPEGPS,
        SRCTYPE_BGR24,
        SRCTYPE_RGB565,
    } SensorayImageType_e;

    typedef enum {
        SRVIDEOSYS_NTSC,
        SRVIDEOSYS_PAL
    }SensorayVideosystem_e;

    class Sensoray2253ImageCapture : public ImageCaptureAClass
    {
        //Device 1 is the primary channel for Image Capture.
        //The enable flag is set by the config to indicate whether or not
        //to use the specific device number.
        bool _device_1_enabled = true;
        //Device 2 is the secondary... typically used to capture a comressed
        //image for sending to the ground station.
        bool _device_2_enabled = false;
        std::string _device_1_connectionStr;
        std::string _device_2_connectionStr;

        //The device handle will be > 0 if the device is setup and active
        int _device_1_handle = 0;
        int _device_2_handle = 0;

        int _device_1_ImageWidth = 640;
        int _device_1_ImageHeight = 480;
        SensorayVideosystem_e _device_1_VideoSystem = SRVIDEOSYS_NTSC;    //NTSC or PAL
        size_t _device_1_frameSize;


        int _device_2_ImageWidth = 640;
        int _device_2_ImageHeight = 480;
        SensorayVideosystem_e _device_2_VideoSystem = SRVIDEOSYS_NTSC;    //NTSC or PAL
        size_t _device_2_frameSize;

        //Compression levels are for JPEG only and are a
        //value in the range: [10, 90]... 90 is the best quality.
        int _device_1_CompressionLevel = 90;
        int _device_2_CompressionLevel = 90;

    public:

        SensorayImageType_e Device_1_ImageType = SRCTYPE_UYVY;
        SensorayImageType_e Device_2_ImageType = SRCTYPE_JPEG;



        //NTSC is interlaced... set to true to convert image
        //to half size to remove interlace effects.
        bool ConvertToHalfSize = true;

    public:
        Sensoray2253ImageCapture(std::shared_ptr<ConfigData> config,
                                 std::shared_ptr<ImageCaptureControlMessage> imageCaptureControlMsg)
                : ImageCaptureAClass(config)
        {}

        ~Sensoray2253ImageCapture()
        {
            Close();
        }

        std::string GetDevice1()
        {
            return _device_1_connectionStr;
        }

        int GetDevice1CompressionLevel()
        {
            return _device_1_CompressionLevel;
        }

        void SetDevice1CompressionLevel(int value)
        {
            _device_1_CompressionLevel = value < 10 ? 10 : value > 90 ? 90 : value;
        }

        int GetDevice1ImgageHeight()
        {
            return _device_1_ImageHeight;
        }

        void SetDevice1ImgageHeight(int value)
        {
            _device_1_ImageHeight = value < 64 ? 64 : value > 1080 ? 1080 : value;
        }

        int GetDevice1ImgageWidth()
        {
            return _device_1_ImageWidth;
        }

        void SetDevice1ImgageWidth(int value)
        {
            _device_1_ImageWidth = value < 64 ? 64 : value > 1920 ? 1920 : value;
        }



        bool GetDevice1Enabled()
        {
            return _device_1_enabled;
        }

        std::string GetDevice2()
        {
            return _device_2_connectionStr;
        }

        bool GetDevice2Enabled()
        {
            return _device_2_enabled;
        }

        int GetDevice2CompressionLevel()
        {
            return _device_2_CompressionLevel;
        }

        void SetDevice2CompressionLevel(int value)
        {
            _device_2_CompressionLevel = value < 10 ? 10 : value > 90 ? 90 : value;
        }

        int GetDevice2ImgageHeight()
        {
            return _device_2_ImageHeight;
        }

        void SetDevice2ImgageHeight(int value)
        {
            _device_2_ImageHeight = value < 64 ? 64 : value > 1080 ? 1080 : value;
        }

        int GetDevice2ImgageWidth()
        {
            return _device_2_ImageWidth;
        }

        void SetDevice2ImgageWidth(int value)
        {
            _device_2_ImageWidth = value < 64 ? 64 : value > 1920 ? 1920 : value;
        }

        uint32_t ToPixelFormat(SensorayImageType_e imgType);

        SensorayImageType_e ImageCompressionTypeStringToEnum(const std::string imageType);

        SensorayVideosystem_e VideosystemStringToEnum(const std::string videoSys);

        //Initalize the Image Capture process.
        //Returns true if error, false if ok.
        virtual bool Initialize();

        //Initalize the Image Capture process.
        //Returns true if error, false if ok.
        virtual bool Initialize(ImageCaptureControlMessage &imgCapCtrlMsg);

        virtual ImageCaptureReturnType_e GetNextImage(ImagePlusMetadataMessage *imgPMetadataMsgPtr);

        //Close the Image Capture Process.
        virtual void Close();

        int xioctrl(int fd, int request, void *arg);

        bool ReadSenorayDeviceConfig();

        bool InitializedSensorayDevice(int deviceNumber, std::string connectionString,
                                       SensorayImageType_e imgType,
                                       SensorayVideosystem_e videoSystem,
                                       int imgWidth, int imgHeight,
                                       int compressionLvl );


        void CloseSensorayDevice(int deviceNumber);

    };

}


#endif //VIDERE_DEV_SENSORAY2253IMAGECAPTURE_H
