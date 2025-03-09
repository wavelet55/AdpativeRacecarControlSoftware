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

#ifndef VIDERE_DEV_IMAGECAPTUREACLASS_H
#define VIDERE_DEV_IMAGECAPTUREACLASS_H

#include <string>
#include <RabitManager.h>
#include "image_plus_metadata_message.h"
#include "ImageCaptureControlMessage.h"
#include "../../Utils/logger.h"
#include "../../Utils/global_defines.h"
#include <SystemTimeClock.h>

using namespace Rabit;

namespace videre
{


    enum ImageCaptureReturnType_e
    {
        ICRT_Error,              //Error in the image capture process no image returned
        ICRT_NoImageCaptured,   //No Image captured.
        IRCT_EndOfImages,        //No more images in the finite source
        IRCT_ImageOnly,         //Only an Image is returned.. not the Vehicle Inetial States or other Info
        IRCT_ImagePlusMetadata, //the Image with all associated metadata
        IRCT_MetadataOnly,         //Only the metadata is returned.. there is no image associate with the metadata
    };

    //Abstract Class for Image Capture.
    //Image capture can pull images from a camera, a directory
    //of image files, the Image Plus Metadata files or other possible
    //sources. This is a top-level class that the Image Capture manager
    //can use to obtain image without being concerned with the source
    //of the images.  There will be concrete classes derived from this
    //class for each image source.
    //Images must be converted to a OpenCV::Mat type by the image capture
    //system.  This means images from a compressed (JPEG or other) source
    //must be de-compressed into a OpenCV::Mat type.
    class ImageCaptureAClass
    {
    protected:

        //Each time a image is obtain, the _imageNumberCounter
        //is incremented.  It keep unique image numbers across a
        //mission run, there is only one counter so that it does not
        //matter where the image is obtained from.  This counter
        //must be incremented by the GetNextImage() method whenever
        //an image is returned.  The ImageNumber must come from this counter.
        static unsigned int _imageNumberCounter;

        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        std::shared_ptr<ConfigData> _config_sptr;

        //Messages

        //The Image Width and Height
        //This is the current/last image captured width and height
        //or may be fixed.
        int _capture_width = 0;
        int _capture_height = 0;
        int _capture_framerate = 30;

        //The time the image is captured.
        double _captureGPSTimeSec = 0;

        bool _IsImageCaptureInitialized = false;

        //This is a re-settable counter to keep track of the
        //current number of images captured.  Can be used to
        //limit the number of images captured.
        unsigned int _numberOfImagesCaptured = 0;

        double _desiredImgCaptureMgrLoopTimeMSec = 250.0;


    public:
        //Use to set a limit on the number of images captured
        //A value of zero allows and infinite number of images to be captured.
        unsigned int MaxNumberOfImagesToCapture = 0;

        //set this flag to true to allow looping back to the
        //start of the list of images.  This is used when images
        //are pulled from files or other finite sources.
        bool LoopBackToStartOfImages = false;

        //Set-Get the desired Framerate.
        //A value of 0 indicates the framerate will be driven
        //by image processing..
        double DesiredFrameRateFPS = 30.0;

        //Status of the Capture Process... basically a copy of the
        //ImageCaptureControlMessage... used for feedback.
        //The ImageCaptureAClass is responsible for keeping the
        //inforamation up-to-date
        ImageCaptureControlMessage ImageCaptureControlStatusMsg;

        ImageCaptureAClass(std::shared_ptr<ConfigData> config)
        {
            _config_sptr = config;

            //Logger Setup
            log4cpp_ = log4cxx::Logger::getLogger("aobj");
            log4cpp_->setAdditivity(false);
            _IsImageCaptureInitialized = false;
            _captureGPSTimeSec = 0;
            ImageCaptureControlStatusMsg.Clear();
        }

        int GetImageWidth()
        {
            return _capture_width;
        }

        int GetImageHeight()
        {
            return _capture_height;
        }

        int GetFrameRate()
        {
            return _capture_framerate;
        }

        //Get the time stamp when the image was captured.
        //Time stamps are GPS Time in Seconds.
        double GetImageCaptureTime()
        {
            return _captureGPSTimeSec;
        }


        bool GetIsImageCaptureInitialized()
        {
            return _IsImageCaptureInitialized;
        }

        double GetDesiredImgCaptureMgrLoopTimeMSec()
        {
            return _desiredImgCaptureMgrLoopTimeMSec;
        }

        void SetImageCaptureTime(double gpsTimeSec)
        {
            _captureGPSTimeSec = gpsTimeSec;
        }

        void SetImageCaptureTimeToNow()
        {
           _captureGPSTimeSec = SystemTimeClock::GetSystemTimeClock()->GetCurrentGpsTimeInSeconds();
        }

        unsigned int GetTotalNumberOfImagesCaptured()
        {
            return _imageNumberCounter;
        }

        unsigned int GetNumberOfImagesCaptured()
        {
            return _numberOfImagesCaptured;
        }

        void ClearNumberOfImagesCaptured()
        {
            _numberOfImagesCaptured = 0;
        }

        //Initalize the Image Capture process.
        //Returns true if error, false if ok.
        virtual bool Initialize() = 0;

        //Initalize the Image Capture process.
        //Returns true if error, false if ok.
        virtual bool Initialize(ImageCaptureControlMessage &imgCapCtrlMsg) = 0;

        //PreCaptureImage is used primarily by webcam and
        //simular devices that have a capture buffer that can be
        //filled faster than the images are being used by image processing.
        //A precapture can be called to capture an image.. if image processing
        //is not ready, the image will be thrown away.
        //PreCaptureImage must be called by the Image Capture manager before
        //GetNextImage.
        virtual bool PreCaptureImage()
        {
            return false;
        }

        virtual ImageCaptureReturnType_e GetNextImage(ImagePlusMetadataMessage *imgPMetadataMsgPtr) = 0;

        //Close the Image Capture Process.
        virtual void Close() = 0;
    };

}
#endif //VIDERE_DEV_IMAGECAPTUREACLASS_H
