/* ****************************************************************
 * Vision Bridge
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Aug. 2016
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *******************************************************************/

#ifndef VIDERE_DEV_IMAGECAPTURECONTROLMESSAGE_H
#define VIDERE_DEV_IMAGECAPTURECONTROLMESSAGE_H

#include <string>
#include <RabitMessage.h>
#include "global_defines.h"

namespace videre
{
    //The Inertial States of the Vehicle typically sent
    //over from the vehicle.
    class ImageCaptureControlMessage : public Rabit::RabitMessage
    {
    public:
        /// <summary>
        /// The Image Capture On flag must be true
        /// to capture images and do any of the processing
        /// of images.  Setting this flag to false will disable
        /// all other image processing.
        /// Changes to all configureation parameters will only
        /// </summary>
        bool ImageCaptureEnabled = false;

        /// <summary>
        /// If NumberOfImagesToCapture is greater than zero,
        /// the Vision System will capture the set number of images,
        /// process the images, and then disable ImageCapture
        /// If NumberOfImagesToCapture is zero or less... the Vision
        /// System will continue capturing and processing images until
        /// ImageCaptureEnabled is disabled by the user.
        /// </summary>
        uint32_t NumberOfImagesToCapture = 0;

        /// <summary>
        /// Desired Frames per second.
        /// In general the frame rate will be controlled by the time
        /// to do the image processing.  If image processing is disabled
        /// or very quick, this value can be used to slow down the image
        /// capture and processing rate.  Set to a higher value to get the
        /// max frame rate that is supported by the image processing time.
        /// </summary>
        double DesiredFramesPerSecond = 100;

        /// <summary>
        /// Desired Image Width and Height
        /// Where Image Width and Height can be controlled, use
        /// these parameters.  If set to zero, the Videre Config
        /// info will be used.
        /// </summary>
        uint32_t DesiredImageWidth = 0;
        uint32_t DesiredImageHeight = 0;

        /// <summary>
        /// Source of Images for Image Capture
        /// </summary>
        ImageCaptureSource_e ImageCaptureSource;


        /// <summary>
        /// Image Capture Format:
        ///    MJPEG, YUV... Depends on webcam capability
        /// </summary>
        ImageCaptureFormat_e ImageCaptureFormat;

        /// <summary>
        /// Primary Configuration String for the ImageCaptureSource.
        /// This could be the Device number for the WebCam,
        ///  or it could be the Directory of Image Files.
        /// If this is empty the Videre Config info will be used.
        /// </summary>
        std::string ImageCaptureSourceConfigPri;

        /// <summary>
        /// Secondary Configuration String for the ImageCaptureSource.
        /// This could be the Device number for the WebCam,
        ///  or it could be the Directory of Image Files.
        /// If this is empty the Videre Config info will be used.
        /// </summary>
        std::string ImageCaptureSourceConfigSec;

        /// <summary>
        /// When images are being captured from a finite
        /// source such as a directory of image files,
        /// if this flag is true, Image capture will restart
        /// capture from the start of the source after reaching
        /// the end.
        /// </summary>
        bool ImageSourceLoopAround = false;

        bool AutoFocusEnable = false;

    public:
        ImageCaptureControlMessage() : RabitMessage()
        {
            Clear();
        }

        ImageCaptureControlMessage(const ImageCaptureControlMessage& msg)
        {
            *this = msg;
        }

        virtual void Clear() final;

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final;

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final;

    };

}

#endif //VIDERE_DEV_IMAGECAPTURECONTROLMESSAGE_H
