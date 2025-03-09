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


#ifndef VIDERE_DEV_IMAGECAPTURESTATUSMESSAGE_H
#define VIDERE_DEV_IMAGECAPTURESTATUSMESSAGE_H

#include <string>
#include <RabitMessage.h>
#include "global_defines.h"

namespace videre
{
    //The Inertial States of the Vehicle typically sent
    //over from the vehicle.
    class ImageCaptureStatusMessage : public Rabit::RabitMessage
    {
    public:
        /// <summary>
        /// The Image Capture Enabled/Disabled Status:
        /// If there is a error or NumberOfImagesToCapture has
        /// been reached, or the images have been exhausted,
        /// this will be false.
        /// </summary>
        bool ImageCaptureEnabled;

        /// <summary>
        /// Set to true when the number of images capture
        /// equals the NumberOfImagesToCapture (assuming
        /// NumberOfImagesToCapture > 0);
        /// </summary>
        bool ImageCaptureComplete;

        /// <summary>
        /// Set to true if the source of images is exhausted.
        /// which can occur if images are being pulled from a
        /// directory of images.
        /// </summary>
        bool EndOfImages;

        /// <summary>
        /// Total Number of Images Captured Since Start of Videre
        /// </summary>
        uint32_t TotalNumberOfImagesCaptured;

        /// <summary>
        /// Total Number of Images Captured Since Image Capture
        /// Enabled... Gets reset to zero when image capture is
        /// disabled.
        /// </summary>
        uint32_t CurrentNumberOfImagesCaptured;


        /// <summary>
        /// Average Frames per second base on
        /// CurrentNumberOfImagesCaptured / Time since Last Enabled.
        /// </summary>
        double AverageFramesPerSecond;

        /// <summary>
        /// Source of Images for Image Capture
        /// </summary>
        ImageCaptureSource_e ImageCaptureSource;

        /// <summary>
        /// The Error Number will be non-zero if there is an
        /// error in the image capture process.  The error
        /// number may be used to indicate what the error is.
        /// </summary>
        ImageCaptureError_e  ErrorCode;

    public:
        ImageCaptureStatusMessage() : RabitMessage()
        {
            Clear();
        }

        ImageCaptureStatusMessage(const ImageCaptureStatusMessage& msg)
        {
            *this = msg;
        }

        virtual void Clear() final;

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final;

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final;

    };

}

class ImageCaptureStatusMessage
{

};


#endif //VIDERE_DEV_IMAGECAPTURESTATUSMESSAGE_H
