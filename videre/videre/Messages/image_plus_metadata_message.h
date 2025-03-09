/* ****************************************************************
 * Message
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: July 2016
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
  *******************************************************************/

#ifndef VIDERE_DEV_IMAGE_PLUS_METADATA_MESSAGE_H
#define VIDERE_DEV_IMAGE_PLUS_METADATA_MESSAGE_H

#include <iostream>
#include <string>
#include <RabitMessage.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include "VehicleInertialStatesMessage.h"
#include "CameraOrientationMessage.h"
#include "Utilities/ImagePixelLocationToRealWorldLocation.h"

namespace videre
{
    //This Message carries a OpenCV Mat Image plus Metadata associated
    //with the image.
    //The Vision Process manager and the Stream Record Manager maintains
    //pools of these messages.  With the Image and Metadata there can be a fair amount
    //of data contained in the message so in general copies sould be avoided and
    //there are some places in the message where a copy of a pointer is made rather
    //than a "deep" copy... so be careful how the message is used in the system.
    class ImagePlusMetadataMessage : public Rabit::RabitMessage
    {

    public:
        cv::Mat ImageFrame;             //This is the Image in OpenCV Mat Format

        //Every Image has a unique number associated with it... created
        //when the image is captured/
        unsigned int ImageNumber = 0;       //Every Image has a unique

        bool IsFromHOPS = false;        /* Did we get this from HOPS or is it generated */

        bool ForceTxImage = false;      //Force sending this image out so it is not skipped.

        //The time the image was captured.
        //Timestamp should be in GPS Time in Seconds.
        double ImageCaptureTimeStampSec = 0;

        int ImageNoPixelsWide = 0;
        int ImageNoPixelsHigh = 0;

        //Vehicle Inertial states at the time the Image was captured.
        VehicleInertialStatesMessage VehicleInertialStates;

        //Camera Orientation Information
        CameraOrientationMessage CameraOrientation;

        //A reference to the Image Pixel to Real-World Translation
        //Routine.  A refernce is here because the Image Capture Manager
        //sets the object up... makes sure it has the latest camera calibration
        //data and then typically calculates the corner locations of the image.
        //When this message is passed on the Image Processing, this object can
        //be used by Image Processing to calculate target or other item locations
        //on the ground.  The oject will already be setup with the UAV location and
        //and attitude along with the Camera Orientation information.
        //It is assumed that the Stream Recorder Manager is not making use of this
        //Translation object... it should be ok if it is... but there is a slight chance
        //of an issue if the Stream Recorder and Image Processing get out of sync.

        std::shared_ptr<VidereImageprocessing::ImagePixelLocationToRealWorldLocation> PixelToRWTranslator = nullptr;

        //An array of Image Corner Locations.
        XYZCoord_t ImageCorners[4];


    private:
        //If this Message is part of an array of ImagePlusMetadataMessages
        //this is the index into the array.  It is kept for house-keeping
        //and error checking purposes.
        //It is only set by the constructor that sets it.
        int ImageBufferIndex = 0;

        bool _cudaImageAvailable = false;

        //If CUDA is supported.. add a GPU Matrix
        //Note:  the cuda image is never copied explicitly...
        //it must be done by the user if required.
#ifdef CUDA
        cv::cuda::GpuMat _cudaImageFrame;
#endif


    public:
        ImagePlusMetadataMessage() : RabitMessage()
        {
            Clear();
        }

        //If the ImagePlusMetadataMessage is part of a array/buffer of ImagePlusMetadataMessages
        //use this constructor and pass in the array index.
        ImagePlusMetadataMessage(int imageBufferIndex) : RabitMessage()
        {
            ImageBufferIndex = imageBufferIndex;
            Clear();
        }

        int getImageBufferIndex()
        {
            return ImageBufferIndex;
        }

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final;

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final;

        bool CopyMessageWithoutImage(Rabit::RabitMessage *msg);

        virtual void Clear() final;

        virtual std::string ToString() const final;

        //This only works on machines that have NVidia Cuda
        //support.  Returns true if image is copied, false otherwise
        bool CopyImageToCudaMat();

        //This only works on machines that have NVidia Cuda
        //support.  Returns true if image is copied, false otherwise
        //The Cuda Image is copied into the ImageFrame
        bool CopyImageFromCudaMat();

        void ReleaseCudaMatMem()
        {
#ifdef CUDA
            _cudaImageFrame.release();
#endif
        }

#ifdef CUDA
        cv::cuda::GpuMat &GetCudaImageMat()
        {
            return _cudaImageFrame;
        }
#endif

    };

    typedef std::shared_ptr<ImagePlusMetadataMessage> ImagePlusMetadataMessageSP;
    typedef std::unique_ptr<ImagePlusMetadataMessage> ImagePlusMetadataMessageUP;

}

#endif //VIDERE_DEV_IMAGE_PLUS_METADATA_MESSAGE_H
