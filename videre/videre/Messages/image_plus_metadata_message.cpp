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


#include "image_plus_metadata_message.h"

using namespace VidereImageprocessing;
using namespace GeoCoordinateSystemNS;

namespace videre
{
    std::unique_ptr<Rabit::RabitMessage> ImagePlusMetadataMessage::Clone() const
    {
        auto clone = std::unique_ptr<ImagePlusMetadataMessage>(new ImagePlusMetadataMessage());
        clone->CopyBase(this);
        clone->ImageFrame = ImageFrame.clone();
        clone->ImageNumber = ImageNumber;
        clone->ImageNoPixelsWide = ImageNoPixelsWide;
        clone->ImageNoPixelsHigh = ImageNoPixelsHigh;
        clone->IsFromHOPS = IsFromHOPS;
        clone->ImageCaptureTimeStampSec = ImageCaptureTimeStampSec;
        clone->ImageBufferIndex = ImageBufferIndex;
        clone->VehicleInertialStates = VehicleInertialStates;
        clone->CameraOrientation = CameraOrientation;
        for(int i = 0; i < 4; i++)
            clone->ImageCorners[i] = ImageCorners[i];

        //Note: the ImagePixelLocationToRealWorldLocation is not designed
        //to be cloned.  If the user of the ImagePlusMetadataMessage needs
        //to use the ImagePixelLocationToRealWorldLocation, the user will
        //have to create a new one and set it up.
        clone->PixelToRWTranslator = nullptr;
        //The Cuda Image is not copied.. it must be done explicitly by the user
        //if needed.
        clone->_cudaImageAvailable = false;
        return std::move(clone);
    }

    //Copy the input message to "this" message.
    //Note: the ImagePixelLocationToRealWorldLocation object is not copied across,
    //if the user needs it he/she will have to handle the copy in a way that
    //makes sence.
    bool ImagePlusMetadataMessage::CopyMessage(Rabit::RabitMessage *msg)
    {
        Rabit::RabitMessage::CopyMessage(msg); // call baseclass
        if (msg->GetTypeIndex() == std::type_index(typeid(ImagePlusMetadataMessage)))
        {
            ImagePlusMetadataMessage *inpMsg = static_cast<ImagePlusMetadataMessage *>(msg);

            inpMsg->ImageFrame.copyTo(this->ImageFrame);
            ImageNumber = inpMsg->ImageNumber;
            ImageNoPixelsWide = inpMsg->ImageNoPixelsWide;
            ImageNoPixelsHigh = inpMsg->ImageNoPixelsHigh;
            IsFromHOPS = inpMsg->IsFromHOPS;
            ForceTxImage = inpMsg->ForceTxImage;
            ImageCaptureTimeStampSec = inpMsg->ImageCaptureTimeStampSec;
            ImageBufferIndex = inpMsg->ImageBufferIndex;
            VehicleInertialStates = inpMsg->VehicleInertialStates;
            CameraOrientation = inpMsg->CameraOrientation;
            for(int i = 0; i < 4; i++)
                ImageCorners[i] = inpMsg->ImageCorners[i];

            //Note: PixelToRWTranslator is not copied on purpose.
            //The Cuda Image is not copied.. it must be done explicitly by the user
            //if needed.
            _cudaImageAvailable = false;
            return true;
        }
        return false;
    }

    bool ImagePlusMetadataMessage::CopyMessageWithoutImage(Rabit::RabitMessage *msg)
    {
        Rabit::RabitMessage::CopyMessage(msg); // call baseclass
        if (msg->GetTypeIndex() == std::type_index(typeid(ImagePlusMetadataMessage)))
        {
            ImagePlusMetadataMessage *inpMsg = static_cast<ImagePlusMetadataMessage *>(msg);

            ImageNumber = inpMsg->ImageNumber;
            ImageNoPixelsWide = inpMsg->ImageNoPixelsWide;
            ImageNoPixelsHigh = inpMsg->ImageNoPixelsHigh;
            IsFromHOPS = inpMsg->IsFromHOPS;
            ForceTxImage = inpMsg->ForceTxImage;
            ImageCaptureTimeStampSec = inpMsg->ImageCaptureTimeStampSec;
            ImageBufferIndex = inpMsg->ImageBufferIndex;
            VehicleInertialStates = inpMsg->VehicleInertialStates;
            CameraOrientation = inpMsg->CameraOrientation;
            for(int i = 0; i < 4; i++)
                ImageCorners[i] = inpMsg->ImageCorners[i];

            //Note: PixelToRWTranslator is not copied on purpose.
            //The Cuda Image is not copied.. it must be done explicitly by the user
            //if needed.
            _cudaImageAvailable = false;

            return true;
        }
        return false;
    }

    //The Clear method does not do anything with the
    //PixelToRWTranslator object.
    void ImagePlusMetadataMessage::Clear()
    {
        //Note... do not clear the ImageBufferIndex
        ImageNumber = 0;
        IsFromHOPS = false;
        ForceTxImage = false;
        ImageCaptureTimeStampSec = 0;
        VehicleInertialStates.Clear();
        CameraOrientation.Clear();
        for(int i = 0; i < 4; i++)
            ImageCorners[i].Clear();
    }

    std::string ImagePlusMetadataMessage::ToString() const
    {
        std::ostringstream os;
        os << "ImagePlusMetadataMessage: " << IsFromHOPS << ", "
           << ImageCaptureTimeStampSec;
        return os.str();
    }


    bool ImagePlusMetadataMessage::CopyImageToCudaMat()
    {
        bool _cudaImageAvailable = false;
#ifdef CUDA
        try
        {
            _cudaImageFrame.upload(ImageFrame);
            _cudaImageAvailable = true;
        }
        catch (std::exception &e)
        {
            _cudaImageAvailable = false;
            //LOGERROR("ImagePlusMetadataMessage: upload image to cuda matrix Exception: " << e.what());
        }
#endif
        return _cudaImageAvailable;
    }

    //This only works on machines that have NVidia Cuda
    //support.  Returns true if image is copied, false otherwise
    //The Cuda Image is copied into the ImageFrame
    bool ImagePlusMetadataMessage::CopyImageFromCudaMat()
    {
        bool imageCopied = false;
#ifdef CUDA
        try
        {
            if(_cudaImageAvailable)
            {
                _cudaImageFrame.download(ImageFrame);
                imageCopied = true;
            }
        }
        catch (std::exception &e)
        {
            _cudaImageAvailable = false;
            //LOGERROR("ImagePlusMetadataMessage: upload image to cuda matrix Exception: " << e.what());
        }
#endif
        return imageCopied;
    }


}
