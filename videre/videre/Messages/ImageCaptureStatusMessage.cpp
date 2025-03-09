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


#include "ImageCaptureStatusMessage.h"

namespace videre
{


    void ImageCaptureStatusMessage::Clear()
    {
        ImageCaptureEnabled = false;
        ImageCaptureComplete = false;
        EndOfImages = false;
        TotalNumberOfImagesCaptured = 0;
        CurrentNumberOfImagesCaptured = 0;
        AverageFramesPerSecond = 0;
        ImageCaptureSource = ImageCaptureSource_e::ImageCaptureSource_OpenCVWebCam;
        ErrorCode = ImageCaptureError_e::ImageCaptureError_None;
    }

    std::unique_ptr<Rabit::RabitMessage> ImageCaptureStatusMessage::Clone() const
    {
        auto clone = std::unique_ptr<ImageCaptureStatusMessage>(new ImageCaptureStatusMessage(*this));
        return std::move(clone);
    }

    bool ImageCaptureStatusMessage::CopyMessage(Rabit::RabitMessage *msg)
    {
        bool msgCopied = false;
        if (msg->GetTypeIndex() == std::type_index(typeid(ImageCaptureStatusMessage)))
        {
            //Ensure the Copy process does not loose the mesages'
            //publish subscribe reference.
            std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

            Rabit::RabitMessage::CopyMessage(msg); // call baseclass
            ImageCaptureStatusMessage *aMsg = static_cast<ImageCaptureStatusMessage *>(msg);
            *this = *aMsg;

            this->SetGlobalPublishSubscribeMessageRef(psmr);
            msgCopied = true;
        }
        return msgCopied;
    }


}

