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

#include "ImageCaptureControlMessage.h"

namespace videre
{


    void ImageCaptureControlMessage::Clear()
    {
        ImageCaptureEnabled = false;
        NumberOfImagesToCapture = 0;
        DesiredFramesPerSecond = 100;
        DesiredImageWidth = 0;
        DesiredImageHeight = 0;
        ImageCaptureSource = ImageCaptureSource_e::ImageCaptureSource_NoChange;
        ImageCaptureFormat = ImageCaptureFormat_e::Unknown;
        ImageCaptureSourceConfigPri = "";
        ImageCaptureSourceConfigSec = "";
        ImageSourceLoopAround = false;
        AutoFocusEnable = false;
    }

    std::unique_ptr<Rabit::RabitMessage> ImageCaptureControlMessage::Clone() const
    {
        auto clone = std::unique_ptr<ImageCaptureControlMessage>(new ImageCaptureControlMessage(*this));
        return std::move(clone);
    }

    bool ImageCaptureControlMessage::CopyMessage(Rabit::RabitMessage *msg)
    {
        bool msgCopied = false;
        if (msg->GetTypeIndex() == std::type_index(typeid(ImageCaptureControlMessage)))
        {
            Rabit::RabitMessage::CopyMessage(msg); // call baseclass
            ImageCaptureControlMessage *aMsg = static_cast<ImageCaptureControlMessage *>(msg);
            ImageCaptureEnabled = aMsg->ImageCaptureEnabled;
            NumberOfImagesToCapture = aMsg->NumberOfImagesToCapture;
            DesiredFramesPerSecond = aMsg->DesiredFramesPerSecond;
            DesiredImageWidth = aMsg->DesiredImageWidth;
            DesiredImageHeight = aMsg->DesiredImageHeight;
            ImageCaptureSource = aMsg->ImageCaptureSource;
            ImageCaptureFormat = aMsg->ImageCaptureFormat;
            ImageCaptureSourceConfigPri = aMsg->ImageCaptureSourceConfigPri;
            ImageCaptureSourceConfigSec = aMsg->ImageCaptureSourceConfigSec;
            ImageSourceLoopAround = aMsg->ImageSourceLoopAround;
            AutoFocusEnable = aMsg->AutoFocusEnable;
            msgCopied = true;
        }
        return msgCopied;
    }



}

