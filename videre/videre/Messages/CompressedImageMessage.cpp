/* ****************************************************************
* Message
* DireenTech Inc.  (www.direentech.com)
* Athr: Harry Direen PhD
* Date: Aug 2016
*
* Developed under contract for:
* Academy Center for UAS Research
* Department of Electrical and Computer Engineering
       * HQ USAFA/DFEC
* 2354 Fairchild Drive
* USAF Academy, CO 80840-6236
*
*******************************************************************/

#include "CompressedImageMessage.h"

namespace videre
{

    std::unique_ptr<Rabit::RabitMessage> CompressedImageMessage::Clone() const
    {
        auto clone = std::unique_ptr<CompressedImageMessage>(new CompressedImageMessage());
        clone->CopyBase(this);
        clone->ImageFormatType = this->ImageFormatType;
        clone->ImageNumber = this->ImageNumber;
        clone->GpsTimeStampSec = GpsTimeStampSec;
        for(int i = 0; i < ImageBuffer.size(); i++)
        {
            clone->ImageBuffer[i] = ImageBuffer[i];
        }
        return std::move(clone);
    }

    bool CompressedImageMessage::CopyMessage(Rabit::RabitMessage *msg)
    {
        Rabit::RabitMessage::CopyMessage(msg); // call baseclass
        if (msg->GetTypeIndex() == std::type_index(typeid(CompressedImageMessage)))
        {
            //Ensure the Copy process does not loose the mesages'
            //publish subscribe reference.
            std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

            CompressedImageMessage *inpMsg = static_cast<CompressedImageMessage *>(msg);
            ImageFormatType = inpMsg->ImageFormatType;
            ImageNumber = inpMsg->ImageNumber;
            GpsTimeStampSec = inpMsg->GpsTimeStampSec;
            ImageBuffer.clear();
            for(int i = 0; i < ImageBuffer.size(); i++)
            {
                ImageBuffer[i] = inpMsg->ImageBuffer[i];
            }

            this->SetGlobalPublishSubscribeMessageRef(psmr);
            return true;
        }
        return false;
    }


    std::string CompressedImageMessage::ToString() const
    {
        std::ostringstream os;
        os << "CompressedImageMessage: ImageNumber: " << ImageNumber
        << ", ImageFormatType: " << ImageFormatType;
        return os.str();
    }

    std::string CompressedImageMessage::ImageFormatAsString()
    {
        std::string imgFormat;
        if(ImageFormatType == ImageFormatType_e::ImgFType_JPEG)
        {
            imgFormat = "JPEG";
        }
        else
        {
            imgFormat = "RAW";
        }
        return imgFormat;
    }

}