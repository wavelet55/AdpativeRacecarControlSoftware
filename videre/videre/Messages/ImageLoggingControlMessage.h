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

#ifndef VIDERE_DEV_IMAGELOGGINGCONTROLMESSAGE_H
#define VIDERE_DEV_IMAGELOGGINGCONTROLMESSAGE_H

#include <string>
#include <RabitMessage.h>
#include "global_defines.h"

namespace videre
{
    //The Inertial States of the Vehicle typically sent
    //over from the vehicle.
    class ImageLoggingControlMessage : public Rabit::RabitMessage
    {
    public:
        /// <summary>
        /// Vision logging type.
        /// The specific types of logging Falcon Vision Handles.
        /// </summary>
        VisionLoggingType_e VisionLoggingType = VisionLoggingType_e::LogCompressedImages;

        /// <summary>
        /// Enable or Disable Logging
        /// </summary>
        bool EnableLogging = false;


    public:
        ImageLoggingControlMessage() : RabitMessage()
        {
            Clear();
        }

        ImageLoggingControlMessage(const ImageLoggingControlMessage& msg)
        {
            *this = msg;
        }


        void Clear()
        {
            VisionLoggingType = VisionLoggingType_e::LogCompressedImages;
            EnableLogging = false;
        }

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<ImageLoggingControlMessage>(new ImageLoggingControlMessage(*this));
            return std::move(clone);

        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            bool msgCopied = false;
            if (msg->GetTypeIndex() == std::type_index(typeid(ImageLoggingControlMessage)))
            {
                //Ensure the Copy process does not loose the mesages'
                //publish subscribe reference.
                std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

                Rabit::RabitMessage::CopyMessage(msg); // call baseclass
                ImageLoggingControlMessage *visMsg = static_cast<ImageLoggingControlMessage *>(msg);
                *this = *visMsg;

                this->SetGlobalPublishSubscribeMessageRef(psmr);
                msgCopied = true;
            }
            return msgCopied;
        }

    };

}

#endif //VIDERE_DEV_IMAGELOGGINGCONTROLMESSAGE_H