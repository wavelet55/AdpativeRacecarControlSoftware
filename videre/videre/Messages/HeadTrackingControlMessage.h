/* ****************************************************************
 * Message
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Randy Direen PhD
 * Date: May 2016
 *
  *******************************************************************/


#ifndef VIDERE_DEV_HEADTRACKINGCONTROLMESSAGE_H
#define VIDERE_DEV_HEADTRACKINGCONTROLMESSAGE_H

#include <string>
#include <RabitMessage.h>
#include "CommonImageProcTypesDefs.h"

namespace videre
{

    class HeadTrackingControlMessage : public Rabit::RabitMessage
    {

    public:
        ImageProcLibsNS::HeadTrackingParameters_t HeadTrackingParameters;

        ImageProcLibsNS::HeadTrackingImageDisplayType_e HeadTrackingImageDisplayType;

        int GlyphModelIndex = 0;

    public:
        HeadTrackingControlMessage() : RabitMessage()
        {
            Clear();
        }

        HeadTrackingControlMessage(const HeadTrackingControlMessage& msg)
        {
            *this = msg;
        }

        virtual void Clear() final
        {
            HeadTrackingParameters.SetDefaults();
            HeadTrackingImageDisplayType = ImageProcLibsNS::HeadTrackingImageDisplayType_e::HTID_None;
            GlyphModelIndex = 0;
        }


        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<HeadTrackingControlMessage>(new HeadTrackingControlMessage(*this));
            return std::move(clone);
        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            Rabit::RabitMessage::CopyMessage(msg); // call baseclass
            if (msg->GetTypeIndex() == std::type_index(typeid(HeadTrackingControlMessage)))
            {
                //Ensure the Copy process does not loose the mesages'
                //publish subscribe reference.
                std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

                HeadTrackingControlMessage *coMsg = static_cast<HeadTrackingControlMessage *>(msg);
                *this = *coMsg;

                this->SetGlobalPublishSubscribeMessageRef(psmr);
                return true;
            }
            return false;
        }


    };
}


#endif //VIDERE_DEV_HEADTRACKINGCONTROLMESSAGE_H
