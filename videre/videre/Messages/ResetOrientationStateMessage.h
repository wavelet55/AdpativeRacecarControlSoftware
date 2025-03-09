/* ****************************************************************
 * Message
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Randy Direen PhD
 * Date: May 2016
 *
  *******************************************************************/

#ifndef VIDERE_DEV_RESETORIENTATIONSTATEMESSAGE_H
#define VIDERE_DEV_RESETORIENTATIONSTATEMESSAGE_H

#include <string>
#include <RabitMessage.h>
#include "CommonImageProcTypesDefs.h"

namespace videre
{

    class ResetOrientationStateMessage : public Rabit::RabitMessage
    {

    public:

        bool ResetHeadOrientationState;

        bool ResetVehicleOrientationState;


    public:
        ResetOrientationStateMessage() : RabitMessage()
        {
            Clear();
        }

        ResetOrientationStateMessage(const ResetOrientationStateMessage& msg)
        {
            *this = msg;
        }

        virtual void Clear() final
        {
            ResetHeadOrientationState = false;
            ResetVehicleOrientationState = false;
        }


        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<ResetOrientationStateMessage>(new ResetOrientationStateMessage(*this));
            return std::move(clone);
        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            Rabit::RabitMessage::CopyMessage(msg); // call baseclass
            if (msg->GetTypeIndex() == std::type_index(typeid(ResetOrientationStateMessage)))
            {
                //Ensure the Copy process does not loose the mesages'
                //publish subscribe reference.
                std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

                ResetOrientationStateMessage *coMsg = static_cast<ResetOrientationStateMessage *>(msg);
                *this = *coMsg;

                this->SetGlobalPublishSubscribeMessageRef(psmr);
                return true;
            }
            return false;
        }


    };
}

#endif //VIDERE_DEV_RESETORIENTATIONSTATEMESSAGE_H
