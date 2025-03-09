/* ****************************************************************
 * Message
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Randy Direen PhD
 * Date: May 2016
 *
  *******************************************************************/

#ifndef VIDERE_DEV_QUATERNIONMESSAGE_H
#define VIDERE_DEV_QUATERNIONMESSAGE_H

#include <string>
#include <RabitMessage.h>
#include "CommonImageProcTypesDefs.h"
#include "Quaternion.h"

namespace videre
{

    class QuaternionMessage : public Rabit::RabitMessage
    {

    public:
        MathLibsNS::Quaternion_t Quaternion;


    public:
        QuaternionMessage() : RabitMessage()
        {
            Clear();
        }

        QuaternionMessage(const QuaternionMessage& msg)
        {
            *this = msg;
        }

        virtual void Clear() final
        {
            Quaternion.MakeIdentity();
        }


        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<QuaternionMessage>(new QuaternionMessage(*this));
            return std::move(clone);
        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            Rabit::RabitMessage::CopyMessage(msg); // call baseclass
            if (msg->GetTypeIndex() == std::type_index(typeid(QuaternionMessage)))
            {
                //Ensure the Copy process does not loose the mesages'
                //publish subscribe reference.
                std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

                QuaternionMessage *coMsg = static_cast<QuaternionMessage *>(msg);
                *this = *coMsg;

                this->SetGlobalPublishSubscribeMessageRef(psmr);
                return true;
            }
            return false;
        }


    };
}


#endif //VIDERE_DEV_QUATERNIONMESSAGE_H
