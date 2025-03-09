/* ****************************************************************
 * Message
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Randy Direen PhD
 * Date: May 2016
 *
  *******************************************************************/

#ifndef VIDERE_DEV_VEHICLESWITCHINPUTMESSAGE_H
#define VIDERE_DEV_VEHICLESWITCHINPUTMESSAGE_H

#include <string>
#include <RabitMessage.h>
#include "CommonImageProcTypesDefs.h"

namespace videre
{

    class VehicleSwitchInputMessage : public Rabit::RabitMessage
    {

    public:

        bool DriverControlEnabled;

        bool AuxSwitchEnabled;


    public:
        VehicleSwitchInputMessage() : RabitMessage()
        {
            Clear();
        }

        VehicleSwitchInputMessage(const VehicleSwitchInputMessage& msg)
        {
            *this = msg;
        }

        virtual void Clear() final
        {
            DriverControlEnabled = false;
            AuxSwitchEnabled = false;
        }


        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<VehicleSwitchInputMessage>(new VehicleSwitchInputMessage(*this));
            return std::move(clone);
        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            Rabit::RabitMessage::CopyMessage(msg); // call baseclass
            if (msg->GetTypeIndex() == std::type_index(typeid(VehicleSwitchInputMessage)))
            {
                //Ensure the Copy process does not loose the mesages'
                //publish subscribe reference.
                std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

                VehicleSwitchInputMessage *coMsg = static_cast<VehicleSwitchInputMessage *>(msg);
                *this = *coMsg;

                this->SetGlobalPublishSubscribeMessageRef(psmr);
                return true;
            }
            return false;
        }


    };
}

#endif //VIDERE_DEV_VEHICLESWITCHINPUTMESSAGE_H
