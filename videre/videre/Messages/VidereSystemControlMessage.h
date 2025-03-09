/* ****************************************************************
 * Message
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Aug. 2018
 *
  *******************************************************************/

#ifndef VIDERE_DEV_VIDERESYSTEMCONTROLMESSAGE_H
#define VIDERE_DEV_VIDERESYSTEMCONTROLMESSAGE_H

#include <string>
#include <RabitMessage.h>
#include "global_defines.h"

namespace videre
{

    class VidereSystemControlMessage : public Rabit::RabitMessage
    {
    public:
        VidereSystemStates_e SystemState;

        bool StartProcess;

        bool PauseProces;

        bool StopProcess;

        bool HeadControlEnable;
        bool ThrottleControlEnable;
        bool BrakeControlEnable;
        bool BCIControlEnable;


        //Feedback Items
        VidereSystemStatus_e SystemStatus;

        uint32_t StatusCounter;

    public:
        VidereSystemControlMessage() : RabitMessage()
        {
            Clear();
        }

        VidereSystemControlMessage(const VidereSystemControlMessage& msg)
        {
            *this = msg;
        }

        virtual void Clear() final
        {
            SystemState = VidereSystemStates_e::VSS_Init;
            StartProcess = false;
            PauseProces = false;
            StopProcess = false;
            HeadControlEnable = true;  //Default is enabled
            ThrottleControlEnable = true;
            BrakeControlEnable = true;
            BCIControlEnable = false;
            StatusCounter = 0;
            SystemStatus = VidereSystemStatus_e::VSX_Ok;
        }

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<VidereSystemControlMessage>(new VidereSystemControlMessage(*this));
            return std::move(clone);
        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            Rabit::RabitMessage::CopyMessage(msg); // call baseclass
            if (msg->GetTypeIndex() == std::type_index(typeid(VidereSystemControlMessage)))
            {
                //Ensure the Copy process does not loose the mesages'
                //publish subscribe reference.
                std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

                VidereSystemControlMessage *coMsg = static_cast<VidereSystemControlMessage *>(msg);
                *this = *coMsg;

                this->SetGlobalPublishSubscribeMessageRef(psmr);
                return true;
            }
            return false;
        }


    };
}


#endif //VIDERE_DEV_VIDERESYSTEMCONTROLMESSAGE_H
