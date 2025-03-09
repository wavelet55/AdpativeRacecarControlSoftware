/* ****************************************************************
 * Message
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Aug. 2018
 *
  *******************************************************************/

#ifndef VIDERE_DEV_SIPNPUFFCONTROLMESSAGE_H
#define VIDERE_DEV_SIPNPUFFCONTROLMESSAGE_H

#include <string>
#include <RabitMessage.h>
#include "global_defines.h"

namespace videre
{

    class SipnPuffControlMessage : public Rabit::RabitMessage
    {
    public:
        //Enable or disable the Sip-n-Puff integral value
        //output.  When disabled the integral value is cleared.
        bool EnableSipnPuffIntegration = false;

        //Sip-n-Puff Gains... for the integrator
        //When 1.0 it will take approximately 1 second
        //to integrate from 0 to 100% with the the Sip or Puff
        //output at 100%.
        double SipnPuffBlowGain = 1.0;
        double SipnPuffSuckGain = 1.0;

        //When the SipnPuff value is less than the
        //dead-band the output will be forced to zero.
        double SipnPuffDeadBandPercent = 5.0;


    public:
        SipnPuffControlMessage() : RabitMessage()
        {
            Clear();
        }

        SipnPuffControlMessage(const SipnPuffControlMessage& msg)
        {
            *this = msg;
        }

        virtual void Clear() final
        {
            EnableSipnPuffIntegration = false;
            SipnPuffBlowGain = 1.0;
            SipnPuffSuckGain = 1.0;
            SipnPuffDeadBandPercent = 5.0;
        }

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<SipnPuffControlMessage>(new SipnPuffControlMessage(*this));
            return std::move(clone);
        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            Rabit::RabitMessage::CopyMessage(msg); // call baseclass
            if (msg->GetTypeIndex() == std::type_index(typeid(SipnPuffControlMessage)))
            {
                //Ensure the Copy process does not loose the mesages'
                //publish subscribe reference.
                std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

                SipnPuffControlMessage *coMsg = static_cast<SipnPuffControlMessage *>(msg);
                *this = *coMsg;

                this->SetGlobalPublishSubscribeMessageRef(psmr);
                return true;
            }
            return false;
        }


    };
}

#endif //VIDERE_DEV_SIPNPUFFCONTROLMESSAGE_H
