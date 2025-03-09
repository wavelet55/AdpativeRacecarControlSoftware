/* ****************************************************************
 * Message
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Randy Direen PhD
 * Date: May 2016
 *
  *******************************************************************/


#ifndef VIDERE_DEV_SIPNPUFFMESSAGE_H
#define VIDERE_DEV_SIPNPUFFMESSAGE_H

#include <string>
#include <RabitMessage.h>

namespace videre
{

    class SipnPuffMessage : public Rabit::RabitMessage
    {

    public:
        //A -100% to +100% value
        //Positive is Blowing/Puff
        //Negative value for Sip
        double SipnPuffPecent = 0;

        //The SipnPuffValue will integrate up/down over time within the range of -100.0 to +100.0
        //The value will be based upon how hard the Sip/Puff is and for how long.
        double SipnPuffIntegralPercent = 0;

    public:
        SipnPuffMessage() : RabitMessage()
        {
            Clear();
        }

        SipnPuffMessage(const SipnPuffMessage& msg)
        {
            *this = msg;
        }


        void setSipnPuffPercent(double val)
        {
            SipnPuffPecent = val < -100.0 ? -100.0 : val > 100.0 ? 100.0 : val;
        }

        void setSipnPuffIntegralPercent(double val)
        {
            SipnPuffIntegralPercent = val < -100.0 ? -100.0 : val > 100.0 ? 100.0 : val;
        }

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<SipnPuffMessage>(new SipnPuffMessage(*this));
            return std::move(clone);
        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            Rabit::RabitMessage::CopyMessage(msg); // call baseclass
            if (msg->GetTypeIndex() == std::type_index(typeid(SipnPuffMessage)))
            {
                //Ensure the Copy process does not loose the mesages'
                //publish subscribe reference.
                std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

                SipnPuffMessage *coMsg = static_cast<SipnPuffMessage *>(msg);
                *this = *coMsg;

                this->SetGlobalPublishSubscribeMessageRef(psmr);
                return true;
            }
            return false;
        }

        virtual void Clear() final
        {
            SipnPuffPecent = 0;
            SipnPuffIntegralPercent = 0;
        }

    };
}




#endif //VIDERE_DEV_SIPNPUFFMESSAGE_H
