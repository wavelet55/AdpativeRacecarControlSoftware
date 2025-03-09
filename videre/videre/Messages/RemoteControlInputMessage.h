/* ****************************************************************
 * Message
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Randy Direen PhD
 * Date: May 2016
 *
  *******************************************************************/


#ifndef VIDERE_DEV_REMOTECONTROLINPUTMESSAGE_H
#define VIDERE_DEV_REMOTECONTROLINPUTMESSAGE_H

#include <string>
#include <RabitMessage.h>

namespace videre
{

    class RemoteControlInputMessage : public Rabit::RabitMessage
    {

    public:
        //A -100% to +100% value
        double SteeringControlPercent = 0;

        //A -100% to +100% value
        //Postive numbers are Throttle,
        //Negative numbers are brake
        double ThrottleBrakePercent = 0;

        //A -100% to +100% value
        double Chan_1_Percent = 0;

        //A -100% to +100% value
        double Chan_2_Percent = 0;

        //A -100% to +100% value
        double Chan_3_Percent = 0;

        //A -100% to +100% value
        double Chan_4_Percent = 0;


    public:
        RemoteControlInputMessage() : RabitMessage()
        {
            Clear();
        }

        RemoteControlInputMessage(const RemoteControlInputMessage& msg)
        {
            *this = msg;
        }

        virtual void Clear() final
        {
            SteeringControlPercent = 0;
            ThrottleBrakePercent = 0;
        }

        void setSteeringControlPercent(double val)
        {
            SteeringControlPercent = val < -100.0 ? -100.0 : val > 100.0 ? 100.0 : val;
        }

        void setThrottleBrakePercent(double val)
        {
            ThrottleBrakePercent = val < -100.0 ? -100.0 : val > 100.0 ? 100.0 : val;
        }

        void setChannelNPercent(int chanNo, double val)
        {
            val = val < -100.0 ? -100.0 : val > 100.0 ? 100.0 : val;
            switch(chanNo)
            {
                case 0:
                    Chan_1_Percent = val;
                    break;
                case 1:
                    Chan_2_Percent = val;
                    break;
                case 2:
                    Chan_3_Percent = val;
                    break;
                case 3:
                    Chan_4_Percent = val;
                    break;
            }
        }

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<RemoteControlInputMessage>(new RemoteControlInputMessage(*this));
            return std::move(clone);
        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            Rabit::RabitMessage::CopyMessage(msg); // call baseclass
            if (msg->GetTypeIndex() == std::type_index(typeid(RemoteControlInputMessage)))
            {
                //Ensure the Copy process does not loose the mesages'
                //publish subscribe reference.
                std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

                RemoteControlInputMessage *coMsg = static_cast<RemoteControlInputMessage *>(msg);
                *this = *coMsg;

                this->SetGlobalPublishSubscribeMessageRef(psmr);
                return true;
            }
            return false;
        }


    };
}


#endif //VIDERE_DEV_REMOTECONTROLINPUTMESSAGE_H

