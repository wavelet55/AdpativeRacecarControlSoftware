/* ****************************************************************
 * Athr(s): Harry Direen PhD, Randy Direen Phd.
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: June 2018
 *
  *******************************************************************/

#ifndef VIDERE_DEV_STEERINGTORQUECTRLMESSAGE_H
#define VIDERE_DEV_STEERINGTORQUECTRLMESSAGE_H

#include "global_defines.h"

namespace videre
{

    //The steering can be controlled from the torque input
    //to the EPAS Steering system or from a desired steering angle.
    class SteeringTorqueCtrlMessage : public Rabit::RabitMessage
    {
    private:
        //Steering Torque is in percent:  [-100.0 to 100.0]
        //Negative is to the left, postive is to the right
        double _steeringTorquePercent;

        uint32_t _steeringTorqueMap;

    public:

        double getSteeringTorquePercent() { return _steeringTorquePercent; }

        //Setting a torque value also sets the Steering angle flag and
        //clears the Steering angle value.
        void setSteeringTorquePercent(double val)
        {
            _steeringTorquePercent = val < -100.0 ? -100.0 : val > 100.0 ? 100.0 : val;
        }


        uint32_t getSteeringTorqueMap() { return _steeringTorqueMap; }

        void setSteeringTorqueMap(uint32_t val)
        {
            _steeringTorqueMap = val > 5 ? 5 : val;
        }

        bool SteeringControlEnabled;

        bool ManualExtControl;

    public:

        SteeringTorqueCtrlMessage() : RabitMessage()
        {
            Clear();
        }

        SteeringTorqueCtrlMessage(const SteeringTorqueCtrlMessage &msg)
        {
            *this = msg;
        }


        void Clear()
        {
            _steeringTorquePercent = 0;
            _steeringTorqueMap = 0;  //A zero disables the steering control.
            SteeringControlEnabled = false;
            ManualExtControl = false;
        }

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<SteeringTorqueCtrlMessage>(new SteeringTorqueCtrlMessage(*this));
            return std::move(clone);
        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            bool msgCopied = false;
            if (msg->GetTypeIndex() == std::type_index(typeid(SteeringTorqueCtrlMessage)))
            {
                //Ensure the Copy process does not loose the mesages'
                //publish subscribe reference.
                std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

                Rabit::RabitMessage::CopyMessage(msg); // call baseclass
                SteeringTorqueCtrlMessage *visMsg = static_cast<SteeringTorqueCtrlMessage *>(msg);
                *this = *visMsg;

                this->SetGlobalPublishSubscribeMessageRef(psmr);
                msgCopied = true;
            }
            return msgCopied;
        }

    };


}
#endif //VIDERE_DEV_STEERINGTORQUECTRLMESSAGE_H
