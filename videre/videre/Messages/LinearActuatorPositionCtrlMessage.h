/* ****************************************************************
 * Athr(s): Harry Direen PhD, Randy Direen Phd.
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: June 2018
 *
  *******************************************************************/

#ifndef VIDERE_DEV_LINEARACTUATORPOSITIONCTRLMESSAGE_H
#define VIDERE_DEV_LINEARACTUATORPOSITIONCTRLMESSAGE_H

#include "global_defines.h"
#include <RabitMessage.h>
#include <PublishSubscribeMessage.h>


namespace videre
{

    //Camera Calibration Command Message
    class LinearActuatorPositionCtrlMessage : public Rabit::RabitMessage
    {
    public:

        LinearActuatorFunction_e FunctionType;

        bool ClutchEnable;
        bool MotorEnable;

        bool ManualExtControl;

        bool ActuatorSetupMode;

        //Actuator Report Status and Errors
        uint8_t ErrorFlags;

        //Status is a Kar-Tech Internal Use status
        uint8_t Status;

        double TempDegC;

    private:
        //Linear Actuator Postion in Percent [0 - 100.0]
        //The actual actuator postion will depend upon the Min and Max Position settings
        //and whether the actuator pushes or pulls in the postitive direction.
        double _positionPercent;

        //This is primarily for feedback of the motor current.
        double _motorCurrentAmps;

    public:

        LinearActuatorPositionCtrlMessage() : RabitMessage()
        {
            Clear();
        }

        LinearActuatorPositionCtrlMessage(const LinearActuatorPositionCtrlMessage &msg)
        {
            *this = msg;
        }

        bool IsMotorOverLoad() { return ErrorFlags & 0x01 != 0; }
        bool IsMotorOpenLoad() { return ErrorFlags & 0x04 != 0; }
        bool IsClutchOverload() { return ErrorFlags & 0x02 != 0; }
        bool IsClutchOpenLoad() { return ErrorFlags & 0x08 != 0; }
        bool IsPositionReachError() { return ErrorFlags & 0x10 != 0; }
        bool IsHWSensorError1() { return ErrorFlags & 0x20 != 0; }
        bool IsHWSensorError2() { return ErrorFlags & 0x40 != 0; }


        double getPositionPercent()
        {
            return _positionPercent;
        }

        void setPositionPercent(double val)
        {
            _positionPercent = val < 0.0 ? 0.0 : val > 100.0 ? 100.0 : val;
        }

        void setPositionPercent(double val, bool enable)
        {
            _positionPercent = val < 0.0 ? 0.0 : val > 100.0 ? 100.0 : val;
            ClutchEnable = enable;
            MotorEnable = enable;
        }

        double getMotorCurrentAmps()
        { return _motorCurrentAmps; }

        void setMotorCurrentAmps(double val)
        {
            _motorCurrentAmps = val < 0.0 ? 0.0 : val > 100.0 ? 100.0 : val;
        }

        void Clear()
        {
            FunctionType = LinearActuatorFunction_e::LA_Default;
            ClutchEnable = false;
            MotorEnable = false;
            ManualExtControl = false;
            ActuatorSetupMode = false;
            _positionPercent = 0.0;
            _motorCurrentAmps = 0.0;
            ErrorFlags = 0;
            Status = 0;
            TempDegC = 22.0;
        }

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<LinearActuatorPositionCtrlMessage>(
                    new LinearActuatorPositionCtrlMessage(*this));
            return std::move(clone);
        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            bool msgCopied = false;
            if(msg->GetTypeIndex() == std::type_index(typeid(LinearActuatorPositionCtrlMessage)))
            {
                //Ensure the Copy process does not loose the mesages'
                //publish subscribe reference.
                std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

                Rabit::RabitMessage::CopyMessage(msg); // call baseclass
                LinearActuatorPositionCtrlMessage *visMsg = static_cast<LinearActuatorPositionCtrlMessage *>(msg);
                *this = *visMsg;

                this->SetGlobalPublishSubscribeMessageRef(psmr);
                msgCopied = true;
            }
            return msgCopied;
        }

    };

}
#endif //VIDERE_DEV_LINEARACTUATORPOSITIONCTRLMESSAGE_H
