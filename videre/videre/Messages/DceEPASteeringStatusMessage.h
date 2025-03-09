/* ****************************************************************
 * Athr(s): Harry Direen PhD, Randy Direen Phd.
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: June 2018
 *
  *******************************************************************/
  
#ifndef VIDERE_DEV_DCEEPASTEERINGSTATUSMESSAGE_H
#define VIDERE_DEV_DCEEPASTEERINGSTATUSMESSAGE_H

#include "global_defines.h"

namespace videre
{

    //Camera Calibration Command Message
    class DceEPASteeringStatusMessage : public Rabit::RabitMessage
    {
    public:
        double MotorCurrentAmps;
        double PWMDutyCyclePercent;
        double MotorTorquePercent;
        double SupplyVoltage;
        double TempDegC;
        double SteeringAngleDeg;     //0 degrees is neutral position
        SteeringTorqueMap_e SteeringTorqueMapSetting;
        uint32_t SwitchPosition;     //0 - 15
        int TorqueA;
        int TorqueB ;

        uint32_t ErrorCode;
        uint32_t StatusFlags;
        uint32_t LimitFlags;
        bool ManualExtControl;

        //When the TorqueA or TorqueB values get away from the 127 nominal value
        //it indicates the safety driver is trying to take control of the car.
        bool DriverTorqueHit;


    public:

        DceEPASteeringStatusMessage() : RabitMessage()
        {
            Clear();
        }

        DceEPASteeringStatusMessage(const DceEPASteeringStatusMessage &msg)
        {
            *this = msg;
        }


        void Clear()
        {
            MotorCurrentAmps = 0;
            PWMDutyCyclePercent = 0;
            MotorTorquePercent = 0;
            SupplyVoltage = 0;
            TempDegC = 0;
            SteeringAngleDeg = 0;
            SteeringTorqueMapSetting = SteeringTorqueMap_e::STM_Disable;
            SwitchPosition = 0;
            TorqueA = 0;
            TorqueB = 0;
            ErrorCode = 0;
            StatusFlags = 0;
            LimitFlags = 0;
            ManualExtControl = false;
            DriverTorqueHit = false;
        }


        bool IsProgramPaused()
        {
             return (StatusFlags & 0x01) != 0;
        }

        bool IsMotorMovingRight()
        {
            return (StatusFlags & 0x02) != 0;
        }

       bool IsMotorMovingLeft()
       {
           return (StatusFlags & 0x04) != 0;
       }

        bool IsHostModeActive()
        {
            return (StatusFlags & 0x08) != 0;
        }

        bool IsFaultLight()
        {
            return (StatusFlags & 0x10) != 0;
        }

        bool IsSteeringAtLeftHandStop()
        {
            return (LimitFlags & 0x01) != 0;
        }

        bool IsSteeringAtRightHandStop()
        {
            return (LimitFlags & 0x02) != 0;
        }

        bool IsOverTemp()
        {
            return (LimitFlags & 0x04) != 0;
        }



        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<DceEPASteeringStatusMessage>(new DceEPASteeringStatusMessage(*this));
            return std::move(clone);
        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            bool msgCopied = false;
            if (msg->GetTypeIndex() == std::type_index(typeid(DceEPASteeringStatusMessage)))
            {
                //Ensure the Copy process does not loose the mesages'
                //publish subscribe reference.
                std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

                Rabit::RabitMessage::CopyMessage(msg); // call baseclass
                DceEPASteeringStatusMessage *visMsg = static_cast<DceEPASteeringStatusMessage *>(msg);
                *this = *visMsg;

                this->SetGlobalPublishSubscribeMessageRef(psmr);
                msgCopied = true;
            }
            return msgCopied;
        }

    };


}

#endif //VIDERE_DEV_DCEEPASTEERINGSTATUSMESSAGE_H
