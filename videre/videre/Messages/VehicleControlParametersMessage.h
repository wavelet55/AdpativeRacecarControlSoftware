/* ****************************************************************
 * Message
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Aug. 2018
 *
  *******************************************************************/

#ifndef VIDERE_DEV_VEHICLECONTROLPARAMETERSMESSAGE_H
#define VIDERE_DEV_VEHICLECONTROLPARAMETERSMESSAGE_H

#include <string>
#include <RabitMessage.h>
#include "global_defines.h"

namespace videre
{

    class VehicleControlParametersMessage : public Rabit::RabitMessage
    {
    public:
        double SipnPuffBlowGain = 1.0;
        double SipnPuffSuckGain = 1.0;
        double SipnPuffDeadBandPercent = 5.0;

        bool ReverseSipnPuffThrottleBrake = false;
        double ThrottleSipnPuffGain = 1.0;
        double BrakeSipnPuffGain = 1.0;

        bool ThrottleBrakeHeadTiltEnable = true;
        double ThrottleBrakeHeadTiltForwardDeadbandDegrees = 15.0;
        double ThrottleBrakeHeadTiltBackDeadbandDegrees = 10.0;
        double ThrottleHeadTiltGain = 1.0;
        double BrakeHeadTiltGain = 1.0;

        //Steering Angle or Torque Control
        bool UseSteeringAngleControl = false;
        double SteeringDeadband = 2.5;
        double SteeringControlGain = 1.0;
        double SteeringBiasAngleDegrees = 0.0;
        double RCSteeringGain = 0.25;

        double MaxLRHeadRotationDegrees = 60.0;

        //The HeadLeftRighLPFOrder can be: 0, 2, 4, 6
        //a zero by-passes the filter
        int HeadLeftRighLPFOrder = 4;
        double HeadLeftRighLPFCutoffFreqHz = 5.0;

        double SteeringAngleFeedback_Kp = 1.0;
        double SteeringAngleFeedback_Kd = 0.0;
        double SteeringAngleFeedback_Ki = 0.0;


    public:
        VehicleControlParametersMessage() : RabitMessage()
        {
            Clear();
        }

        VehicleControlParametersMessage(const VehicleControlParametersMessage& msg)
        {
            *this = msg;
        }

        virtual void Clear() final
        {
            SipnPuffBlowGain = 1.0;
            SipnPuffSuckGain = 1.0;
            SipnPuffDeadBandPercent = 5.0;

            ReverseSipnPuffThrottleBrake = false;
            ThrottleSipnPuffGain = 1.0;
            BrakeSipnPuffGain = 1.0;

            ThrottleBrakeHeadTiltEnable = true;
            ThrottleBrakeHeadTiltForwardDeadbandDegrees = 15.0;
            ThrottleBrakeHeadTiltBackDeadbandDegrees = 10.0;
            ThrottleHeadTiltGain = 1.0;
            BrakeHeadTiltGain = 1.0;

            UseSteeringAngleControl = true;
            SteeringDeadband = 2.5;
            SteeringControlGain = 1.0;
            SteeringBiasAngleDegrees = 0.0;
            RCSteeringGain = 0.25;

            MaxLRHeadRotationDegrees = 60.0;

            HeadLeftRighLPFOrder = 4;
            HeadLeftRighLPFCutoffFreqHz = 5.0;

            SteeringAngleFeedback_Kp = 10.0;
            SteeringAngleFeedback_Kd = 0.0;
            SteeringAngleFeedback_Ki = 0.0;
        }

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<VehicleControlParametersMessage>(new VehicleControlParametersMessage(*this));
            return std::move(clone);
        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            Rabit::RabitMessage::CopyMessage(msg); // call baseclass
            if (msg->GetTypeIndex() == std::type_index(typeid(VehicleControlParametersMessage)))
            {
                //Ensure the Copy process does not loose the mesages'
                //publish subscribe reference.
                std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

                VehicleControlParametersMessage *coMsg = static_cast<VehicleControlParametersMessage *>(msg);
                *this = *coMsg;

                this->SetGlobalPublishSubscribeMessageRef(psmr);
                return true;
            }
            return false;
        }


    };
}

#endif //VIDERE_DEV_VEHICLECONTROLPARAMETERSMESSAGE_H
