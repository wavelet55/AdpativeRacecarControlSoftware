/* ****************************************************************
 * Message
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Randy Direen PhD
 * Date: May 2016
 *
  *******************************************************************/

#ifndef VIDERE_DEV_HEADORIENTATIONCONTROLMESSAGE_H
#define VIDERE_DEV_HEADORIENTATIONCONTROLMESSAGE_H

#include <string>
#include <RabitMessage.h>
#include "CommonImageProcTypesDefs.h"

namespace videre
{
    enum HeadOrientationOutputSelect_e
    {
        NoOutput,
        ImageProcTrackHead,
        HeadOrientation,
        VehicleOrientation
    };


    class HeadOrientationControlMessage : public Rabit::RabitMessage
    {

    public:
        HeadOrientationOutputSelect_e HeadOrientationOutputSelect;

        bool DisableHeadOrientationKalmanFilter;

        bool DisableVehicleInputToHeadOrientation;

        bool DisableVehicleGravityFeedback;

        //Use this gain if >0 && < 1;
        double VehicleGravityFeedbackGain;

        double HeadOrientation_QVar;

        double HeadOrientation_RVar;


    public:
        HeadOrientationControlMessage() : RabitMessage()
        {
            Clear();
        }

        HeadOrientationControlMessage(const HeadOrientationControlMessage& msg)
        {
            *this = msg;
        }

        virtual void Clear() final
        {
            HeadOrientationOutputSelect = HeadOrientationOutputSelect_e::NoOutput;
            DisableHeadOrientationKalmanFilter = false;
            DisableVehicleInputToHeadOrientation = false;
            DisableVehicleGravityFeedback = false;
            VehicleGravityFeedbackGain = 0;
            HeadOrientation_QVar = 0;
            HeadOrientation_RVar = 0;
        }

        void SetVehicleGravityFeedbackGain(double val)
        {
            VehicleGravityFeedbackGain = val < 0 ? 0 : val >= 1.0 ? 0.999 : val;
        }

        void SetHeadOrientation_QVar(double val)
        {
            HeadOrientation_QVar = val < 0 ? 0 : val > 1000.0 ? 1000.0 : val;
        }

        void SetHeadOrientation_RVar(double val)
        {
            HeadOrientation_RVar = val < 0 ? 0 : val > 1000.0 ? 1000.0 : val;
        }

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<HeadOrientationControlMessage>(new HeadOrientationControlMessage(*this));
            return std::move(clone);
        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            Rabit::RabitMessage::CopyMessage(msg); // call baseclass
            if (msg->GetTypeIndex() == std::type_index(typeid(HeadOrientationControlMessage)))
            {
                //Ensure the Copy process does not loose the mesages'
                //publish subscribe reference.
                std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

                HeadOrientationControlMessage *coMsg = static_cast<HeadOrientationControlMessage *>(msg);
                *this = *coMsg;

                this->SetGlobalPublishSubscribeMessageRef(psmr);
                return true;
            }
            return false;
        }


    };
}


#endif //VIDERE_DEV_HEADORIENTATIONCONTROLMESSAGE_H
