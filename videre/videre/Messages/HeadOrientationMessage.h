/* ****************************************************************
 * Athr(s): Harry Direen PhD, Randy Direen Phd.
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: June 2018
 *
  *******************************************************************/

#ifndef VIDERE_DEV_HEADORIENTATIONMESSAGE_H
#define VIDERE_DEV_HEADORIENTATIONMESSAGE_H

#include "global_defines.h"
#include "RollPitchYaw_t.h"

using namespace MathLibsNS;

namespace videre
{

    //Camera Calibration Command Message
    class HeadOrientationMessage : public Rabit::RabitMessage
    {
    public:
        RollPitchYaw_t HeadRollPitchYawAnlges;

        double CovarianceNorm = 0;

        double IMUTimeStampSec;

    public:

        HeadOrientationMessage() : RabitMessage()
        {
            Clear();
        }

        HeadOrientationMessage(const HeadOrientationMessage &msg)
        {
            *this = msg;
        }

        void SetEulerAnglesDegrees(double xAxisDeg, double yAxisDeg, double zAxisDeg)
        {
            HeadRollPitchYawAnlges.SetRollDegrees(xAxisDeg);
            HeadRollPitchYawAnlges.SetPitchDegrees(yAxisDeg);
            HeadRollPitchYawAnlges.SetYawDegrees(zAxisDeg);
        }

        void SetEulerAnglesRadians(double xAxisRad, double yAxisRad, double zAxisRad)
        {
            HeadRollPitchYawAnlges.SetRollRadians(xAxisRad);
            HeadRollPitchYawAnlges.SetPitchRadians(yAxisRad);
            HeadRollPitchYawAnlges.SetYawRadians(zAxisRad);
        }

        void Clear()
        {
            HeadRollPitchYawAnlges.Clear();
            CovarianceNorm = 0;
            IMUTimeStampSec = 0;
        }

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<HeadOrientationMessage>(
                    new HeadOrientationMessage(*this));
            return std::move(clone);
        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            bool msgCopied = false;
            if(msg->GetTypeIndex() == std::type_index(typeid(HeadOrientationMessage)))
            {
                //Ensure the Copy process does not loose the mesages'
                //publish subscribe reference.
                std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

                Rabit::RabitMessage::CopyMessage(msg); // call baseclass
                HeadOrientationMessage *visMsg = static_cast<HeadOrientationMessage *>(msg);
                *this = *visMsg;

                this->SetGlobalPublishSubscribeMessageRef(psmr);
                msgCopied = true;
            }
            return msgCopied;
        }

    };

}

#endif //VIDERE_DEV_HEADORIENTATIONMESSAGE_H

