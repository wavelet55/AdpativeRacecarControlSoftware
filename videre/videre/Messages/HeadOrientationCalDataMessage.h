/* ****************************************************************
 * Athr(s): Harry Direen PhD, Randy Direen Phd.
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: June 2018
 *
  *******************************************************************/

#ifndef VIDERE_DEV_HEADORIENTATIONCALDATAMESSAGE_H
#define VIDERE_DEV_HEADORIENTATIONCALDATAMESSAGE_H

#include "global_defines.h"
#include "RollPitchYaw_t.h"
#include "CommonImageProcTypesDefs.h"
#include "Quaternion.h"

using namespace MathLibsNS;

namespace videre
{

    //Camera Calibration Command Message
    class HeadOrientationCalDataMessage : public Rabit::RabitMessage
    {
    public:
        ImageProcLibsNS::HeadOrientationCalData_t CalData;

    public:

        HeadOrientationCalDataMessage() : RabitMessage()
        {
            Clear();
        }

        HeadOrientationCalDataMessage(const HeadOrientationCalDataMessage &msg)
        {
            *this = msg;
        }

        void Clear()
        {
            CalData.Clear();
        }

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<HeadOrientationCalDataMessage>(
                    new HeadOrientationCalDataMessage(*this));
            return std::move(clone);
        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            bool msgCopied = false;
            if(msg->GetTypeIndex() == std::type_index(typeid(HeadOrientationCalDataMessage)))
            {
                //Ensure the Copy process does not loose the mesages'
                //publish subscribe reference.
                std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

                Rabit::RabitMessage::CopyMessage(msg); // call baseclass
                HeadOrientationCalDataMessage *visMsg = static_cast<HeadOrientationCalDataMessage *>(msg);
                *this = *visMsg;

                this->SetGlobalPublishSubscribeMessageRef(psmr);
                msgCopied = true;
            }
            return msgCopied;
        }

    };

}
#endif //VIDERE_DEV_HEADORIENTATIONCALDATAMESSAGE_H
