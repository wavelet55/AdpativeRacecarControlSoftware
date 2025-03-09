/* ****************************************************************
 * Athr(s): Harry Direen PhD, Randy Direen Phd.
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: March. 2016
 *
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
 *******************************************************************/


#ifndef VIDERE_DEV_CAMERACALSTATUSMESSAGE_H
#define VIDERE_DEV_CAMERACALSTATUSMESSAGE_H

#include <iostream>
#include <string>
#include <RabitMessage.h>
#include "global_defines.h"

namespace videre
{

    //Camera Calibration Command Message
    class CameraCalStatusMessage : public Rabit::RabitMessage
    {
    public:
        CameraCalibrationState_e CameraCalibrationState;

        int NumberOfImagesCaptured;

        std::string CameraCalStatusMsg;

        bool ImageOk;

    public:

        CameraCalStatusMessage() : RabitMessage()
        {
            Clear();
        }

        CameraCalStatusMessage(const CameraCalStatusMessage &msg)
        {
            *this = msg;
        }

        void Clear()
        {
            CameraCalibrationState = CameraCalibrationState_e::CCalState_Reset;
            NumberOfImagesCaptured = 0;
            CameraCalStatusMsg = "";
            ImageOk = false;
        }

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<CameraCalStatusMessage>(new CameraCalStatusMessage(*this));
            return std::move(clone);
        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            bool msgCopied = false;
            if (msg->GetTypeIndex() == std::type_index(typeid(CameraCalStatusMessage)))
            {
                //Ensure the Copy process does not loose the mesages'
                //publish subscribe reference.
                std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

                Rabit::RabitMessage::CopyMessage(msg); // call baseclass
                CameraCalStatusMessage *visMsg = static_cast<CameraCalStatusMessage *>(msg);
                *this = *visMsg;

                this->SetGlobalPublishSubscribeMessageRef(psmr);
                msgCopied = true;
            }
            return msgCopied;
        }

    };

}







#endif //VIDERE_DEV_CAMERACALSTATUSMESSAGE_H
