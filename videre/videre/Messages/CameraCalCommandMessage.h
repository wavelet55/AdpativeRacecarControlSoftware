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


#ifndef VIDERE_DEV_CAMERACALCOMMANDMESSAGE_H
#define VIDERE_DEV_CAMERACALCOMMANDMESSAGE_H

#include <iostream>
#include <string>
#include <RabitMessage.h>
#include "global_defines.h"

namespace videre
{

    //Camera Calibration Command Message
    class CameraCalCommandMessage : public Rabit::RabitMessage
    {
    public:
        CameraCalibrationType_e CameraCalibrationType;

        CameraCalCmd_e CameraCalCmd;

        /// <summary>
        /// Stream Images On/Off
        /// </summary>
        std::string CameraCalBaseFilename;

        int NumberOfRows;

        int NumberOfCols;

        double SquareSizeMilliMeters;

        double YawCorrectionDegrees;
        double PitchCorrectionDegrees;
        double RollCorrectionDegrees;
        double DelXCorrectionCentiMeters;
        double DelYCorrectionCentiMeters;
        double DelZCorrectionCentiMeters;

    public:

        CameraCalCommandMessage() : RabitMessage()
        {
            Clear();
        }

        CameraCalCommandMessage(const CameraCalCommandMessage &msg)
        {
            *this = msg;
        }

        void Clear()
        {
            CameraCalibrationType = CameraCalibrationType_e::CameraCal_2DPlaneCheckerBoard;
            CameraCalCmd = CameraCalCmd_e::CCalCmd_NullCmd;
            CameraCalBaseFilename = "";
            NumberOfRows = 7;
            NumberOfCols = 6;
            SquareSizeMilliMeters = 25.4;
            YawCorrectionDegrees = 0;
            PitchCorrectionDegrees = 0;
            RollCorrectionDegrees = 0;
            DelXCorrectionCentiMeters = 0;
            DelYCorrectionCentiMeters = 0;
            DelZCorrectionCentiMeters = 0;
        }

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<CameraCalCommandMessage>(new CameraCalCommandMessage(*this));
            return std::move(clone);
        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            bool msgCopied = false;
            if (msg->GetTypeIndex() == std::type_index(typeid(CameraCalCommandMessage)))
            {
                //Ensure the Copy process does not loose the mesages'
                //publish subscribe reference.
                std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

                Rabit::RabitMessage::CopyMessage(msg); // call baseclass
                CameraCalCommandMessage *visMsg = static_cast<CameraCalCommandMessage *>(msg);
                *this = *visMsg;

                this->SetGlobalPublishSubscribeMessageRef(psmr);
                msgCopied = true;
            }
            return msgCopied;
        }

    };

}

#endif //VIDERE_DEV_CAMERACALCOMMANDMESSAGE_H
