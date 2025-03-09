/* ****************************************************************
 * Athr(s): Harry Direen PhD, Randy Direen Phd.
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: Nov. 2017
 *
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
 *******************************************************************/


#ifndef VIDERE_DEV_CAMERAPARAMETERSSETUPMESSAGE_H
#define VIDERE_DEV_CAMERAPARAMETERSSETUPMESSAGE_H

#include <iostream>
#include <string>
#include <RabitMessage.h>
#include "global_defines.h"

namespace videre
{

    //Camera Calibration Command Message
    class CameraParametersSetupMessage : public Rabit::RabitMessage
    {
    public:
        ImageCaptureFormat_e ImageCaptureFormat;

        /// <summary>
        /// Mode
        /// </summary>
        u_int32_t Mode;

        /// <summary>
        /// FrameWidth
        /// </summary>
        u_int32_t FrameWidth;

        /// <summary>
        /// FrameWidth
        /// </summary>
        u_int32_t FrameHeight;

        /// <summary>
        /// FrameRateFPS
        /// </summary>
        double FrameRateFPS;

        /// <summary>
        /// Autofocus on --> true
        /// </summary>
        bool Autofocus;

        /// <summary>
        /// Focus
        /// </summary>
        double FocusValue;

        /// <summary>
        /// Brightness
        /// </summary>
        double Brightness;

        /// <summary>
        /// Contrast
        /// </summary>
        double Contrast;

        /// <summary>
        /// Saturation
        /// </summary>
        double Saturation;

        /// <summary>
        /// Hue
        /// </summary>
        double Hue;

        /// <summary>
        /// Gain
        /// </summary>
        double Gain;

        /// <summary>
        /// Exposure
        /// </summary>
        double Exposure;

        /// <summary>
        /// Exposure
        /// </summary>
        double WhiteBallanceBlue;

        /// <summary>
        /// Exposure
        /// </summary>
        double WhiteBallanceRed;

        /// <summary>
        /// ExternalTrigger on --> true
        /// </summary>
        bool ExternalTrigger;

    public:

        CameraParametersSetupMessage() : RabitMessage()
        {
            Clear();
        }

        CameraParametersSetupMessage(const CameraParametersSetupMessage &msg)
        {
            *this = msg;
        }

        void Clear()
        {
            ImageCaptureFormat = ImageCaptureFormat_e::Unknown;
            Mode = 0;
            FrameWidth = 640;
            FrameHeight = 480;
            FrameRateFPS = 30;
            Autofocus = true;
            FocusValue = 0;
            Brightness = 0;
            Contrast = 0;
            Saturation = 0;
            Hue = 0;
            Gain = 0;
            Exposure = 0;
            WhiteBallanceBlue = 0;
            WhiteBallanceRed = 0;
            ExternalTrigger = false;
        }

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<CameraParametersSetupMessage>(new CameraParametersSetupMessage(*this));
            return std::move(clone);
        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            bool msgCopied = false;
            if (msg->GetTypeIndex() == std::type_index(typeid(CameraParametersSetupMessage)))
            {
                //Ensure the Copy process does not loose the mesages'
                //publish subscribe reference.
                std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

                Rabit::RabitMessage::CopyMessage(msg); // call baseclass
                CameraParametersSetupMessage *visMsg = static_cast<CameraParametersSetupMessage *>(msg);
                *this = *visMsg;

                this->SetGlobalPublishSubscribeMessageRef(psmr);
                msgCopied = true;
            }
            return msgCopied;
        }

    };

}

#endif //VIDERE_DEV_CAMERACALCOMMANDMESSAGE_H



