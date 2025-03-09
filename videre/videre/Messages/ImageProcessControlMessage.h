/* ****************************************************************
 * Vision Bridge
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Aug. 2016
 * 
 * Developed under contract for:
 * Academy Center for UAS Research 
 * Department of Electrical and Computer Engineering 
 * HQ USAFA/DFEC 
 * 2354 Fairchild Drive 
 * USAF Academy, CO 80840-6236 
 *******************************************************************/

#ifndef VIDERE_DEV_IMAGEPROCESSCONTROLMESSAGE_H
#define VIDERE_DEV_IMAGEPROCESSCONTROLMESSAGE_H

#include "global_defines.h"

namespace videre
{
    //The Inertial States of the Vehicle typically sent
    //over from the vehicle.
    class ImageProcessControlMessage : public Rabit::RabitMessage
    {
    public:

    public:

        /// <summary>
        /// GPU Processing On
        /// Turns on/off all GPU Image Processing.
        /// Finer level control of the Image processing is given below.
        /// </summary>
        bool GPUProcessingEnabled;

        /// <summary>
        /// Target Image Processing On/Off
        /// </summary>
        bool TargetImageProcessingEnabled;


        /// <summary>
        /// Vision Processing Mode
        /// This is the High-Level Vision / Image Processing mode of operation.
        /// </summary>
        VisionProcessingMode_e VisionProcessingMode;

        /// <summary>
        /// Target Image Processing Mode
        /// Various types of Target Processing could be supported,
        /// this enum selects the active Target Processing Mode.
        /// </summary>
        TargetProcessingMode_e TargetProcessingMode;

        /// <summary>
        /// Target Image Processing On/Off
        /// </summary>
        bool GPSDeniedProcessingEnabled;

        /// <summary>
        /// GPS Denied Processing Mode
        /// Various types of Target Processing could be supported,
        /// this enum selects the active Target Processing Mode.
        /// </summary>
        GPSDeniedProcessingMode_e GPSDeniedProcessingMode;


        ImageProcessControlMessage() : RabitMessage()
        {
            Clear();
        }

        ImageProcessControlMessage(const ImageProcessControlMessage &msg)
        {
            *this = msg;
        }

        void Clear()
        {
            GPUProcessingEnabled = false;
            TargetImageProcessingEnabled = false;
            GPSDeniedProcessingEnabled = false;
            VisionProcessingMode = VisionProcessingMode_e::VisionProcMode_None;
            TargetProcessingMode = TargetProcessingMode_e::TgtProcMode_None;
            GPSDeniedProcessingMode = GPSDeniedProcessingMode_e::GpsDeniedMode_None;
        }

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<ImageProcessControlMessage>(new ImageProcessControlMessage(*this));
            return std::move(clone);

        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            bool msgCopied = false;
            if (msg->GetTypeIndex() == std::type_index(typeid(ImageProcessControlMessage)))
            {
                //Ensure the Copy process does not loose the mesages'
                //publish subscribe reference.
                std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

                Rabit::RabitMessage::CopyMessage(msg); // call baseclass
                ImageProcessControlMessage *visMsg = static_cast<ImageProcessControlMessage *>(msg);
                *this = *visMsg;

                this->SetGlobalPublishSubscribeMessageRef(psmr);
                msgCopied = true;
            }
            return msgCopied;
        }

    };

}


#endif //VIDERE_DEV_IMAGEPROCESSCONTROLMESSAGE_H
