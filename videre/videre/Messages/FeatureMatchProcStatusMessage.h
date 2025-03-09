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

#ifndef VIDERE_DEV_FEATUREMATCHPROCSTATUSMESSAGE_H
#define VIDERE_DEV_FEATUREMATCHPROCSTATUSMESSAGE_H

#include <iostream>
#include <string>
#include <RabitMessage.h>
#include "global_defines.h"

namespace videre
{
    //Message for controlling the Feature Matching Process
    //in the Vision Processing Manager.  This largely used for developing
    //Image processing routing and testing them within the NVidia Enviroment.
    class FeatureMatchProcStatusMessage : public Rabit::RabitMessage
    {
    public:

        FeatureMatchingState_e FeatureMatchingState;

        FeatureExtractionTypeRoutine_e FeatureExtractionTypeRoutine;

        FeatureMatchTypeRoutine_e FeatureMatchTypeRoutine;

        int NumberOfImagesCaptured;

        std::string StatusMessage;

        //General purpose Process time 1... typically in seconds.
        double ProcessTimer_1;

        //General purpose Process time 2... typically in seconds.
        double ProcessTimer_2;

        //General Purpose status values.
        int StatusValI_1;
        int StatusValI_2;
        int StatusValI_3;
        int StatusValI_4;
        int StatusValI_5;
        int StatusValI_6;
        int StatusValI_7;
        int StatusValI_8;
        int StatusValI_9;

        double StatusValF_10;
        double StatusValF_11;
        double StatusValF_12;
        double StatusValF_13;
        double StatusValF_14;
        double StatusValF_15;
        double StatusValF_16;
        double StatusValF_17;
        double StatusValF_18;
        double StatusValF_19;


        FeatureMatchProcStatusMessage() : RabitMessage()
        {
            Clear();
        }

        FeatureMatchProcStatusMessage(const FeatureMatchProcStatusMessage &msg)
        {
            *this = msg;
        }

        void Clear()
        {
            FeatureMatchingState = FeatureMatchingState_e::FMPState_Reset;
            FeatureExtractionTypeRoutine = FeatureExtractionTypeRoutine_e::FETR_ORB;
            FeatureMatchTypeRoutine = FeatureMatchTypeRoutine_e::FMTR_BruteForce;
            StatusMessage = "";
            NumberOfImagesCaptured = 0;
            ProcessTimer_1 = 0;
            ProcessTimer_2 = 0;

            StatusValI_1 = 0;
            StatusValI_2 = 0;
            StatusValI_3 = 0;
            StatusValI_4 = 0;
            StatusValI_5 = 0;
            StatusValI_6 = 0;
            StatusValI_7 = 0;
            StatusValI_8 = 0;
            StatusValI_9 = 0;

            StatusValF_10 = 0;
            StatusValF_11 = 0;
            StatusValF_12 = 0;
            StatusValF_13 = 0;
            StatusValF_14 = 0;
            StatusValF_15 = 0;
            StatusValF_16 = 0;
            StatusValF_17 = 0;
            StatusValF_18 = 0;
            StatusValF_19 = 0;

        }

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<FeatureMatchProcStatusMessage>(new FeatureMatchProcStatusMessage(*this));
            return std::move(clone);
        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            bool msgCopied = false;
            if (msg->GetTypeIndex() == std::type_index(typeid(FeatureMatchProcStatusMessage)))
            {
                //Ensure the Copy process does not loose the mesages'
                //publish subscribe reference.
                std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

                Rabit::RabitMessage::CopyMessage(msg); // call baseclass
                FeatureMatchProcStatusMessage *visMsg = static_cast<FeatureMatchProcStatusMessage *>(msg);
                *this = *visMsg;

                this->SetGlobalPublishSubscribeMessageRef(psmr);
                msgCopied = true;
            }

            return msgCopied;
        }

    };

}



#endif //VIDERE_DEV_FEATUREMATCHPROCSTATUSMESSAGE_H
