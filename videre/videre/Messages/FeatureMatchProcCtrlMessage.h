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


#ifndef VIDERE_DEV_FEATUREMATCHPROCCTRLMESSAGE_H
#define VIDERE_DEV_FEATUREMATCHPROCCTRLMESSAGE_H

#include <iostream>
#include <string>
#include <RabitMessage.h>
#include "global_defines.h"

namespace videre
{
    //Message for controlling the Feature Matching Process
    //in the Vision Processing Manager.  This largely used for developing
    //Image processing routing and testing them within the NVidia Enviroment.
    class FeatureMatchProcCtrlMessage : public Rabit::RabitMessage
    {
    public:
        FeatureMatchingProcCmd_e FeatureMatchingProcCmd;

        FeatureExtractionTypeRoutine_e FeatureExtractionTypeRoutine;

        FeatureMatchTypeRoutine_e FeatureMatchTypeRoutine;

        FMImagePostProcessMethod_e FMImagePostProcessMethod;

        //Use NVidia GPU/Cuda processing or standard processing.
        //Helps with measuring perfomance differences between the two.
        bool UseGPUProcessing;

        //Generic Parameters for the processing routines
        int ParamI_1;
        int ParamI_2;
        int ParamI_3;
        int ParamI_4;
        int ParamI_5;
        int ParamI_6;
        int ParamI_7;
        int ParamI_8;
        int ParamI_9;

        double ParamF_10;
        double ParamF_11;
        double ParamF_12;
        double ParamF_13;
        double ParamF_14;
        double ParamF_15;
        double ParamF_16;
        double ParamF_17;
        double ParamF_18;
        double ParamF_19;


        FeatureMatchProcCtrlMessage() : RabitMessage()
        {
            Clear();
        }

        FeatureMatchProcCtrlMessage(const FeatureMatchProcCtrlMessage &msg)
        {
            *this = msg;
        }

        void Clear()
        {
            FeatureMatchingProcCmd = FeatureMatchingProcCmd_e::FMPCmd_NullCmd;
            FeatureExtractionTypeRoutine = FeatureExtractionTypeRoutine_e::FETR_ORB;
            FeatureMatchTypeRoutine = FeatureMatchTypeRoutine_e::FMTR_BruteForce;
            FMImagePostProcessMethod = FMImagePostProcessMethod_e::FMIPPM_None;
            UseGPUProcessing = false;

            ParamI_1 = 0;
            ParamI_2 = 0;
            ParamI_3 = 0;
            ParamI_4 = 0;
            ParamI_5 = 0;
            ParamI_6 = 0;
            ParamI_7 = 0;
            ParamI_8 = 0;
            ParamI_9 = 0;

            ParamF_10 = 0;
            ParamF_11 = 0;
            ParamF_12 = 0;
            ParamF_13 = 0;
            ParamF_14 = 0;
            ParamF_15 = 0;
            ParamF_16 = 0;
            ParamF_17 = 0;
            ParamF_18 = 0;
            ParamF_19 = 0;

        }

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<FeatureMatchProcCtrlMessage>(new FeatureMatchProcCtrlMessage(*this));
            return std::move(clone);
        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            bool msgCopied = false;
            if (msg->GetTypeIndex() == std::type_index(typeid(FeatureMatchProcCtrlMessage)))
            {
                //Ensure the Copy process does not loose the mesages'
                //publish subscribe reference.
                std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

                Rabit::RabitMessage::CopyMessage(msg); // call baseclass
                FeatureMatchProcCtrlMessage *visMsg = static_cast<FeatureMatchProcCtrlMessage *>(msg);
                *this = *visMsg;

                this->SetGlobalPublishSubscribeMessageRef(psmr);
                msgCopied = true;
            }
            return msgCopied;
        }




    };

}
#endif //VIDERE_DEV_FEATUREMATCHPROCCTRLMESSAGE_H
