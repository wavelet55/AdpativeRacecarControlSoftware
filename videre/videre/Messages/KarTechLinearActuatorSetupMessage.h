/* ****************************************************************
 * Athr(s): Harry Direen PhD, Randy Direen Phd.
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: June 2018
 *
  *******************************************************************/

#ifndef VIDERE_DEV_KARTECHLINEARACTUATORSETUPMESSAGE_H
#define VIDERE_DEV_KARTECHLINEARACTUATORSETUPMESSAGE_H

#include "global_defines.h"

namespace videre
{

    class KarTechLinearActuatorSetupMessage : public Rabit::RabitMessage
    {
    public:
        LinearActuatorFunction_e FunctionType;

        bool ResetOutputs;
        bool ResetHardwareCfgs;
        bool ResetUserCfgs;
        bool ResetAll;

        //Force the KarTech actuator go go through the auto zero calibration process.
        bool AutoZeroCal;

        //Set up the CAN bus Command and Response addresses.
        bool SetCanCommandResponsIDs;


    public:

        KarTechLinearActuatorSetupMessage() : RabitMessage()
        {
            Clear();
        }

        KarTechLinearActuatorSetupMessage(const KarTechLinearActuatorSetupMessage &msg)
        {
            *this = msg;
        }


        void Clear()
        {
            FunctionType = LinearActuatorFunction_e::LA_Default;
            ResetOutputs = false;
            ResetHardwareCfgs = false;
            ResetUserCfgs = false;
            ResetAll = false;
            AutoZeroCal = false;
            SetCanCommandResponsIDs = false;
        }

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<KarTechLinearActuatorSetupMessage>(new KarTechLinearActuatorSetupMessage(*this));
            return std::move(clone);
        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            bool msgCopied = false;
            if (msg->GetTypeIndex() == std::type_index(typeid(KarTechLinearActuatorSetupMessage)))
            {
                //Ensure the Copy process does not loose the mesages'
                //publish subscribe reference.
                std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

                Rabit::RabitMessage::CopyMessage(msg); // call baseclass
                KarTechLinearActuatorSetupMessage *visMsg = static_cast<KarTechLinearActuatorSetupMessage *>(msg);
                *this = *visMsg;

                this->SetGlobalPublishSubscribeMessageRef(psmr);
                msgCopied = true;
            }
            return msgCopied;
        }

    };


}
#endif //VIDERE_DEV_KARTECHLINEARACTUATORSETUPMESSAGE_H

