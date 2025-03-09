/* ****************************************************************
 * Athr(s): Harry Direen PhD, Randy Direen Phd.
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: June 2018
 *
  *******************************************************************/

#ifndef VIDERE_DEV_KARTECHLINEARACTUATORPARAMSMESSAGE_H
#define VIDERE_DEV_KARTECHLINEARACTUATORPARAMSMESSAGE_H

#include "global_defines.h"

namespace videre
{

    class KarTechLinearActuatorParamsMessage : public Rabit::RabitMessage
    {
    public:
        LinearActuatorFunction_e FunctionType;

    private:
        double MinPositionInches;
        double MaxPositionInches;

        double MotorMaxCurrentLimitAmps;

        uint32_t PositionReachedErrorTimeMSec;

        uint32_t FeedbackCtrl_KP;
        uint32_t FeedbackCtrl_KI;
        uint32_t FeedbackCtrl_KD;
        uint32_t FeedbackCtrl_CLFreq;
        double FeedbackCtrl_ErrDeadbandInces;

        //Motor PWM Setting
        uint32_t Motor_MinPWM;
        uint32_t Motor_MaxPWM;
        uint32_t Motor_pwmFreq;

    public:

        KarTechLinearActuatorParamsMessage() : RabitMessage()
        {
            Clear();
        }

        KarTechLinearActuatorParamsMessage(const KarTechLinearActuatorParamsMessage &msg)
        {
            *this = msg;
        }

        double getMinPositionInches() { return MinPositionInches; }
        void setMinPositionInches(double val)
        {
            MinPositionInches = val < 0 ? 0 : val > 2.9 ? 2.9 : val;
        }

        double getMaxPositionInches() { return MaxPositionInches; }
        void setMaxPositionInches(double val)
        {
            MaxPositionInches = val < 0.1 ? 0.1 : val > 3.0 ? 3.0 : val;
        }

        double getPositionRange()
        {
            double rng = MaxPositionInches - MinPositionInches;
            rng = rng < 0.1 ? 0.1 : rng > 3.0 ? 3.0 : rng;
        }

        double getMotorMaxCurrentLimitAmps() { return MotorMaxCurrentLimitAmps; }
        void setMotorMaxCurrentLimitAmps(double val)
        {
            MotorMaxCurrentLimitAmps = val < 2.5 ? 2.5 : val > 65.0 ? 65.0: val;
        }

        uint32_t getPositionReachedErrorTimeMSec() { return PositionReachedErrorTimeMSec; }
        void setPositionReachedErrorTimeMSec(uint32_t val)
        {
            PositionReachedErrorTimeMSec = val < 5 ? 5 : val > 1000 ? 1000: val;
        }

        uint32_t getFeedbackCtrl_KP() { return  FeedbackCtrl_KP; }
        void setFeedbackCtrl_KP(uint32_t val)
        {
            FeedbackCtrl_KP = val < 10 ? 10 : val > 10000 ? 10000: val;
        }

        uint32_t getFeedbackCtrl_KI() { return  FeedbackCtrl_KI; }
        void setFeedbackCtrl_KI(uint32_t val)
        {
            FeedbackCtrl_KI = val < 0 ? 0 : val > 10000 ? 10000: val;
        }

        uint32_t getFeedbackCtrl_KD() { return  FeedbackCtrl_KD; }
        void setFeedbackCtrl_KD(uint32_t val)
        {
            FeedbackCtrl_KD = val < 0 ? 0 : val > 100 ? 100: val;
        }

        uint32_t getFeedbackCtrl_CLFreq() { return  FeedbackCtrl_CLFreq; }
        void setFeedbackCtrl_CLFreq(uint32_t val)
        {
            FeedbackCtrl_CLFreq = val < 30 ? 30 : val > 100 ? 100: val;
        }

        double getFeedbackCtrl_ErrDeadbandInces() { return  FeedbackCtrl_ErrDeadbandInces; }
        void setFeedbackCtrl_ErrDeadbandInces(double val)
        {
            FeedbackCtrl_ErrDeadbandInces = val < 0.001 ? 0.001 : val > 0.1 ? 0.1: val;
        }

        //Motor PWM Setting
        uint32_t getMotor_MinPWM() { return  Motor_MinPWM; }
        void setMotor_MinPWM(uint32_t val)
        {
            Motor_MinPWM = val < 0 ? 0 : val > 25 ? 25: val;
        }

        uint32_t getMotor_MaxPWM() { return  Motor_MaxPWM; }
        void setMotor_MaxPWM(uint32_t val)
        {
            Motor_MaxPWM = val < 25 ? 25 : val > 100 ? 100: val;
        }

        uint32_t getMotor_pwmFreq() { return  Motor_pwmFreq; }
        void setMotor_pwmFreq(uint32_t val)
        {
            Motor_pwmFreq = val < 1000 ? 1000 : val > 5000 ? 5000: val;
        }

        void Clear()
        {
            FunctionType = LinearActuatorFunction_e::LA_Default;
            MinPositionInches = 0.0;
            MaxPositionInches = 3.0;
            MotorMaxCurrentLimitAmps = 65.0;

            PositionReachedErrorTimeMSec = 40;

            FeedbackCtrl_KP = 1000;
            FeedbackCtrl_KI = 1000;
            FeedbackCtrl_KD = 10;
            FeedbackCtrl_CLFreq = 60;
            FeedbackCtrl_ErrDeadbandInces = 0.05;

            //Motor PWM Setting
            Motor_MinPWM = 20;
            Motor_MaxPWM = 90;
            Motor_pwmFreq = 2000;
        }

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<KarTechLinearActuatorParamsMessage>(new KarTechLinearActuatorParamsMessage(*this));
            return std::move(clone);
        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            bool msgCopied = false;
            if (msg->GetTypeIndex() == std::type_index(typeid(KarTechLinearActuatorParamsMessage)))
            {
                //Ensure the Copy process does not loose the mesages'
                //publish subscribe reference.
                std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

                Rabit::RabitMessage::CopyMessage(msg); // call baseclass
                KarTechLinearActuatorParamsMessage *visMsg = static_cast<KarTechLinearActuatorParamsMessage *>(msg);
                *this = *visMsg;

                this->SetGlobalPublishSubscribeMessageRef(psmr);
                msgCopied = true;
            }
            return msgCopied;
        }

    };


}
#endif //VIDERE_DEV_KARTECHLINEARACTUATORPARAMSMESSAGE_H
