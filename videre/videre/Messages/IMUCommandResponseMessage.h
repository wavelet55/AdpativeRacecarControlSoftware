/* ****************************************************************
 * Athr(s): Harry Direen PhD, Randy Direen Phd.
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: June 2018
 *
  *******************************************************************/

#ifndef VIDERE_DEV_IMUCOMMANDRESPONSEMESSAGE_H
#define VIDERE_DEV_IMUCOMMANDRESPONSEMESSAGE_H

#include "global_defines.h"

namespace videre
{

    //Camera Calibration Command Message
    class IMUCommandResponseMessage : public Rabit::RabitMessage
    {
    public:

        LinearActuatorFunction_e FunctionType;

        bool IMURemoteCtrlEnable;
        std::string CmdRspMsg;

    public:

        IMUCommandResponseMessage() : RabitMessage()
        {
            Clear();
        }

        IMUCommandResponseMessage(const IMUCommandResponseMessage &msg)
        {
            *this = msg;
        }

        void Clear()
        {
            IMURemoteCtrlEnable = false;
            CmdRspMsg = "";
        }

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<IMUCommandResponseMessage>(
                    new IMUCommandResponseMessage());
            clone->CopyBase(this);
            clone->IMURemoteCtrlEnable = IMURemoteCtrlEnable;
            clone->CmdRspMsg = CmdRspMsg;
            return std::move(clone);
        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            bool msgCopied = false;
            if(msg->GetTypeIndex() == std::type_index(typeid(IMUCommandResponseMessage)))
            {
                Rabit::RabitMessage::CopyMessage(msg); // call baseclass
                IMUCommandResponseMessage *visMsg = static_cast<IMUCommandResponseMessage *>(msg);
                IMURemoteCtrlEnable = visMsg->IMURemoteCtrlEnable;
                CmdRspMsg = visMsg->CmdRspMsg;
                msgCopied = true;
            }
            return msgCopied;
        }

    };

}
#endif //VIDERE_DEV_IMUCOMMANDRESPONSEMESSAGE_H



