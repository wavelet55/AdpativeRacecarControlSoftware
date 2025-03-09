/* ****************************************************************
 * Message
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Randy Direen PhD
 * Date: May 2016
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
  *******************************************************************/

#ifndef SYSINFO_DYNAMIC_MESSAGE
#define SYSINFO_DYNAMIC_MESSAGE

#include <string>
#include <RabitMessage.h>

namespace videre
{
    class SysInfoDynamicMessage : public Rabit::RabitMessage
    {

    public:
        std::string msg_str;

    public:
        SysInfoDynamicMessage() : RabitMessage()
        {
            Clear();
        }

        virtual std::unique_ptr <Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<SysInfoDynamicMessage>(new SysInfoDynamicMessage());
            clone->CopyBase(this);
            clone->msg_str = this->msg_str;
            return std::move(clone);
        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            Rabit::RabitMessage::CopyMessage(msg); // call baseclass
            if (msg->GetTypeIndex() == std::type_index(typeid(SysInfoDynamicMessage)))
            {
                this->msg_str = static_cast<SysInfoDynamicMessage *>(msg)->msg_str;
                return true;
            }
            return false;
        }

        virtual void Clear() final
        {
            msg_str = "{}";
        }

        virtual std::string ToString() const final
        {
            std::ostringstream os;
            os << "SysInfoDynamicMessage: " << msg_str;
            return os.str();
        }

        void PostDynamicInfo(std::string info)
        {
            msg_str = info;
            this->PostMessage();
        }

    };
}

#endif //SYSINFO_DYNAMIC_MESSAGE
