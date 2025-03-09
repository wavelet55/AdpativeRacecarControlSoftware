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

#ifndef SYSINFO_STATIC_MESSAGE
#define SYSINFO_STATIC_MESSAGE

#include <string>
#include <RabitMessage.h>

namespace videre
{
    class SysInfoStaticMessage : public Rabit::RabitMessage
    {

    public:
        bool message_filled = false;
        std::string msg_str;

    public:
        SysInfoStaticMessage() : RabitMessage()
        {
            Clear();
        }

        virtual std::unique_ptr <Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<SysInfoStaticMessage>(new SysInfoStaticMessage());
            clone->CopyBase(this);
            clone->message_filled = this->message_filled;
            clone->msg_str = this->msg_str;
            return std::move(clone);
        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            Rabit::RabitMessage::CopyMessage(msg); // call baseclass
            if (msg->GetTypeIndex() == std::type_index(typeid(SysInfoStaticMessage)))
            {
                this->message_filled = static_cast<SysInfoStaticMessage *>(msg)->message_filled;
                this->msg_str = static_cast<SysInfoStaticMessage *>(msg)->msg_str;
                return true;
            }
            return false;
        }

        virtual void Clear() final
        {
            message_filled = false;
            msg_str = "{}";
        }

        virtual std::string ToString() const final
        {
            std::ostringstream os;
            os << "SysInfoStaticMessage: message_filled: " << message_filled
               << " , msg_str:" << msg_str;
            return os.str();
        }

    };
}

#endif //SYSINFO_STATIC_MESSAGE
