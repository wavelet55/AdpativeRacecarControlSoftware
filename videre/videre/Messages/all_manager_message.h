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

#ifndef All_MANAGER_MESSAGE
#define All_MANAGER_MESSAGE

#include <string>
#include <iostream>
#include <sstream>
#include <memory>
#include <RabitMessage.h>

namespace videre
{
    class AllManagerMessage : public Rabit::RabitMessage
    {

    public:
        bool kill = false;

    public:
        AllManagerMessage() : RabitMessage()
        {
            Clear();
        }

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<AllManagerMessage>(new AllManagerMessage());
            clone->CopyBase(this);
            clone->kill = this->kill;
            return std::move(clone);
        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            Rabit::RabitMessage::CopyMessage(msg); // call baseclass
            if (msg->GetTypeIndex() == std::type_index(typeid(AllManagerMessage)))
            {
                this->kill = static_cast<AllManagerMessage *>(msg)->kill;
                return true;
            }
            return false;
        }

        virtual void Clear() final
        {
            kill = false;
        }

        virtual std::string ToString() const final
        {
            std::ostringstream os;
            os << "AllManagerMessage: kill: " << kill;
            return os.str();
        }

    };
}

#endif //All_MANAGER_MESSAGE
