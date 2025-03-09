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

#ifndef VIDEO_CONTROL_MESSAGE
#define VIDEO_CONTROL_MESSAGE

#include <string>
#include <RabitMessage.h>

namespace videre
{

    class VideoControlMessage : public Rabit::RabitMessage
    {

    public:
        bool stream = false;
        bool record = false;

    public:
        VideoControlMessage() : RabitMessage()
        {
            Clear();
        }

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<VideoControlMessage>(new VideoControlMessage());
            clone->CopyBase(this);
            clone->stream = this->stream;
            clone->record = this->record;
            return std::move(clone);
        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            Rabit::RabitMessage::CopyMessage(msg); // call baseclass
            if (msg->GetTypeIndex() == std::type_index(typeid(VideoControlMessage)))
            {
                this->stream = static_cast<VideoControlMessage *>(msg)->stream;
                this->record = static_cast<VideoControlMessage *>(msg)->record;
                return true;
            }
            return false;
        }

        virtual void Clear() final
        {
            stream = false;
            record = false;
        }

        virtual std::string ToString() const final
        {
            std::ostringstream os;
            os << "VideoControlMessage: stream: " << stream
               << ", record: " << record;
            return os.str();
        }

    };
}

#endif //VIDEO_PROCESS_MESSAGE
