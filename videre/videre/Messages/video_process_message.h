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

#ifndef VIDEO_PROCESS_MESSAGE
#define VIDEO_PROCESS_MESSAGE

#include <string>
#include <RabitMessage.h>

namespace videre
{
    class VideoProcessMessage : public Rabit::RabitMessage
    {

    public:
        bool do_process = false;
        bool use_gpu = false;

    public:
        VideoProcessMessage() : RabitMessage()
        {
            Clear();
        }

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<VideoProcessMessage>(new VideoProcessMessage(*this));
            return std::move(clone);
        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            Rabit::RabitMessage::CopyMessage(msg); // call baseclass
            if (msg->GetTypeIndex() == std::type_index(typeid(VideoProcessMessage)))
            {
                //Ensure the Copy process does not loose the mesages'
                //publish subscribe reference.
                std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

                VideoProcessMessage *inpMsg = static_cast<VideoProcessMessage *>(msg);
                *this = *inpMsg;

                this->SetGlobalPublishSubscribeMessageRef(psmr);
                return true;
            }
            return false;
        }

        virtual void Clear() final
        {
            do_process = false;
            use_gpu = false;
        }

        virtual std::string ToString() const final
        {
            std::ostringstream os;
            os << "VideoProcessMessage: do_process: " << do_process
               << ", use_gpu: " << use_gpu;
            return os.str();
        }

    };
}

#endif //VIDEO_PROCESS_MESSAGE
