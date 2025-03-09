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


#ifndef VIDERE_DEV_PLAYBACKCONTROLMESSAGE_H
#define VIDERE_DEV_PLAYBACKCONTROLMESSAGE_H

#include <string>
#include <RabitMessage.h>

namespace videre
{

    class PlaybackControlMessage : public Rabit::RabitMessage
    {

    public:
        //The data log directory which contains the log files to be
        //played back.  If null or empy the default directory will be used.
        std::string DataLogDirectory = "";

        //Set this value to true to enable playback.
        //This variable does not start or stop the playback... it
        //informs the various managers the system is in playback mode
        //which typically stops the processing from sensor inputs.
        bool EnablePlayback = false;

        //If true, the playback will occur in a continuous loop...
        //once the logs have been read, they process will
        //automatically start over.
        bool LoopBackToStartOfDataRecords = false;

        //Start the playback process when true, stop when false;
        bool StartPlayback = false;

        //When true, the managers will sync playback to real-time
        //as the log records are read.
        bool TimeSyncPlayback = false;

        //When true, the playback will be restarted from the begining
        //of the log files.
        bool ResetPlayback = false;

        //Play the data records forward in time for the given number
        //of seconds and then automatically stop the playback.
        //If PlayTimeSeconds <= 0, then the play will continue indefinitely.
        double PlayForTimeSeconds = 0;

    public:
        PlaybackControlMessage() : RabitMessage()
        {
            Clear();
        }

        PlaybackControlMessage(const PlaybackControlMessage& msg)
        {
            *this = msg;
        }

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<PlaybackControlMessage>(new PlaybackControlMessage(*this));
            return std::move(clone);
        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            Rabit::RabitMessage::CopyMessage(msg); // call baseclass
            if (msg->GetTypeIndex() == std::type_index(typeid(PlaybackControlMessage)))
            {
                //Ensure the Copy process does not loose the mesages'
                //publish subscribe reference.
                std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

                PlaybackControlMessage *coMsg = static_cast<PlaybackControlMessage *>(msg);
                *this = *coMsg;

                this->SetGlobalPublishSubscribeMessageRef(psmr);
                return true;
            }
            return false;
        }

        virtual void Clear() final
        {
            DataLogDirectory = "";
            EnablePlayback = false;
            LoopBackToStartOfDataRecords = false;
            StartPlayback = false;
            TimeSyncPlayback = false;
            PlayForTimeSeconds = 0.0;
        }

    };
}


#endif //VIDERE_DEV_PLAYBACKCONTROLMESSAGE_H
