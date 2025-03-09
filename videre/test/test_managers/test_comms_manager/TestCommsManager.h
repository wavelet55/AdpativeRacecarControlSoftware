#ifndef TEST_COMMS_MANAGER
#define TEST_COMMS_MANAGER

#include <iostream>
#include <string>
#include <RabitManager.h>
#include "all_manager_message.h"
#include "video_process_message.h"
#include "video_control_message.h"
#include "telemetry_message.h"


class TestCommsManager : public Rabit::RabitManager
{

private:
    int count = 0;

    std::shared_ptr<VideoProcessMessage> _vid_process_msg;
    std::shared_ptr<VideoControlMessage> _vid_control_msg;
    std::shared_ptr<TelemetryMessage> _telemetry_msg;


public:
    TestCommsManager(std::string name) : RabitManager(name)
    {
        this->SetWakeupTimeDelayMSec(1000);

        _vid_process_msg = std::make_shared<VideoProcessMessage>("VidProcessMessage");
        this->AddPublishSubscribeMessage(_vid_process_msg->GetMessageTypeName(), _vid_process_msg);

        _vid_control_msg = std::make_shared<VideoControlMessage>("VidControlMessage");
        this->AddPublishSubscribeMessage(_vid_control_msg->GetMessageTypeName(), _vid_control_msg);

        _telemetry_msg = std::make_shared<TelemetryMessage>("TelemetryMessage");
        this->AddPublishSubscribeMessage(_telemetry_msg->GetMessageTypeName(), _telemetry_msg);
    }

    void ExecuteUnitOfWork() final
    {
        count++;
        if (count > 60)
        {
            std::cout << "*** Minute is up, shutting down ***" << std::endl;
            this->ShutdownAllManagers(true);
        }

        if (_vid_process_msg->FetchMessage())
        {
            std::cout << _vid_process_msg->ToString() << std::endl;
        }

        if (_vid_control_msg->FetchMessage())
        {
            std::cout << _vid_control_msg->ToString() << std::endl;
        }

        if (_telemetry_msg->FetchMessage())
        {
            std::cout << _telemetry_msg->ToString() << std::endl;
        }
    }

};

#endif //TEST_COMMS_MANAGER
