
#include "TestImageManager.h"
#include <iostream>

TestImageManager::TestImageManager(std::string name)
        : RabitManager(name)
{
    this->SetWakeupTimeDelayMSec(500);

    //Messages
    _vid_control_msg = std::make_shared<VideoControlMessage>("VidControlMessage");
    this->AddPublishSubscribeMessage(_vid_control_msg->GetMessageTypeName(), _vid_control_msg);

    _vid_process_msg = std::make_shared<VideoProcessMessage>("VidProcessMessage");
    this->AddPublishSubscribeMessage(_vid_process_msg->GetMessageTypeName(), _vid_process_msg);

    //Queues
    _frame_queue_sptr = std::make_shared<RabitFrames>(NFrames, "ImageFrames");
    this->AddManagerMessageQueue(_frame_queue_sptr->GetMessageQueueName(), _frame_queue_sptr);

    _copy_queue_sptr = std::make_shared<RabitFrames>(NFrames, "StreamMgrRecordImageQueue");
    this->AddManagerMessageQueue(_copy_queue_sptr->GetMessageQueueName(), _copy_queue_sptr);

    //Create a Image Message Pool to test and play with.
    ImagePlusMetadataMessage imgMsg("ImgMsg");
    _imageMsgPool = new MessagePool(&imgMsg, 5);

    for(int i = 0; i < 3; i++)
    {
        _imageMsgArray[i] = nullptr;
    }
}

TestImageManager::~TestImageManager()
{
    if(_imageMsgPool != nullptr)
        delete _imageMsgPool;
}

void TestImageManager::ExecuteUnitOfWork()
{

    _vid_control_msg->stream = true;
    _vid_control_msg->PostMessage();

    _vid_process_msg->do_process = true;
    _vid_process_msg->PostMessage();

    //Test Message Pool
    if(count % 4 < 2 )
    {
        if (_imageMsgPool->GetNumberOfFreeMessages() > 0)
        {
            ImagePlusMetadataMessage* imgMsg = static_cast<ImagePlusMetadataMessage *>(_imageMsgPool->CheckOutMessage());
            imgMsg->ImageNumber = 10 + count;
            _imageMsgArray[_imageMsgCOIdx] = imgMsg;
            _imageMsgCOIdx = ++_imageMsgCOIdx % 3;
        }
    }
    if(count % 4 >= 2 )
    {
        //Checkin the oldest imgMsg
        ImagePlusMetadataMessage* imgMsg = _imageMsgArray[_imageMsgCIIdx];
        int error = _imageMsgPool->CheckInMessage(imgMsg);
        _imageMsgCIIdx = ++_imageMsgCIIdx % 3;
        if( error != 0 )
        {
            std::cout << "Image Pool Checkin Error: " << error << std::endl;
        }
    }



    count++;
    if (count > 8)
    {
        std::cout << "*** Time is up, shutting down ***" << std::endl;
        this->ShutdownAllManagers(true);
    }

    if (_frame_queue_sptr->NoMessagesInQueue() > 0)
    {
        auto frame_sptr = _frame_queue_sptr->GetMessage();
        std::ostringstream os;
        os << "frame_" << count << ".jpg";
        cv::imwrite(os.str(), *frame_sptr);

    }
}
