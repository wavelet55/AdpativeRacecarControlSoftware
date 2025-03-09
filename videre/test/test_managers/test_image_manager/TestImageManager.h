#ifndef TEST_IMAGE_MANAGER
#define TEST_IMAGE_MANAGER

#include <iostream>
#include <string>


#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
// #include <opencv2/imagecodecs/imgcodecs.hpp> //OpenCV 3.0 only
//#include <opencv2/videoio/videoio.hpp> //OpenCV 3.0 only
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <RabitManager.h>
#include "all_manager_message.h"
#include "video_control_message.h"
#include "video_process_message.h"
#include "image_plus_metadata_message.h"
#include "message_pool.h"

using namespace videre;
using namespace Rabit;

class TestImageManager : public Rabit::RabitManager
{

private:
    int count = 0;

    const int NFrames = 2;

    MessagePool* _imageMsgPool;

    int _imageMsgCIIdx = 0;
    int _imageMsgCOIdx = 0;
    ImagePlusMetadataMessage*  _imageMsgArray[3];

    //Messages
    std::shared_ptr<VideoControlMessage> _vid_control_msg;
    std::shared_ptr<VideoProcessMessage> _vid_process_msg;

    //Queues
    typedef Rabit::RabitMessageQueue<std::shared_ptr<cv::Mat> > RabitFrames;
    std::shared_ptr<RabitFrames> _frame_queue_sptr;
    std::shared_ptr<RabitFrames> _copy_queue_sptr;


public:
    TestImageManager(std::string name);

    ~TestImageManager();

    void ExecuteUnitOfWork() final;

};

#endif //TEST_IMAGE_MANAGER
