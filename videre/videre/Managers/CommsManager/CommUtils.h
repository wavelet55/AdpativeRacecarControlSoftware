/* ****************************************************************
 * Athr(s): Harry Direen PhD, Randy Direen Phd.
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: Aug. 2016
 *
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
 *******************************************************************/

#ifndef VIDERE_DEV_COMMUTILS_H
#define VIDERE_DEV_COMMUTILS_H

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <boost/algorithm/string.hpp>
#include <RabitManager.h>
#include <RabitMessageQueue.h>
#include <ManagerStatusMessage.h>
#include <ManagerControlMessage.h>
#include <ManagerStats.h>
#include <ManagerStatusMessage.h>

#include "all_manager_message.h"
#include "video_process_message.h"
#include "video_control_message.h"
#include "CompressedImageMessage.h"
#include "ImageProcTargetInfoResultsMessage.h"
#include "ImageCaptureStatusMessage.h"
#include "AccelerometerGyroMessage.h"
#include "HeadTrackingOrientationMessage.h"
#include "GPSFixMessage.h"
#include "../../ProtobufMessages/vision_messages.pb.h"
#include "../../Utils/config_data.h"
#include "../../Utils/logger.h"
#include "../../Utils/timing.h"

using namespace vision_messages;

namespace videre
{

    class CommUtils
    {

    };



    //Helper Functions

    void ImageCaptureStatusMsgToProtobufMsg(std::shared_ptr<ImageCaptureStatusMessage> mgrStatsMsg,
                                      vision_messages::ImageCaptureStatusPBMsg &imgCapStatsPBMsg);

    //Load a protobuf ManagerStatsMsg with the values from a ManagerStatusMessage
    void ManagerStatsMsgToProtobufMsg(std::shared_ptr<Rabit::ManagerStatusMessage> mgrStatsMsg,
                                      vision_messages::ManagerStatsPBMsg &mgrStatsTxPBMsg);

    bool CompressedImgageMsgToTxMsg(CompressedImageMessage *cImgMsgPtr,
                                    vision_messages::VisionMessageWrapperPBMsg &msgWrapperPBMsg);

    bool ManagerStatsMsgToTxMsg(std::shared_ptr<Rabit::ManagerStatusMessage> mgrStatsMsg,
                                vision_messages::VisionMessageWrapperPBMsg &msgWrapperPBMsg);

    bool TargetInfoMsgToTxMsg(ImageProcTargetInfoResultsMessage *tgtMsgPtr,
                              vision_messages::VisionMessageWrapperPBMsg &msgWrapperPBMsg);

    bool VehicleInertialStatesPBMsgToVISMsg(VehicleInertialStatePBMsg &visPBMsg,
                                            std::shared_ptr<VehicleInertialStatesMessage> visMsg_sptr);

    bool CameraSteeringPBMsgToCOMsg(CameraSteeringPBMsg &visPBMsg,
                                       std::shared_ptr<CameraOrientationMessage> visMsg_sptr);

    bool IMUAccelGyroMsgToTxMsg(AccelerometerGyroMessage * agMsgPtr,
                                vision_messages::VisionMessageWrapperPBMsg &msgWrapperPBMsg);

    bool TrackHeadOrientationMsgToTxMsg(TrackHeadOrientationMessage * htoMsgPtr,
                                        vision_messages::VisionMessageWrapperPBMsg &msgWrapperPBMsg);

    bool GPSFixeMsgToTxMsg(GPSFixMessage * gpsMsgPtr,
                           vision_messages::VisionMessageWrapperPBMsg &msgWrapperPBMsg);

}


#endif //VIDERE_DEV_COMMUTILS_H
