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

#ifndef VIDERE_DEV_TELEMETRYMESSAGEPROCESS_H
#define VIDERE_DEV_TELEMETRYMESSAGEPROCESS_H

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
#include <opencv2/core.hpp>
#include "zeromq_wrapper.h"
#include "all_manager_message.h"
#include "video_process_message.h"
#include "video_control_message.h"
#include "CommUtils.h"
#include "../../ProtobufMessages/vision_messages.pb.h"
#include "../../Utils/config_data.h"
#include "../../Utils/logger.h"
#include "../../Utils/timing.h"
#include "VehicleInertialStatesMessage.h"
#include "CameraOrientationMessage.h"

using namespace vision_messages;

namespace videre
{
    //The Vision System Receives Telemetry messages form HOPS which
    //contaion the vehicle inertial states, camera orientation an
    //other information as required.  This information is received over
    //a zeromq publish subscribe socket where HOPS publishes the information
    //and the vision system subscribes to the information.  HOPS does not have
    //to wait for the vision system to process or respond to the messages.
    //
    //All messages are wrapped in a VisionMessageWrapperPBMsg which establishes
    //the message type and has a payload of the actual message.
    //Each message is passed to a message handler.
    //There is no responce supplied back to HOPS for the messages received.
    class TelemetryMessageProcessor
    {
    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        ZeroMQWrapper* _zmqCommPtr;

        //A reference to the CommManger... primarily used during setup of this
        //class object.
        Rabit::RabitManager* _mgrPtr;

        typedef boost::function<void(VisionMessageWrapperPBMsg *telMsg)> messageHandler_t;
        //A dictionary of Message Handlers.
        std::unordered_map<std::string, messageHandler_t> _message_map;

        //The system can only process one message at a time... so
        //create message wrapper here.
        VisionMessageWrapperPBMsg _messageWrapper;
        VehicleInertialStatePBMsg _vehicleInertialStatesPBMsg;
        CameraSteeringPBMsg _cameraSteeringPBMsg;

        //Vision system publish subscribe messages... This handler
        //posts these messages to the rest of the vision system.

        std::shared_ptr<VehicleInertialStatesMessage> _vehicleInertialStatesMsg_sptr;
        std::shared_ptr<CameraOrientationMessage> _cameraOrientationMsg_sptr;

    public:

        TelemetryMessageProcessor();

        bool Intialize(Rabit::RabitManager* mgrPtr, ZeroMQWrapper* zmqComm);

        void Shutdown();

        //Call this method to check for and handle all messages from the remote system.
        //returns the number of messages processed.
        int HandleMessagesFromRemote();

    private:
        bool DispatchToFunctionHandler(VisionMessageWrapperPBMsg *msgWpr);

        void VehicleInertialStatesHandler(VisionMessageWrapperPBMsg *msg);

        void CameraOrientationHandler(VisionMessageWrapperPBMsg *msg);


    };

}

#endif //VIDERE_DEV_TELEMETRYMESSAGEPROCESS_H
