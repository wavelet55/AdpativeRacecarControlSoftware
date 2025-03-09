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

#ifndef VIDERE_DEV_COMMAND_RESPONSE_MESSAGE_HANDLER_H
#define VIDERE_DEV_COMMAND_RESPONSE_MESSAGE_HANDLER_H

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
#include "ImageCaptureControlMessage.h"
#include "ImageCaptureStatusMessage.h"
#include "ImageProcessControlMessage.h"
#include "StreamRecordImagesControlMessage.h"
#include "TargetParametersMessage.h"
#include "CameraCalCommandMessage.h"
#include "CameraCalStatusMessage.h"
#include "CameraParametersSetupMessage.h"
#include "ImageLoggingControlMessage.h"
#include "FeatureMatchProcCtrlMessage.h"
#include "FeatureMatchProcStatusMessage.h"
#include "LinearActuatorPositionCtrlMessage.h"
#include "VidereSystemControlMessage.h"
#include "SteeringTorqueCtrlMessage.h"
#include "HeadOrientationControlMessage.h"
#include "VehicleControlParametersMessage.h"
#include "CommUtils.h"
#include "../../ProtobufMessages/vision_messages.pb.h"
#include "../../Utils/config_data.h"
#include "../../Utils/logger.h"
#include "../../Utils/timing.h"
#include "IMUCommandResponseMessage.h"
#include "KarTechLinearActuatorParamsMessage.h"
#include "KarTechLinearActuatorSetupMessage.h"
#include "PlaybackControlMessage.h"
#include "HeadTrackingControlMessage.h"


using namespace vision_messages;

namespace videre
{
    //The Vision system can receive Command messages from HOPS
    //(the UAV system, or other system).  One command is received
    //at a time and the command must respond with a response message.
    //This is part of a ZeroMQ Command / Reply protocol.
    //Command and Response messages use the Protobuf Message Format.
    //
    //Command messages have the format:
    //  Cmd string:  This is the command name in string format which is case sensitive
    //  Cmd Qualifier:  This is an optional command parmeter or qualifier in string format
    //  Cmd Data:  This is and optional binary data array which can contain most anthing.
    //              The data array could contain a serialize protobuf message or JSON
    //              formated message... this is up to the designer of particular command.
    //
    // A response message is returned with the format:
    //   CmdResponseType  (OK, Error...)  an enum... this is required
    //   CmdResponseMessage,  and optional string message
    //   CmdResponseData:  an optional binary data array which can contain most anything.
    //
    //A dictionary of command response handlers is maintained by the class.
    //Each valid command will have a handler in the dictionary.
    class CommandResponseMessageHandler
    {
    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        ZeroMQWrapper* _zmqCommPtr;

        //A reference to the CommManger... primarily used during setup of this
        //class object.
        Rabit::RabitManager* _mgrPtr;

        typedef boost::function<void(VisionCommandPBMsg *cmdMsg, VisionResponsePBMsg *respMsg)> commandHandler_t;
        //A dictionary of Command Handlers.
        std::unordered_map<std::string, commandHandler_t> _command_map;

        //The Command Response messages.  Since the system can only handle
        //one Command/Response message at a time... it is ok to allocate these
        //messages as part of the object.
        VisionCommandPBMsg _visionCmdMsg;
        VisionResponsePBMsg _visionCmdResponseMsg;

        //Vision System Messages that this handler works with
        //Messages
        std::shared_ptr<VidereSystemControlMessage> _videreSystemCtrlMsg;

        std::shared_ptr<ImageCaptureControlMessage> _imageCaptureControlMsg;
        std::shared_ptr<ImageCaptureControlMessage> _imageCaptureControlStatusMsg;

        std::shared_ptr<PlaybackControlMessage> _playbackControlMsg;

        std::shared_ptr<ImageLoggingControlMessage> _imageLoggingControlMsg;

        std::shared_ptr<ImageCaptureStatusMessage> _imageCaptureStatusMsg;

        std::shared_ptr<CameraParametersSetupMessage> _cameraParametersSetupMsg;

        std::shared_ptr<ImageProcessControlMessage> _imageProcessControlMsg;
        //Parot back the status of Image Processing
        std::shared_ptr<ImageProcessControlMessage> _imageProcessControlStatusMsg;

        std::shared_ptr<StreamRecordImageControlMesssage> _streamRecordImageControlMsg;

        std::shared_ptr<VehicleInertialStatesMessage> _vehicleInertialStatesMsg;

        std::shared_ptr<CameraOrientationMessage> _cameraOrientationMsg;

        std::shared_ptr<CameraCalCommandMessage> _cameraCalControlMsg;

        std::shared_ptr<FeatureMatchProcCtrlMessage> _featureMatchProcCtrlMsg;

        std::shared_ptr<IMUCommandResponseMessage> _imuCommandMessage;

        std::shared_ptr<HeadTrackingControlMessage> _headTrackingControlMsg;

        std::shared_ptr<HeadOrientationControlMessage> _headOrientationControlMsg;

        std::shared_ptr<LinearActuatorPositionCtrlMessage> _brakeCtrlMsg;
        std::shared_ptr<LinearActuatorPositionCtrlMessage> _throttleCtrlMsg;
        std::shared_ptr<SteeringTorqueCtrlMessage> _steeringCtrlMsg;

        std::shared_ptr<KarTechLinearActuatorParamsMessage> _brakeActuatorParamsControlMsg;
        std::shared_ptr<KarTechLinearActuatorParamsMessage> _brakeActuatorParamsFeedbackMsg;
        std::shared_ptr<KarTechLinearActuatorParamsMessage> _throttleActuatorParamsControlMsg;
        std::shared_ptr<KarTechLinearActuatorParamsMessage> _throttleActuatorParamsFeedbackMsg;

        std::shared_ptr<KarTechLinearActuatorSetupMessage> _brakeActuatorSetuplMsg;
        std::shared_ptr<KarTechLinearActuatorSetupMessage> _throttleActuatorSetuplMsg;

        std::shared_ptr<VehicleControlParametersMessage> _vehicleControlParametersMsg;

        std::shared_ptr<TargetParametersMessage> _targetType1ParamsMsg;
        std::shared_ptr<TargetParametersMessage> _targetType2ParamsMsg;
        std::shared_ptr<TargetParametersMessage> _targetType3ParamsMsg;
        std::shared_ptr<TargetParametersMessage> _targetType4ParamsMsg;

        //Status Messages from other Managers... Note:  this manager's
        //status message is obtained from:  GetManagerStatusMessage()
        std::shared_ptr<Rabit::ManagerStatusMessage> _videreSystemControMgrStatsMsg;
        std::shared_ptr<Rabit::ManagerStatusMessage> _headOrientationMgrStatsMsg;
        std::shared_ptr<Rabit::ManagerStatusMessage> _vehicleStateMgrStatsMsg;
        std::shared_ptr<Rabit::ManagerStatusMessage> _commMgrStatsMsg;
        std::shared_ptr<Rabit::ManagerStatusMessage> _ImageCaptureMgrStatsMsg;
        std::shared_ptr<Rabit::ManagerStatusMessage> _VisionProcessMgrStatsMsg;
        std::shared_ptr<Rabit::ManagerStatusMessage> _StreamRecordMgrStatsMsg;
        std::shared_ptr<Rabit::ManagerStatusMessage> _IMUCommMgrStatsMsg;
        std::shared_ptr<Rabit::ManagerStatusMessage> _DTXIMUInterfaceMgrStatsMsg;
        std::shared_ptr<Rabit::ManagerStatusMessage> _GPSMgrStatsMsg;
        std::shared_ptr<Rabit::ManagerStatusMessage> _SipnPuffMgrStatsMsg;
        std::shared_ptr<Rabit::ManagerStatusMessage> _VehicleActuatorInterfaceMgrStatsMsg;
        std::shared_ptr<Rabit::ManagerStatusMessage> _RemoteControlMgrStatsMsg;
        std::shared_ptr<Rabit::ManagerStatusMessage> _RobotArmMgrStatsMsg;
        std::shared_ptr<Rabit::ManagerStatusMessage> _SystemInfoMgrStatsMsg;

        //A Protobuf message for transmitting a Manager Stats Message
        vision_messages::ManagerStatsPBMsg _mgrStatsTxPBMsg;

        std::shared_ptr<Rabit::ManagerControlMessage> _videreSystemControMgrCtrlMsg;
        std::shared_ptr<Rabit::ManagerControlMessage> _headOrientationMgrCtrlMsg;
        std::shared_ptr<Rabit::ManagerControlMessage> _vehicleStateMgrCtrlMsg;
        std::shared_ptr<Rabit::ManagerControlMessage> _commMgrCtrlMsg;
        std::shared_ptr<Rabit::ManagerControlMessage> _ImageCaptureMgrCtrlMsg;
        std::shared_ptr<Rabit::ManagerControlMessage> _VisionProcessMgrCtrlMsg;
        std::shared_ptr<Rabit::ManagerControlMessage> _StreamRecordMgrCtrlMsg;
        std::shared_ptr<Rabit::ManagerControlMessage> _IMUCommMgrCtrlMsg;
        std::shared_ptr<Rabit::ManagerControlMessage> _DTXIMUInterfaceMgrCtrlMsg;
        std::shared_ptr<Rabit::ManagerControlMessage> _GPSMgrCtrlMsg;
        std::shared_ptr<Rabit::ManagerControlMessage> _SipnPuffMgrCtrlMsg;
        std::shared_ptr<Rabit::ManagerControlMessage> _VehicleActuatorInterfaceMgrCtrlMsg;
        std::shared_ptr<Rabit::ManagerControlMessage> _RemoteControlMgrCtrlMsg;
        std::shared_ptr<Rabit::ManagerControlMessage> _RobotArmMgrCtrlMsg;
        std::shared_ptr<Rabit::ManagerControlMessage> _SystemInfoMgrCtrlMsg;

        std::shared_ptr<Rabit::ManagerControlMessage> _AllMgrCtrlMsg;

        //A Protobuf Message for recieving a Manager Control Message;
        vision_messages::ManagerControlPBMsg _mgrControlRxPBMsg;

    public:
        CommandResponseMessageHandler();

        bool Intialize(Rabit::RabitManager* mgrPtr, ZeroMQWrapper* zmqComm);

        void Shutdown();

        //Call this method to check for and handle a single message from the remote system.
        //Returns true if a message was handled... otherwise returns false.
        bool HandleMessageFromRemote();

    private:

        bool DispatchToFunctionHandler(VisionCommandPBMsg *cmdMsg,
                                       VisionResponsePBMsg *respMsg);

        //Command / Response Handlers
        //Each Handler is resposible for filling in the _visionCmdResponseMsg
        //Since only one Vision Command can be handled at a time this is safe.
        void KillHandler(VisionCommandPBMsg *cmdMsg,
                         VisionResponsePBMsg *respMsg);

        void InfoHandler(VisionCommandPBMsg *cmdMsg,
                         VisionResponsePBMsg *respMsg);

        void StartVisionHandler(VisionCommandPBMsg *cmdMsg,
                                VisionResponsePBMsg *respMsg);

        void StopVisionHandler(VisionCommandPBMsg *cmdMsg,
                               VisionResponsePBMsg *respMsg);

        void StartStreamHandler(VisionCommandPBMsg *cmdMsg,
                                VisionResponsePBMsg *respMsg);

        void StopStreamHandler(VisionCommandPBMsg *cmdMsg,
                               VisionResponsePBMsg *respMsg);

        void TargetImageProcessingEnableHandler(VisionCommandPBMsg *cmdMsg,
                                                VisionResponsePBMsg *respMsg);

        void TargetImageProcessingDisableHandler(VisionCommandPBMsg *cmdMsg,
                                                 VisionResponsePBMsg *respMsg);

        void StartRecordingHandler(VisionCommandPBMsg *cmdMsg,
                                   VisionResponsePBMsg *respMsg);

        void StopRecordingHandler(VisionCommandPBMsg *cmdMsg,
                                  VisionResponsePBMsg *respMsg);

        void SystemHandler(VisionCommandPBMsg *cmdMsg,
                           VisionResponsePBMsg *respMsg);

        void ImageCaptureControlCmdHandler(VisionCommandPBMsg *cmdMsg,
                                              VisionResponsePBMsg *respMsg);

        void ImageCaptureStatusCmdHandler(VisionCommandPBMsg *cmdMsg,
                                           VisionResponsePBMsg *respMsg);

        void PlaybackControlCmdHandler(VisionCommandPBMsg *cmdMsg,
                                       VisionResponsePBMsg *respMsg);


        void VisionProcessingControlCmdHandler(VisionCommandPBMsg *cmdMsg,
                                               VisionResponsePBMsg *respMsg);

        void VisionProcessCmdStatusHandler(VisionCommandPBMsg *cmdMsg,
                                           VisionResponsePBMsg *respMsg);

        void GetManagerStatsHandler(VisionCommandPBMsg *cmdMsg,
                           VisionResponsePBMsg *respMsg);


        void MangerControlCmdHandler(VisionCommandPBMsg *cmdMsg,
                             VisionResponsePBMsg *respMsg);

        //Setup the GeoCoordinate System... This should be done before
        //starting a mission.  Do not change the setup during a mission.
        void SetupGeoCoordinateSystemCmdHandler(VisionCommandPBMsg *cmdMsg,
                                                VisionResponsePBMsg *respMsg);

        //Get the GeoCoordinate Setup that the Vision System is currently
        //setup for.
        void GetGeoCoordinateSetupHandler(VisionCommandPBMsg *cmdMsg,
                                           VisionResponsePBMsg *respMsg);

        //This is primarily for test purposes... a Lat/Lon value will
        //be converted to the X-Y coordinate... or an X-Y coordinate
        //will be converted to the Lat/Lon position.  Lat/Lon are in Degrees.
        void LatLonXYConversionHandler(VisionCommandPBMsg *cmdMsg,
                                       VisionResponsePBMsg *respMsg);

        //Process Incoming Target Parameters Messages.
        void TargetParametersMsgHandler(VisionCommandPBMsg *cmdMsg,
                                        VisionResponsePBMsg *respMsg);

        void CameraCalCommandCmdMsgHandler(VisionCommandPBMsg *cmdMsg,
                                           VisionResponsePBMsg *respMsg);

        void VehicleInertialStatesMsgHandler(VisionCommandPBMsg *cmdMsg,
                                           VisionResponsePBMsg *respMsg);

        void CameraOrientationMsgHandler(VisionCommandPBMsg *cmdMsg,
                                         VisionResponsePBMsg *respMsg);

        void CameraParametersSetupMsgHandler(VisionCommandPBMsg *cmdMsg,
                                             VisionResponsePBMsg *respMsg);

        void GetImageCaptureControlSettingsHandler(VisionCommandPBMsg *cmdMsg,
                                                   VisionResponsePBMsg *respMsg);

        void StreamRecordImageControlMsgHandler(VisionCommandPBMsg *cmdMsg,
                                                    VisionResponsePBMsg *respMsg);

        void GetStreamRecordImageControlMsgHandler(VisionCommandPBMsg *cmdMsg,
                                                    VisionResponsePBMsg *respMsg);

        void ImageLoggingControlMsgHandler(VisionCommandPBMsg *cmdMsg,
                                           VisionResponsePBMsg *respMsg);

        void GetImageLoggingControlMsgHandler(VisionCommandPBMsg *cmdMsg,
                                              VisionResponsePBMsg *respMsg);

        void FeatureMatchProcCtrlMsgHandler(VisionCommandPBMsg *cmdMsg,
                                            VisionResponsePBMsg *respMsg);

        void GetListOfManagerNamesMsgHandler(VisionCommandPBMsg *cmdMsg,
                                            VisionResponsePBMsg *respMsg);

        void HeadTrackingControlMsgHandler(VisionCommandPBMsg *cmdMsg,
                                           VisionResponsePBMsg *respMsg);

        void BrakeControlMsgHandler(VisionCommandPBMsg *cmdMsg,
                                    VisionResponsePBMsg *respMsg);

        void BrakeSetupMsgHandler(VisionCommandPBMsg *cmdMsg,
                                    VisionResponsePBMsg *respMsg);

        void SetBrakeConfigParamsMsgHandler(VisionCommandPBMsg *cmdMsg,
                                            VisionResponsePBMsg *respMsg);

        void GetBrakeConfigParamsMsgHandler(VisionCommandPBMsg *cmdMsg,
                                            VisionResponsePBMsg *respMsg);

        void ThrottleControlMsgHandler(VisionCommandPBMsg *cmdMsg,
                                    VisionResponsePBMsg *respMsg);

        void ThrottleSetupMsgHandler(VisionCommandPBMsg *cmdMsg,
                                  VisionResponsePBMsg *respMsg);

        void SetThrottleConfigParamsMsgHandler(VisionCommandPBMsg *cmdMsg,
                                            VisionResponsePBMsg *respMsg);

        void GetThrottleConfigParamsMsgHandler(VisionCommandPBMsg *cmdMsg,
                                            VisionResponsePBMsg *respMsg);


        void SteeringControlMsgHandler(VisionCommandPBMsg *cmdMsg,
                                    VisionResponsePBMsg *respMsg);

        void IMUCommandMessageHandler(VisionCommandPBMsg *cmdMsg,
                                      VisionResponsePBMsg *respMsg);

        void VidereSystemControlCmdHandler(VisionCommandPBMsg *cmdMsg,
                                           VisionResponsePBMsg *respMsg);

        void HeadOrientationCommandHandler(VisionCommandPBMsg *cmdMsg,
                                             VisionResponsePBMsg *respMsg);

        void VehicleControlParametersHandler(VisionCommandPBMsg *cmdMsg,
                                             VisionResponsePBMsg *respMsg);

        void GetVehicleControlParametersHandler(VisionCommandPBMsg *cmdMsg,
                                                 VisionResponsePBMsg *respMsg);

    };

}
#endif //VIDERE_DEV_COMMAND_RESPONSE_MESSAGE_HANDLER_H
