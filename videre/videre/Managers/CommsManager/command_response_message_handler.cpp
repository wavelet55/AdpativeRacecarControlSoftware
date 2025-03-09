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


#include <global_defines.h>
#include "command_response_message_handler.h"
#include "GeoCoordinateSystem.h"
#include "LatLonAltStruct.h"
#include "XYZCoord_t.h"
#include "CommsManager.h"

using namespace GeoCoordinateSystemNS;
using namespace MathLibsNS;

namespace videre
{

    CommandResponseMessageHandler::CommandResponseMessageHandler()
    {
        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);
    }

    bool CommandResponseMessageHandler::Intialize(Rabit::RabitManager* mgrPtr, ZeroMQWrapper* zmqComm)
    {
        bool error = false;
        _mgrPtr = mgrPtr;
        _zmqCommPtr = zmqComm;

        //Messages
        _videreSystemCtrlMsg = std::make_shared<VidereSystemControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage("VidereSystemCtrlMsg", _videreSystemCtrlMsg);

        _imageCaptureControlMsg = std::make_shared<ImageCaptureControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage("ImageCaptureControlMessage", _imageCaptureControlMsg);

        _imageCaptureControlStatusMsg = std::make_shared<ImageCaptureControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage("ImageCaptureControlStatusMessage", _imageCaptureControlStatusMsg);

        _playbackControlMsg = std::make_shared<PlaybackControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage("PlaybackControlMessage", _playbackControlMsg);

        _imageLoggingControlMsg = std::make_shared<ImageLoggingControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage("ImageLoggingControlMessage", _imageLoggingControlMsg);

        _imageCaptureStatusMsg = std::make_shared<ImageCaptureStatusMessage>();
        _mgrPtr->AddPublishSubscribeMessage("ImageCaptureStatusMessage", _imageCaptureStatusMsg);

        _imageProcessControlMsg = std::make_shared<ImageProcessControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage("ImageProcessControlMessage", _imageProcessControlMsg);

        _imageProcessControlStatusMsg = std::make_shared<ImageProcessControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage("ImageProcessControlStatusMessage", _imageProcessControlStatusMsg);

        _streamRecordImageControlMsg = std::make_shared<StreamRecordImageControlMesssage>();
        _mgrPtr->AddPublishSubscribeMessage("StreamRecordImageControlMesssage", _streamRecordImageControlMsg);

        _vehicleInertialStatesMsg = std::make_shared<VehicleInertialStatesMessage>();
        mgrPtr->AddPublishSubscribeMessage("VehicleInertialStatesMessage", _vehicleInertialStatesMsg);

        _cameraOrientationMsg = std::make_shared<CameraOrientationMessage>();
        mgrPtr->AddPublishSubscribeMessage("CameraOrientationMessage", _cameraOrientationMsg);

        _cameraParametersSetupMsg = std::make_shared<CameraParametersSetupMessage>();
        mgrPtr->AddPublishSubscribeMessage("CameraParametersSetupMessage", _cameraParametersSetupMsg);

        _cameraCalControlMsg = std::make_shared<CameraCalCommandMessage>();
        _mgrPtr->AddPublishSubscribeMessage("CameraCalCommandMessage", _cameraCalControlMsg);

        _imuCommandMessage = std::make_shared<IMUCommandResponseMessage>();
        _mgrPtr->AddPublishSubscribeMessage("IMUCommandMessage", _imuCommandMessage);

        _targetType1ParamsMsg = std::make_shared<TargetParametersMessage>();
        _mgrPtr->AddPublishSubscribeMessage("TargetType1ParamsMsg", _targetType1ParamsMsg);

        _targetType2ParamsMsg = std::make_shared<TargetParametersMessage>();
        _mgrPtr->AddPublishSubscribeMessage("TargetType2ParamsMsg", _targetType2ParamsMsg);

        _targetType3ParamsMsg = std::make_shared<TargetParametersMessage>();
        _mgrPtr->AddPublishSubscribeMessage("TargetType3ParamsMsg", _targetType3ParamsMsg);

        _targetType4ParamsMsg = std::make_shared<TargetParametersMessage>();
        _mgrPtr->AddPublishSubscribeMessage("TargetType4ParamsMsg", _targetType4ParamsMsg);

        _featureMatchProcCtrlMsg = std::make_shared<FeatureMatchProcCtrlMessage>();
        _mgrPtr->AddPublishSubscribeMessage("FeatureMatchProcCtrlMessage", _featureMatchProcCtrlMsg);

        _headTrackingControlMsg = make_shared<HeadTrackingControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage("HeadTrackingControlMessage", _headTrackingControlMsg);

        _headOrientationControlMsg = make_shared<HeadOrientationControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage("HeadOrientationControlMessage", _headOrientationControlMsg);

        _brakeCtrlMsg = std::make_shared<LinearActuatorPositionCtrlMessage>();
        _mgrPtr->AddPublishSubscribeMessage("BrakeControlFromVidereMonitorMsg", _brakeCtrlMsg);

        _throttleCtrlMsg = std::make_shared<LinearActuatorPositionCtrlMessage>();
        _mgrPtr->AddPublishSubscribeMessage("ThrottleControlFromVidereMonitorMsg", _throttleCtrlMsg);

        _steeringCtrlMsg = std::make_shared<SteeringTorqueCtrlMessage>();
        _mgrPtr->AddPublishSubscribeMessage("SteeringTorqueCtrlFromVidereMonitorMsg", _steeringCtrlMsg);

        _brakeActuatorParamsControlMsg = std::make_shared<KarTechLinearActuatorParamsMessage>();
        _mgrPtr->AddPublishSubscribeMessage("BrakeLAParamsControlMsg", _brakeActuatorParamsControlMsg);

        _brakeActuatorParamsFeedbackMsg = std::make_shared<KarTechLinearActuatorParamsMessage>();
        _mgrPtr->AddPublishSubscribeMessage("BrakeLAParamsFeedbackMsg", _brakeActuatorParamsFeedbackMsg);

        _brakeActuatorSetuplMsg = std::make_shared<KarTechLinearActuatorSetupMessage>();
        _mgrPtr->AddPublishSubscribeMessage("BrakeLASetupMsg", _brakeActuatorSetuplMsg);

        _throttleActuatorParamsControlMsg = std::make_shared<KarTechLinearActuatorParamsMessage>();
        _mgrPtr->AddPublishSubscribeMessage("ThrottleLAParamsControlMsg", _throttleActuatorParamsControlMsg);

        _throttleActuatorParamsFeedbackMsg = std::make_shared<KarTechLinearActuatorParamsMessage>();
        _mgrPtr->AddPublishSubscribeMessage("ThrottleLAParamsFeedbackMsg", _throttleActuatorParamsFeedbackMsg);

        _throttleActuatorSetuplMsg = std::make_shared<KarTechLinearActuatorSetupMessage>();
        _mgrPtr->AddPublishSubscribeMessage("ThrottleLASetupMsg", _throttleActuatorSetuplMsg);

        std::string msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_VidereSystemControlManager, "ManagerStatusMessage");
        _videreSystemControMgrStatsMsg = std::make_shared<Rabit::ManagerStatusMessage>();
        _mgrPtr->AddPublishSubscribeMessage(msgName, _videreSystemControMgrStatsMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_HeadOrientationManager, "ManagerStatusMessage");
        _headOrientationMgrStatsMsg = std::make_shared<Rabit::ManagerStatusMessage>();
        _mgrPtr->AddPublishSubscribeMessage(msgName, _headOrientationMgrStatsMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_VehicleStateManager, "ManagerStatusMessage");
        _vehicleStateMgrStatsMsg = std::make_shared<Rabit::ManagerStatusMessage>();
        _mgrPtr->AddPublishSubscribeMessage(msgName, _vehicleStateMgrStatsMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_CommsManager, "ManagerStatusMessage");
        _commMgrStatsMsg = std::make_shared<Rabit::ManagerStatusMessage>();
        _mgrPtr->AddPublishSubscribeMessage(msgName, _commMgrStatsMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_ImageCaptureManager, "ManagerStatusMessage");
        _ImageCaptureMgrStatsMsg = std::make_shared<Rabit::ManagerStatusMessage>();
        _mgrPtr->AddPublishSubscribeMessage(msgName, _ImageCaptureMgrStatsMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_VisionProcessManager, "ManagerStatusMessage");
        _VisionProcessMgrStatsMsg = std::make_shared<Rabit::ManagerStatusMessage>();
        _mgrPtr->AddPublishSubscribeMessage(msgName, _VisionProcessMgrStatsMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_StreamRecordManager, "ManagerStatusMessage");
        _StreamRecordMgrStatsMsg = std::make_shared<Rabit::ManagerStatusMessage>();
        _mgrPtr->AddPublishSubscribeMessage(msgName, _StreamRecordMgrStatsMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_IMUCommManager, "ManagerStatusMessage");
        _IMUCommMgrStatsMsg = std::make_shared<Rabit::ManagerStatusMessage>();
        _mgrPtr->AddPublishSubscribeMessage(msgName, _IMUCommMgrStatsMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_DTX_IMU_InterfaceManager, "ManagerStatusMessage");
        _DTXIMUInterfaceMgrStatsMsg = std::make_shared<Rabit::ManagerStatusMessage>();
        _mgrPtr->AddPublishSubscribeMessage(msgName, _DTXIMUInterfaceMgrStatsMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_GPSManager, "ManagerStatusMessage");
        _GPSMgrStatsMsg = std::make_shared<Rabit::ManagerStatusMessage>();
        _mgrPtr->AddPublishSubscribeMessage(msgName, _GPSMgrStatsMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_SipnPuffManager, "ManagerStatusMessage");
        _SipnPuffMgrStatsMsg = std::make_shared<Rabit::ManagerStatusMessage>();
        _mgrPtr->AddPublishSubscribeMessage(msgName, _SipnPuffMgrStatsMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_VehicleActuatorInterfaceManager, "ManagerStatusMessage");
        _VehicleActuatorInterfaceMgrStatsMsg = std::make_shared<Rabit::ManagerStatusMessage>();
        _mgrPtr->AddPublishSubscribeMessage(msgName, _VehicleActuatorInterfaceMgrStatsMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_RemoteControlManager, "ManagerStatusMessage");
        _RemoteControlMgrStatsMsg = std::make_shared<Rabit::ManagerStatusMessage>();
        _mgrPtr->AddPublishSubscribeMessage(msgName, _RemoteControlMgrStatsMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_RobotArmManager, "ManagerStatusMessage");
        _RobotArmMgrStatsMsg = std::make_shared<Rabit::ManagerStatusMessage>();
        _mgrPtr->AddPublishSubscribeMessage(msgName, _RobotArmMgrStatsMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_SystemInfoManager, "ManagerStatusMessage");
        _SystemInfoMgrStatsMsg = std::make_shared<Rabit::ManagerStatusMessage>();
        _mgrPtr->AddPublishSubscribeMessage(msgName, _SystemInfoMgrStatsMsg);


        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_VidereSystemControlManager, "ManagerControlMessage");
        _videreSystemControMgrCtrlMsg = std::make_shared<Rabit::ManagerControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage(msgName, _videreSystemControMgrCtrlMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_HeadOrientationManager, "ManagerControlMessage");
        _headOrientationMgrCtrlMsg = std::make_shared<Rabit::ManagerControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage(msgName, _headOrientationMgrCtrlMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_VehicleStateManager, "ManagerControlMessage");
        _vehicleStateMgrCtrlMsg = std::make_shared<Rabit::ManagerControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage(msgName, _vehicleStateMgrCtrlMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_CommsManager, "ManagerControlMessage");
        _commMgrCtrlMsg = std::make_shared<Rabit::ManagerControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage(msgName, _commMgrCtrlMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_ImageCaptureManager, "ManagerControlMessage");
        _ImageCaptureMgrCtrlMsg = std::make_shared<Rabit::ManagerControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage(msgName, _ImageCaptureMgrCtrlMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_VisionProcessManager, "ManagerControlMessage");
        _VisionProcessMgrCtrlMsg = std::make_shared<Rabit::ManagerControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage(msgName, _VisionProcessMgrCtrlMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_StreamRecordManager, "ManagerControlMessage");
        _StreamRecordMgrCtrlMsg = std::make_shared<Rabit::ManagerControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage(msgName, _StreamRecordMgrCtrlMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_SystemInfoManager, "ManagerControlMessage");
        _SystemInfoMgrCtrlMsg = std::make_shared<Rabit::ManagerControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage(msgName, _SystemInfoMgrCtrlMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_IMUCommManager, "ManagerControlMessage");
        _IMUCommMgrCtrlMsg = std::make_shared<Rabit::ManagerControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage(msgName, _IMUCommMgrCtrlMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_DTX_IMU_InterfaceManager, "ManagerControlMessage");
        _DTXIMUInterfaceMgrCtrlMsg = std::make_shared<Rabit::ManagerControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage(msgName, _DTXIMUInterfaceMgrCtrlMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_GPSManager, "ManagerControlMessage");
        _GPSMgrCtrlMsg = std::make_shared<Rabit::ManagerControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage(msgName, _GPSMgrCtrlMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_SipnPuffManager, "ManagerControlMessage");
        _SipnPuffMgrCtrlMsg = std::make_shared<Rabit::ManagerControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage(msgName, _SipnPuffMgrCtrlMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_VehicleActuatorInterfaceManager, "ManagerControlMessage");
        _VehicleActuatorInterfaceMgrCtrlMsg = std::make_shared<Rabit::ManagerControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage(msgName, _VehicleActuatorInterfaceMgrCtrlMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_RemoteControlManager, "ManagerControlMessage");
        _RemoteControlMgrCtrlMsg = std::make_shared<Rabit::ManagerControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage(msgName, _RemoteControlMgrCtrlMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_RobotArmManager, "ManagerControlMessage");
        _RobotArmMgrCtrlMsg = std::make_shared<Rabit::ManagerControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage(msgName, _RobotArmMgrCtrlMsg);

        _AllMgrCtrlMsg = std::make_shared<Rabit::ManagerControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage("ManagerControlMessage", _AllMgrCtrlMsg);

        _vehicleControlParametersMsg = make_shared<VehicleControlParametersMessage>();
        _mgrPtr->AddPublishSubscribeMessage("VehicleControlParametersMsg", _vehicleControlParametersMsg);


        //Create a Command Map
        _command_map = std::unordered_map<std::string, commandHandler_t>();

        //Add Message Handlers to the Dictionary.
        _command_map["Kill"] = boost::bind(&CommandResponseMessageHandler::KillHandler, this, _1, _2);
        _command_map["Info"] = boost::bind(&CommandResponseMessageHandler::InfoHandler, this, _1, _2);
        _command_map["StartVision"] = boost::bind(&CommandResponseMessageHandler::StartVisionHandler, this, _1, _2);
        _command_map["StopVision"] = boost::bind(&CommandResponseMessageHandler::StopVisionHandler, this, _1, _2);
        _command_map["StartStream"] = boost::bind(&CommandResponseMessageHandler::StartStreamHandler, this, _1, _2);
        _command_map["StopStream"] = boost::bind(&CommandResponseMessageHandler::StopStreamHandler, this, _1, _2);
        _command_map["TargetProcessingEnable"] = boost::bind(&CommandResponseMessageHandler::TargetImageProcessingEnableHandler, this, _1, _2);
        _command_map["TargetProcessingDisable"] = boost::bind(&CommandResponseMessageHandler::TargetImageProcessingDisableHandler, this, _1, _2);
        _command_map["StartRecording"] = boost::bind(&CommandResponseMessageHandler::StartRecordingHandler, this, _1, _2);
        _command_map["StopRecording"] = boost::bind(&CommandResponseMessageHandler::StopRecordingHandler, this, _1, _2);
        _command_map["System"] = boost::bind(&CommandResponseMessageHandler::SystemHandler, this, _1, _2);
        _command_map["ImageCaptureControlCmd"] = boost::bind(&CommandResponseMessageHandler::ImageCaptureControlCmdHandler, this, _1, _2);
        _command_map["VidereSystemControlMsg"] = boost::bind(&CommandResponseMessageHandler::VidereSystemControlCmdHandler, this, _1, _2);
        _command_map["PlaybackControlCmd"] = boost::bind(&CommandResponseMessageHandler::PlaybackControlCmdHandler, this, _1, _2);
        _command_map["GetImageCaptureStatus"] = boost::bind(&CommandResponseMessageHandler::ImageCaptureStatusCmdHandler, this, _1, _2);
        _command_map["VisionProcessingControlCmd"] = boost::bind(&CommandResponseMessageHandler::VisionProcessingControlCmdHandler, this, _1, _2);
        _command_map["VisionProcessCmdStatus"] = boost::bind(&CommandResponseMessageHandler::VisionProcessCmdStatusHandler, this, _1, _2);
        _command_map["GetManagerStats"] = boost::bind(&CommandResponseMessageHandler::GetManagerStatsHandler, this, _1, _2);
        _command_map["ManagerControlCmd"] = boost::bind(&CommandResponseMessageHandler::MangerControlCmdHandler, this, _1, _2);
        _command_map["SetupGeoCoordinateSystem"] = boost::bind(&CommandResponseMessageHandler::SetupGeoCoordinateSystemCmdHandler, this, _1, _2);
        _command_map["GetGeoCoordinateSetup"] = boost::bind(&CommandResponseMessageHandler::GetGeoCoordinateSetupHandler, this, _1, _2);
        _command_map["LatLonXYConversion"] = boost::bind(&CommandResponseMessageHandler::LatLonXYConversionHandler, this, _1, _2);
        _command_map["TargetParameters"] = boost::bind(&CommandResponseMessageHandler::TargetParametersMsgHandler, this, _1, _2);
        _command_map["CameraCalCommand"] = boost::bind(&CommandResponseMessageHandler::CameraCalCommandCmdMsgHandler, this, _1, _2);
        _command_map["FeatureMatchProcCtrlMsg"] = boost::bind(&CommandResponseMessageHandler::FeatureMatchProcCtrlMsgHandler, this, _1, _2);
        _command_map["VehicleInertialStatesMsg"] = boost::bind(&CommandResponseMessageHandler::VehicleInertialStatesMsgHandler, this, _1, _2);
        _command_map["CameraOrientationMsg"] = boost::bind(&CommandResponseMessageHandler::CameraOrientationMsgHandler, this, _1, _2);
        _command_map["CameraParametersSetupMsg"] = boost::bind(&CommandResponseMessageHandler::CameraParametersSetupMsgHandler, this, _1, _2);
        _command_map["GetImageCaptureControlSettings"] = boost::bind(&CommandResponseMessageHandler::GetImageCaptureControlSettingsHandler, this, _1, _2);

        _command_map["StreamRecordImageControlMsg"] = boost::bind(&CommandResponseMessageHandler::StreamRecordImageControlMsgHandler, this, _1, _2);
        _command_map["GetStreamRecordImageControlMsg"] = boost::bind(&CommandResponseMessageHandler::GetStreamRecordImageControlMsgHandler, this, _1, _2);

        _command_map["ImageLoggingControlMsg"] = boost::bind(&CommandResponseMessageHandler::ImageLoggingControlMsgHandler, this, _1, _2);
        _command_map["GetImageLoggingControlMsg"] = boost::bind(&CommandResponseMessageHandler::GetImageLoggingControlMsgHandler, this, _1, _2);

        _command_map["GetListOfManagerNames"] = boost::bind(&CommandResponseMessageHandler::GetListOfManagerNamesMsgHandler, this, _1, _2);

        _command_map["HeadTrackingControlMsg"] = boost::bind(&CommandResponseMessageHandler::HeadTrackingControlMsgHandler, this, _1, _2);
        _command_map["BrakePositionCtrlMsg"] = boost::bind(&CommandResponseMessageHandler::BrakeControlMsgHandler, this, _1, _2);
        _command_map["BrakeSetupMsg"] = boost::bind(&CommandResponseMessageHandler::BrakeSetupMsgHandler, this, _1, _2);
        _command_map["BrakeActuatorConfigMsg"] = boost::bind(&CommandResponseMessageHandler::SetBrakeConfigParamsMsgHandler, this, _1, _2);
        _command_map["GetBrakeActuatorConfigParams"] = boost::bind(&CommandResponseMessageHandler::GetBrakeConfigParamsMsgHandler, this, _1, _2);
        _command_map["ThrottlePositionCtrlMsg"] = boost::bind(&CommandResponseMessageHandler::ThrottleControlMsgHandler, this, _1, _2);
        _command_map["ThrottleSetupMsg"] = boost::bind(&CommandResponseMessageHandler::ThrottleSetupMsgHandler, this, _1, _2);
        _command_map["ThrottleActuatorConfigMsg"] = boost::bind(&CommandResponseMessageHandler::SetThrottleConfigParamsMsgHandler, this, _1, _2);
        _command_map["GetThrottleActuatorConfigParams"] = boost::bind(&CommandResponseMessageHandler::GetThrottleConfigParamsMsgHandler, this, _1, _2);
        _command_map["SteeringTorqueControlMsg"] = boost::bind(&CommandResponseMessageHandler::SteeringControlMsgHandler, this, _1, _2);
        _command_map["IMUCommandMsg"] = boost::bind(&CommandResponseMessageHandler::IMUCommandMessageHandler, this, _1, _2);
        _command_map["HeadOrientationControlMsg"] = boost::bind(&CommandResponseMessageHandler::HeadOrientationCommandHandler, this, _1, _2);
        _command_map["SetVehicleControlParametersMsg"] = boost::bind(&CommandResponseMessageHandler::VehicleControlParametersHandler, this, _1, _2);
        _command_map["GetVehicleControlParametersMsg"] = boost::bind(&CommandResponseMessageHandler::GetVehicleControlParametersHandler, this, _1, _2);


        return error;
    }

    void CommandResponseMessageHandler::Shutdown()
    {

    }

    //Call this method to check for and handle a single message from the remote system.
    //Returns true if a message was handled... otherwise returns false.
    bool CommandResponseMessageHandler::HandleMessageFromRemote()
    {
        zmq::message_t req;
        std::string serializedResponse_string = "";
        bool messageProcessed = false;

        _visionCmdMsg.Clear();
        if (_zmqCommPtr->ReceiveRequestFromHOPS(&req))
        {
            //LOGTRACE("Received message from HOPS Framework.")
            if(_visionCmdMsg.ParseFromArray(req.data(), req.size()) )
            {
                _visionCmdResponseMsg.Clear();
                DispatchToFunctionHandler(&_visionCmdMsg, &_visionCmdResponseMsg);
            }
            else
            {
                _visionCmdResponseMsg.set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
                _visionCmdResponseMsg.set_cmdresponsemessage("Error Deserializeing a command.");
                LOGERROR("CommandResponseMessageHandler:Error Deserializeing a command.");

            }

            _visionCmdResponseMsg.SerializeToString(&serializedResponse_string);
            _zmqCommPtr->SendResponse(serializedResponse_string);
            messageProcessed = true;
        }
        return messageProcessed;
    }


    bool CommandResponseMessageHandler::DispatchToFunctionHandler(VisionCommandPBMsg *cmdMsg,
                                                                  VisionResponsePBMsg *respMsg)
    {
        bool is_valid_command = false;

        try
        {
            std::string cmd = cmdMsg->command();
            auto key_location = _command_map.find(cmd);
            if (key_location != _command_map.end())
            {
                _command_map[cmd](cmdMsg, respMsg);
                is_valid_command = true;
            }
            else
            {
                respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
                respMsg->set_cmdresponsemessage("Invalid command: " + cmd);
                LOGWARN("CommandResponseMessageHandler:Invalid command: " + cmd);
                is_valid_command = false;
            }
        }
        catch (std::exception &e)
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
            respMsg->set_cmdresponsemessage(e.what());
            LOGERROR("CommandResponseMessageHandler:Exception in handler processing: " << e.what());
        }

        return is_valid_command;
    }


/************************************************************************************
*                        Handlers from HOPS messages
************************************************************************************/

    void CommandResponseMessageHandler::KillHandler(VisionCommandPBMsg *cmdMsg,
                                                    VisionResponsePBMsg *respMsg)
    {
        LOGINFO("CommsManager: Received Kill Command... shutting down Videre.")
        _mgrPtr->ShutdownAllManagers(true);
        respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
        respMsg->set_cmdresponsemessage("[HANDLER] KILL");
    }

    void CommandResponseMessageHandler::InfoHandler(VisionCommandPBMsg *cmdMsg,
                                                    VisionResponsePBMsg *respMsg)
    {
        respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
        respMsg->set_cmdresponsemessage("[HANDLER] I hit the Info Button");
    }

    //Same as Image Capture Enable
    void CommandResponseMessageHandler::StartVisionHandler(VisionCommandPBMsg *cmdMsg,
                                                           VisionResponsePBMsg *respMsg)
    {
        _imageCaptureControlMsg->ImageCaptureEnabled = true;
        _imageCaptureControlMsg->PostMessage();
        respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
        respMsg->set_cmdresponsemessage("[HANDLER] Starting Vision");
    }

    //Same as Image Capture Disable
    void CommandResponseMessageHandler::StopVisionHandler(VisionCommandPBMsg *cmdMsg,
                                                          VisionResponsePBMsg *respMsg)
    {
        _imageCaptureControlMsg->ImageCaptureEnabled = false;
        _imageCaptureControlMsg->PostMessage();
        respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
        respMsg->set_cmdresponsemessage("[HANDLER] Stopping Vision");
    }

    //Same as StreamImagesEnabled
    void CommandResponseMessageHandler::StartStreamHandler(VisionCommandPBMsg *cmdMsg,
                                                           VisionResponsePBMsg *respMsg)
    {
        _streamRecordImageControlMsg->StreamImagesEnabled = true;
        _streamRecordImageControlMsg->PostMessage();
        respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
        respMsg->set_cmdresponsemessage("[HANDLER] Starting Stream");
    }

    void CommandResponseMessageHandler::StopStreamHandler(VisionCommandPBMsg *cmdMsg,
                                                          VisionResponsePBMsg *respMsg)
    {
        _streamRecordImageControlMsg->StreamImagesEnabled = false;
        _streamRecordImageControlMsg->PostMessage();
        respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
        respMsg->set_cmdresponsemessage("[HANDLER] Stopping Stream");
    }

    void CommandResponseMessageHandler::TargetImageProcessingEnableHandler(VisionCommandPBMsg *cmdMsg,
                                                                           VisionResponsePBMsg *respMsg)
    {
        _imageProcessControlMsg->TargetImageProcessingEnabled = true;
        _imageProcessControlMsg->PostMessage();
        respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
        respMsg->set_cmdresponsemessage("[HANDLER] Enable GPU");
    }

    void CommandResponseMessageHandler::TargetImageProcessingDisableHandler(VisionCommandPBMsg *cmdMsg,
                                                                            VisionResponsePBMsg *respMsg)
    {
        _imageProcessControlMsg->TargetImageProcessingEnabled = false;
        _imageProcessControlMsg->PostMessage();
        respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
        respMsg->set_cmdresponsemessage("[HANDLER] Disable GPU");
    }

    void CommandResponseMessageHandler::StartRecordingHandler(VisionCommandPBMsg *cmdMsg,
                                                              VisionResponsePBMsg *respMsg)
    {
        _imageLoggingControlMsg->EnableLogging = true;
        _imageLoggingControlMsg->PostMessage();
        _streamRecordImageControlMsg->RecordImagesEnabled = true;
        _streamRecordImageControlMsg->PostMessage();
        respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
        respMsg->set_cmdresponsemessage("[HANDLER] Start Recording");
    }

    void CommandResponseMessageHandler::StopRecordingHandler(VisionCommandPBMsg *cmdMsg,
                                                             VisionResponsePBMsg *respMsg)
    {
        _imageLoggingControlMsg->EnableLogging = false;
        _imageLoggingControlMsg->PostMessage();
        _streamRecordImageControlMsg->RecordImagesEnabled = false;
        _streamRecordImageControlMsg->PostMessage();
        respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
        respMsg->set_cmdresponsemessage("[HANDLER] Stop Recording");
    }

    void CommandResponseMessageHandler::SystemHandler(VisionCommandPBMsg *cmdMsg,
                                                      VisionResponsePBMsg *respMsg)
    {
        respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
        respMsg->set_cmdresponsemessage("[HANDLER] NotSuported Yet");
    }


    void CommandResponseMessageHandler::ImageCaptureControlCmdHandler(VisionCommandPBMsg *cmdMsg,
                                                                          VisionResponsePBMsg *respMsg)
    {
        vision_messages::ImageCaptureControlPBMsg visionCtrlPBMsg;
        std::shared_ptr<Rabit::ManagerControlMessage> mgrCtrlMsg;

        bool ctrlMsgSet = false;
        if( cmdMsg->has_cmddata() )
        {
            try
            {
                visionCtrlPBMsg.ParseFromString(cmdMsg->cmddata());
                ctrlMsgSet = true;
                _imageCaptureControlMsg->ImageCaptureEnabled = visionCtrlPBMsg.imagecaptureenabled();
                _imageCaptureControlMsg->NumberOfImagesToCapture = visionCtrlPBMsg.numberofimagestocapture();
                _imageCaptureControlMsg->DesiredFramesPerSecond = visionCtrlPBMsg.desiredframespersecond();
                _imageCaptureControlMsg->DesiredImageWidth = visionCtrlPBMsg.desiredimagewidth();
                _imageCaptureControlMsg->DesiredImageHeight = visionCtrlPBMsg.desiredimageheight();
                _imageCaptureControlMsg->ImageCaptureSource = (ImageCaptureSource_e)visionCtrlPBMsg.imagecapturesource();
                _imageCaptureControlMsg->ImageCaptureFormat = (ImageCaptureFormat_e)visionCtrlPBMsg.imagecaptureformat();
                _imageCaptureControlMsg->ImageCaptureSourceConfigPri = visionCtrlPBMsg.imagecapturesourceconfigpri();
                _imageCaptureControlMsg->ImageCaptureSourceConfigSec = visionCtrlPBMsg.imagecapturesourceconfigsec();
                _imageCaptureControlMsg->ImageSourceLoopAround = visionCtrlPBMsg.imagesourcelooparound();
                _imageCaptureControlMsg->AutoFocusEnable = visionCtrlPBMsg.autofocusenable();
                _imageCaptureControlMsg->PostMessage();
            }
            catch (std::exception &e)
            {
                LOGERROR("VisionProcessingControlCmdHandler Exception: " << e.what());
            }
        }
        if(ctrlMsgSet )
        {
            //Handle the command changes:

        }
        else
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
            respMsg->set_cmdresponsemessage("[HANDLER] Could Not read Vision Processing Control Message");
        }
    }



    void CommandResponseMessageHandler::VidereSystemControlCmdHandler(VisionCommandPBMsg *cmdMsg,
                                                                  VisionResponsePBMsg *respMsg)
    {
        vision_messages::VidereSystemControlPBMsg cmdPBMsg;
        if( cmdMsg->has_cmddata() )
        {
            try
            {
                cmdPBMsg.ParseFromString(cmdMsg->cmddata());
                _videreSystemCtrlMsg->SystemState = (VidereSystemStates_e)cmdPBMsg.systemstate();
                _videreSystemCtrlMsg->StartProcess = cmdPBMsg.startprocess();
                _videreSystemCtrlMsg->PauseProces = cmdPBMsg.pauseproces();
                _videreSystemCtrlMsg->StopProcess = cmdPBMsg.stopprocess();
                _videreSystemCtrlMsg->HeadControlEnable = cmdPBMsg.headcontrolenable();
                _videreSystemCtrlMsg->ThrottleControlEnable = cmdPBMsg.throttlecontrolenable();
                _videreSystemCtrlMsg->BrakeControlEnable = cmdPBMsg.brakecontrolenable();
                _videreSystemCtrlMsg->BCIControlEnable = cmdPBMsg.nexusbcicontrolenabled();

                _videreSystemCtrlMsg->SystemStatus = (VidereSystemStatus_e)cmdPBMsg.systemstatus();
                _videreSystemCtrlMsg->StatusCounter = cmdPBMsg.statuscounter();
                _videreSystemCtrlMsg->PostMessage();

                CommsManager *cmPtr = (CommsManager*)_mgrPtr;
                cmPtr->WorkSpace()->BCIControlConfigMsg->FetchMessage();
                cmPtr->WorkSpace()->BCIControlConfigMsg->EnableNexusBCIThrottleControl = cmdPBMsg.nexusbcicontrolenabled();
                cmPtr->WorkSpace()->BCIControlConfigMsg->PostMessage();
            }
            catch (std::exception &e)
            {
                LOGERROR("VidereSystemControlCmdHandler Exception: " << e.what());
            }
        }
    }


    void CommandResponseMessageHandler::PlaybackControlCmdHandler(VisionCommandPBMsg *cmdMsg,
                                                                      VisionResponsePBMsg *respMsg)
    {
        vision_messages::PlaybackControlPBMsg cmdPBMsg;
        if( cmdMsg->has_cmddata() )
        {
            try
            {
                cmdPBMsg.ParseFromString(cmdMsg->cmddata());
                _playbackControlMsg->DataLogDirectory = cmdPBMsg.datalogdirectory();
                _playbackControlMsg->LoopBackToStartOfDataRecords = cmdPBMsg.loopbacktostartofdatarecords();
                _playbackControlMsg->EnablePlayback = cmdPBMsg.enableplayback();
                _playbackControlMsg->StartPlayback = cmdPBMsg.startplayback();
                _playbackControlMsg->ResetPlayback = cmdPBMsg.resetplayback();
                _playbackControlMsg->TimeSyncPlayback = cmdPBMsg.timesyncplayback();
                _playbackControlMsg->PlayForTimeSeconds = cmdPBMsg.playfortimeseconds();
                _playbackControlMsg->PostMessage();
            }
            catch (std::exception &e)
            {
                LOGERROR("VisionProcessingControlCmdHandler Exception: " << e.what());
            }
        }
    }



    void CommandResponseMessageHandler::GetImageCaptureControlSettingsHandler(VisionCommandPBMsg *cmdMsg,
                                                                              VisionResponsePBMsg *respMsg)
    {

        vision_messages::ImageCaptureControlPBMsg visionCtrlPBMsg;
        _imageCaptureControlStatusMsg->FetchMessage();

        visionCtrlPBMsg.set_imagecaptureenabled(_imageCaptureControlStatusMsg->ImageCaptureEnabled);
        visionCtrlPBMsg.set_numberofimagestocapture(_imageCaptureControlStatusMsg->NumberOfImagesToCapture);
        visionCtrlPBMsg.set_desiredframespersecond(_imageCaptureControlStatusMsg->DesiredFramesPerSecond);
        visionCtrlPBMsg.set_desiredimagewidth(_imageCaptureControlStatusMsg->DesiredImageWidth);
        visionCtrlPBMsg.set_desiredimageheight(_imageCaptureControlStatusMsg->DesiredImageHeight);
        visionCtrlPBMsg.set_imagecapturesource((::vision_messages::ImageCaptureSource_e)_imageCaptureControlStatusMsg->ImageCaptureSource);
        visionCtrlPBMsg.set_imagecaptureformat((::vision_messages::CPImageCaptureFormat_e)_imageCaptureControlStatusMsg->ImageCaptureFormat);
        visionCtrlPBMsg.set_imagecapturesourceconfigpri(_imageCaptureControlStatusMsg->ImageCaptureSourceConfigPri);
        visionCtrlPBMsg.set_imagecapturesourceconfigsec(_imageCaptureControlStatusMsg->ImageCaptureSourceConfigSec);
        visionCtrlPBMsg.set_imagesourcelooparound(_imageCaptureControlStatusMsg->ImageSourceLoopAround);
        visionCtrlPBMsg.set_autofocusenable(_imageCaptureControlStatusMsg->AutoFocusEnable);

        respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
        respMsg->set_cmdresponsemessage("[HANDLER] Retrieved Image Control Status");
        respMsg->set_cmdresponsedata(visionCtrlPBMsg.SerializeAsString());
    }


    void CommandResponseMessageHandler::StreamRecordImageControlMsgHandler(VisionCommandPBMsg *cmdMsg,
                                                                      VisionResponsePBMsg *respMsg)
    {
        vision_messages::StreamControlPBMsg srCtrlPBMsg;

        if( cmdMsg->has_cmddata() )
        {
            try
            {
                srCtrlPBMsg.ParseFromString(cmdMsg->cmddata());
                _streamRecordImageControlMsg->StreamImagesEnabled = srCtrlPBMsg.streamimagesenabled();
                _streamRecordImageControlMsg->SetStreamImageFrameRate( srCtrlPBMsg.streamimageframerate() );
                _streamRecordImageControlMsg->SetImageCompressionQuality(srCtrlPBMsg.imagecompressionquality());
                _streamRecordImageControlMsg->SetImageScaleDownFactor(srCtrlPBMsg.streamimagescaledownfactor());
                _streamRecordImageControlMsg->PostMessage();
                respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
                respMsg->set_cmdresponsemessage("[HANDLER] Stream Record ImageControl");
            }
            catch (std::exception &e)
            {
                LOGERROR("StreamRecordImageControlMsgHandler Exception: " << e.what());
                respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
                respMsg->set_cmdresponsemessage("[HANDLER] Stream Record ImageControl Exception Thrown");
            }
        }
    }

    void CommandResponseMessageHandler::GetStreamRecordImageControlMsgHandler(VisionCommandPBMsg *cmdMsg,
                                                                              VisionResponsePBMsg *respMsg)
    {
        vision_messages::StreamControlPBMsg srCtrlPBMsg;
        _streamRecordImageControlMsg->FetchMessage();

        srCtrlPBMsg.set_streamimagesenabled(_streamRecordImageControlMsg->StreamImagesEnabled);
        srCtrlPBMsg.set_streamimageframerate(_streamRecordImageControlMsg->StreamImageFrameRate);
        srCtrlPBMsg.set_imagecompressionquality(_streamRecordImageControlMsg->ImageCompressionQuality);

        respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
        respMsg->set_cmdresponsemessage("[HANDLER] Retrieved Stream Record Image Control");
        respMsg->set_cmdresponsedata(srCtrlPBMsg.SerializeAsString());
    }

    void CommandResponseMessageHandler::ImageLoggingControlMsgHandler(VisionCommandPBMsg *cmdMsg,
                                                                           VisionResponsePBMsg *respMsg)
    {
        vision_messages::ImageLoggingControlPBMsg srCtrlPBMsg;

        if( cmdMsg->has_cmddata() )
        {
            try
            {
                srCtrlPBMsg.ParseFromString(cmdMsg->cmddata());
                _imageLoggingControlMsg->VisionLoggingType = (VisionLoggingType_e)srCtrlPBMsg.loggingtype();
                _imageLoggingControlMsg->EnableLogging = srCtrlPBMsg.enablelogging();
                _imageLoggingControlMsg->PostMessage();
                respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
                respMsg->set_cmdresponsemessage("[HANDLER] ImageLoggingControl");
            }
            catch (std::exception &e)
            {
                LOGERROR("ImageLoggingControlMsgHandler Exception: " << e.what());
                respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
                respMsg->set_cmdresponsemessage("[HANDLER] ImageLoggingControlMsgHandler Exception Thrown");
            }
        }
    }

    void CommandResponseMessageHandler::GetImageLoggingControlMsgHandler(VisionCommandPBMsg *cmdMsg,
                                                                              VisionResponsePBMsg *respMsg)
    {
        vision_messages::ImageLoggingControlPBMsg srCtrlPBMsg;
        _imageLoggingControlMsg->FetchMessage();

        srCtrlPBMsg.set_loggingtype((::vision_messages::ImageLoggingControlPBMsg_VisionLoggingType_e)_imageLoggingControlMsg->VisionLoggingType);
        srCtrlPBMsg.set_enablelogging(_imageLoggingControlMsg->EnableLogging);

        respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
        respMsg->set_cmdresponsemessage("[HANDLER] Get Image Logging Control");
        respMsg->set_cmdresponsedata(srCtrlPBMsg.SerializeAsString());
    }


    void CommandResponseMessageHandler::VisionProcessingControlCmdHandler(VisionCommandPBMsg *cmdMsg,
                                                                VisionResponsePBMsg *respMsg)
    {
        vision_messages::VisionProcessingControlPBMsg visionCtrlPBMsg;
        std::shared_ptr<Rabit::ManagerControlMessage> mgrCtrlMsg;

        bool ctrlMsgSet = false;
        if( cmdMsg->has_cmddata() )
        {
            try
            {
                visionCtrlPBMsg.ParseFromString(cmdMsg->cmddata());
                ctrlMsgSet = true;
                bool cmdChanged = false;

                cmdChanged = false;
                if(visionCtrlPBMsg.gpuprocessingenabled() != _imageProcessControlMsg->GPUProcessingEnabled)
                {
                    //enable or disable GPU/Cuda  Accelerated Routines (as opposed to standard... non-accelerated
                    //routines)
                    _imageProcessControlMsg->GPUProcessingEnabled = visionCtrlPBMsg.gpuprocessingenabled();
                    cmdChanged = true;
                }
                if(visionCtrlPBMsg.targetimageprocessingenabled() != _imageProcessControlMsg->TargetImageProcessingEnabled)
                {
                    _imageProcessControlMsg->TargetImageProcessingEnabled = visionCtrlPBMsg.targetimageprocessingenabled();
                    cmdChanged = true;
                }
                if(visionCtrlPBMsg.gpsdeniedprocessingenabled() != _imageProcessControlMsg->GPSDeniedProcessingEnabled)
                {
                    _imageProcessControlMsg->GPSDeniedProcessingEnabled = visionCtrlPBMsg.gpsdeniedprocessingenabled();
                    cmdChanged = true;
                }
                if((int)visionCtrlPBMsg.visionprocessingmode() != (int)_imageProcessControlMsg->VisionProcessingMode)
                {
                    _imageProcessControlMsg->VisionProcessingMode = (VisionProcessingMode_e)visionCtrlPBMsg.visionprocessingmode();
                    cmdChanged = true;
                }
                if((int)visionCtrlPBMsg.targetprocessingmode() != (int)_imageProcessControlMsg->TargetProcessingMode)
                {
                    _imageProcessControlMsg->TargetProcessingMode = (TargetProcessingMode_e)visionCtrlPBMsg.targetprocessingmode();
                    cmdChanged = true;
                }
                if((int)visionCtrlPBMsg.gpsdeniedprocessingmode() != (int)_imageProcessControlMsg->GPSDeniedProcessingMode)
                {
                    _imageProcessControlMsg->GPSDeniedProcessingMode = (GPSDeniedProcessingMode_e)visionCtrlPBMsg.gpsdeniedprocessingmode();
                    cmdChanged = true;
                }
                if(cmdChanged)
                {
                    _imageProcessControlMsg->PostMessage();
                }

                if(_imageCaptureControlMsg->ImageCaptureEnabled != visionCtrlPBMsg.imagecaptureenabled())
                {
                    _imageCaptureControlMsg->ImageCaptureSource = ImageCaptureSource_e::ImageCaptureSource_NoChange;
                    _imageCaptureControlMsg->ImageCaptureEnabled = visionCtrlPBMsg.imagecaptureenabled();
                    _imageCaptureControlMsg->NumberOfImagesToCapture = visionCtrlPBMsg.numberofimagestocapture();
                    _imageCaptureControlMsg->PostMessage();
                }

                respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
                respMsg->set_cmdresponsemessage("[HANDLER] VisionProcessingControl OK");

            }
            catch (std::exception &e)
            {
                LOGERROR("VisionProcessingControlCmdHandler Exception: " << e.what());
                respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
                respMsg->set_cmdresponsemessage("[HANDLER] VisionProcessingControl ERROR");
            }
        }
    }

    void CommandResponseMessageHandler::VisionProcessCmdStatusHandler(VisionCommandPBMsg *cmdMsg,
                                                                     VisionResponsePBMsg *respMsg)
    {
        bool mgrNameFound = false;
        vision_messages::VisionProcessingControlPBMsg visionCtrlPBMsg;

        _imageCaptureStatusMsg->FetchMessage();
        visionCtrlPBMsg.set_imagecaptureenabled(_imageCaptureControlMsg->ImageCaptureEnabled);
        visionCtrlPBMsg.set_numberofimagestocapture(_imageCaptureControlMsg->NumberOfImagesToCapture);
        visionCtrlPBMsg.set_desiredframespersecond(_imageCaptureControlMsg->DesiredFramesPerSecond);

        _imageProcessControlStatusMsg->FetchMessage();
        visionCtrlPBMsg.set_targetimageprocessingenabled(_imageProcessControlStatusMsg->TargetImageProcessingEnabled);
        visionCtrlPBMsg.set_gpsdeniedprocessingenabled(_imageProcessControlStatusMsg->GPSDeniedProcessingEnabled);
        visionCtrlPBMsg.set_gpuprocessingenabled(_imageProcessControlStatusMsg->GPUProcessingEnabled);
        visionCtrlPBMsg.set_visionprocessingmode((::vision_messages::VisionProcessingControlPBMsg_VisionProcessingModeEnum)_imageProcessControlStatusMsg->VisionProcessingMode);
        visionCtrlPBMsg.set_targetprocessingmode((::vision_messages::VisionProcessingControlPBMsg_TargetProcessingModeEnum)_imageProcessControlStatusMsg->TargetProcessingMode);
        visionCtrlPBMsg.set_gpsdeniedprocessingmode((::vision_messages::VisionProcessingControlPBMsg_GPSDeniedProcessingModeEnum)_imageProcessControlStatusMsg->GPSDeniedProcessingMode);

        _streamRecordImageControlMsg->FetchMessage();
        _imageLoggingControlMsg->FetchMessage();
        visionCtrlPBMsg.set_recordimagesenabled(_imageLoggingControlMsg->EnableLogging);
        visionCtrlPBMsg.set_streamimagesenabled(_streamRecordImageControlMsg->StreamImagesEnabled);

        respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
        respMsg->set_cmdresponsemessage("[HANDLER] Retrieved Vision Process Cmd Status");
        respMsg->set_cmdresponsedata(visionCtrlPBMsg.SerializeAsString());
    }



    void CommandResponseMessageHandler::ImageCaptureStatusCmdHandler(VisionCommandPBMsg *cmdMsg,
                                                               VisionResponsePBMsg *respMsg)
    {
        bool mgrNameFound = false;
        vision_messages::ImageCaptureStatusPBMsg imgCapStatsPBMsg;
        _imageCaptureStatusMsg->FetchMessage();
        ImageCaptureStatusMsgToProtobufMsg(_imageCaptureStatusMsg, imgCapStatsPBMsg);
        respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
        respMsg->set_cmdresponsemessage("[HANDLER] Retrieved Image Capture Status");
        respMsg->set_cmdresponsedata(imgCapStatsPBMsg.SerializeAsString());
    }

    void CommandResponseMessageHandler::GetManagerStatsHandler(VisionCommandPBMsg *cmdMsg,
                                                                VisionResponsePBMsg *respMsg)
    {
        bool mgrNameFound = false;
        vision_messages::ManagerStatsPBMsg mgrStatsPBMsg;
        std::shared_ptr<Rabit::ManagerStatusMessage> mgrStatsMsg = nullptr;

        if (cmdMsg->has_cmdqualifier())
        {
            VisionSystemManagers_e mgrNo = GetVisionSystemManagerEnumFromName(cmdMsg->cmdqualifier());
            switch (mgrNo)
            {
                case VisionSystemManagers_e::VS_VidereSystemControlManager:
                    mgrStatsMsg = _videreSystemControMgrStatsMsg;
                    mgrNameFound = true;
                    break;
                case VisionSystemManagers_e::VS_HeadOrientationManager:
                    mgrStatsMsg = _headOrientationMgrStatsMsg;
                    mgrNameFound = true;
                    break;
                case VisionSystemManagers_e::VS_VehicleStateManager:
                    mgrStatsMsg = _vehicleStateMgrStatsMsg;
                    mgrNameFound = true;
                    break;
                case VisionSystemManagers_e::VS_CommsManager:
                    mgrStatsMsg = _commMgrStatsMsg;
                    mgrNameFound = true;
                    break;
                case VisionSystemManagers_e::VS_ImageCaptureManager:
                    mgrStatsMsg = _ImageCaptureMgrStatsMsg;
                    mgrNameFound = true;
                    break;
                case VisionSystemManagers_e::VS_StreamRecordManager:
                    mgrStatsMsg = _StreamRecordMgrStatsMsg;
                    mgrNameFound = true;
                    break;
                case VisionSystemManagers_e::VS_VisionProcessManager:
                    mgrStatsMsg = _VisionProcessMgrStatsMsg;
                    mgrNameFound = true;
                    break;
                case VisionSystemManagers_e::VS_IMUCommManager:
                    mgrStatsMsg = _IMUCommMgrStatsMsg;
                    mgrNameFound = true;
                    break;
                case VisionSystemManagers_e::VS_DTX_IMU_InterfaceManager:
                    mgrStatsMsg = _DTXIMUInterfaceMgrStatsMsg;
                    mgrNameFound = true;
                    break;
                case VisionSystemManagers_e::VS_GPSManager:
                    mgrStatsMsg = _GPSMgrStatsMsg;
                    mgrNameFound = true;
                    break;
                case VisionSystemManagers_e::VS_RobotArmManager:
                    mgrStatsMsg = _RobotArmMgrStatsMsg;
                    mgrNameFound = true;
                    break;
                case VisionSystemManagers_e::VS_SipnPuffManager:
                    mgrStatsMsg = _SipnPuffMgrStatsMsg;
                    mgrNameFound = true;
                    break;
                case VisionSystemManagers_e::VS_VehicleActuatorInterfaceManager:
                    mgrStatsMsg = _VehicleActuatorInterfaceMgrStatsMsg;
                    mgrNameFound = true;
                    break;
                case VisionSystemManagers_e::VS_RemoteControlManager:
                    mgrStatsMsg = _RemoteControlMgrStatsMsg;
                    mgrNameFound = true;
                    break;
                case VisionSystemManagers_e::VS_SystemInfoManager:
                    mgrStatsMsg = _SystemInfoMgrStatsMsg;
                    mgrNameFound = true;
                    break;
            }
        }

        if( mgrNameFound)
        {
            mgrStatsMsg->FetchMessage();
            ManagerStatsMsgToProtobufMsg(mgrStatsMsg, mgrStatsPBMsg);
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
            respMsg->set_cmdresponsemessage("[HANDLER] Retrieved Manager Stats");
            respMsg->set_cmdresponsedata(mgrStatsPBMsg.SerializeAsString());
        }
        else
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
            respMsg->set_cmdresponsemessage("[HANDLER] Invalid Manager Name: " + cmdMsg->cmdqualifier());
        }
    }



    void CommandResponseMessageHandler::MangerControlCmdHandler(VisionCommandPBMsg *cmdMsg,
                                 VisionResponsePBMsg *respMsg)
    {
        vision_messages::ManagerControlPBMsg mgrCtrlPBMsg;
        std::shared_ptr<Rabit::ManagerControlMessage> mgrCtrlMsg;

        bool ctrlMsgSet = false;
        if( cmdMsg->has_cmddata() )
        {
            try
            {
                mgrCtrlPBMsg.ParseFromString(cmdMsg->cmddata());
                ctrlMsgSet = true;
            }
            catch (std::exception &e)
            {
                LOGERROR("MangerControlCmdHandler Exception: " << e.what());
            }
        }
        if(ctrlMsgSet )
        {
            ctrlMsgSet = false;
            std::string mgrName = boost::algorithm::to_lower_copy(mgrCtrlPBMsg.managername());
            if(mgrName.compare(0, 3, "all") == 0 )
            {
                mgrCtrlMsg = _AllMgrCtrlMsg;
                ctrlMsgSet = true;
            }
            else
            {
                VisionSystemManagers_e mgrNo = GetVisionSystemManagerEnumFromName(mgrName);
                switch(mgrNo)
                {
                    case VisionSystemManagers_e::VS_VidereSystemControlManager:
                        mgrCtrlMsg = _videreSystemControMgrCtrlMsg;
                        ctrlMsgSet = true;
                        break;
                    case VisionSystemManagers_e::VS_HeadOrientationManager:
                        mgrCtrlMsg = _headOrientationMgrCtrlMsg;
                        ctrlMsgSet = true;
                        break;
                    case VisionSystemManagers_e::VS_VehicleStateManager:
                        mgrCtrlMsg = _vehicleStateMgrCtrlMsg;
                        ctrlMsgSet = true;
                        break;
                    case VisionSystemManagers_e::VS_CommsManager:
                        mgrCtrlMsg = _commMgrCtrlMsg;
                        ctrlMsgSet = true;
                        break;
                    case VisionSystemManagers_e::VS_ImageCaptureManager:
                        mgrCtrlMsg = _ImageCaptureMgrCtrlMsg;
                        ctrlMsgSet = true;
                        break;
                    case VisionSystemManagers_e::VS_StreamRecordManager:
                        mgrCtrlMsg = _StreamRecordMgrCtrlMsg;
                        ctrlMsgSet = true;
                        break;
                    case VisionSystemManagers_e::VS_VisionProcessManager:
                        mgrCtrlMsg = _VisionProcessMgrCtrlMsg;
                        ctrlMsgSet = true;
                        break;
                    case VisionSystemManagers_e::VS_IMUCommManager:
                        mgrCtrlMsg = _IMUCommMgrCtrlMsg;
                        ctrlMsgSet = true;
                        break;
                    case VisionSystemManagers_e::VS_DTX_IMU_InterfaceManager:
                        mgrCtrlMsg = _DTXIMUInterfaceMgrCtrlMsg;
                        ctrlMsgSet = true;
                        break;
                    case VisionSystemManagers_e::VS_GPSManager:
                        mgrCtrlMsg = _GPSMgrCtrlMsg;
                        ctrlMsgSet = true;
                        break;
                    case VisionSystemManagers_e::VS_RobotArmManager:
                        mgrCtrlMsg = _RobotArmMgrCtrlMsg;
                        ctrlMsgSet = true;
                        break;
                    case VisionSystemManagers_e::VS_SipnPuffManager:
                        mgrCtrlMsg = _SipnPuffMgrCtrlMsg;
                        ctrlMsgSet = true;
                        break;
                    case VisionSystemManagers_e::VS_VehicleActuatorInterfaceManager:
                        mgrCtrlMsg = _VehicleActuatorInterfaceMgrCtrlMsg;
                        ctrlMsgSet = true;
                        break;
                    case VisionSystemManagers_e::VS_RemoteControlManager:
                        mgrCtrlMsg = _RemoteControlMgrCtrlMsg;
                        ctrlMsgSet = true;
                        break;
                    case VisionSystemManagers_e::VS_SystemInfoManager:
                        mgrCtrlMsg = _SystemInfoMgrCtrlMsg;
                        ctrlMsgSet = true;
                        break;
                }
            }
            if(ctrlMsgSet )
            {
                if(mgrCtrlPBMsg.has_resetmgrstatstoggle() && mgrCtrlPBMsg.resetmgrstatstoggle())
                {
                    //Toggle the Reset Flag.
                    mgrCtrlMsg->ResetMgrStatsToggle = mgrCtrlMsg->ResetMgrStatsToggle ? false : true;
                }
                if( mgrCtrlPBMsg.has_publishmgrstatstime_sec())
                {
                    mgrCtrlMsg->PublishMgrStatsTime_Sec = mgrCtrlPBMsg.publishmgrstatstime_sec();
                }
                if( mgrCtrlPBMsg.has_shutdownmanager() && mgrCtrlPBMsg.shutdownmanager() == true )
                {
                    mgrCtrlMsg->ShutdownManager(true);
                }
                mgrCtrlMsg->PostMessage();
                respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
                respMsg->set_cmdresponsemessage("[HANDLER] Set Manager Control");
            }
            else
            {
                respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
                respMsg->set_cmdresponsemessage("[HANDLER] Invalid Manager Name");
                LOGERROR("MangerControlCmdHandler Invalid Mgr Name: " << mgrCtrlPBMsg.managername());
            }
        }
        else
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
            respMsg->set_cmdresponsemessage("[HANDLER] Could Not read Manager Cmd");
        }
    }


    void CommandResponseMessageHandler::SetupGeoCoordinateSystemCmdHandler(VisionCommandPBMsg *cmdMsg,
                                                                VisionResponsePBMsg *respMsg)
    {
        bool error = false;
        vision_messages::GeoCoordinateSystemSetupPBMsg geoCSetupPBMsg;

        if (cmdMsg->has_cmddata())
        {
            try
            {
                geoCSetupPBMsg.ParseFromString(cmdMsg->cmddata());
                GeoCoordinateSystem *gcsPtr = GeoCoordinateSystem::GetGeoCoordinateSystemReference();
                GeoCoordinateSystemConversionType_e conversionType = GeoCoordinateSystemConversionType_e::Linear;
                if (geoCSetupPBMsg.geocoordinatesystemconversiontype() ==
                    vision_messages::GeoCoordinateSystemSetupPBMsg_GeoCoordinateSystemConversionType_e_WGS84_Relative)
                {
                    conversionType = GeoCoordinateSystemConversionType_e::WGS84_Relative;
                } else if (geoCSetupPBMsg.geocoordinatesystemconversiontype() ==
                           vision_messages::GeoCoordinateSystemSetupPBMsg_GeoCoordinateSystemConversionType_e_WGS84_Map)
                {
                    conversionType = GeoCoordinateSystemConversionType_e::WGS84_Map;
                }


                if (conversionType == GeoCoordinateSystemConversionType_e::Linear
                    && geoCSetupPBMsg.deltalatitudedegrees() > 0 && geoCSetupPBMsg.deltalongitudedegrees() > 0)
                {
                    //Create a List of Corner Lat/Lon items and pass to setup.
                    LatLonAltCoord_t CenterLatLonAlt(geoCSetupPBMsg.centerlatitudedegrees(),
                                                     geoCSetupPBMsg.centerlongitudedegrees(),
                                                     geoCSetupPBMsg.groundaltitudemsl(), true);
                    LatLonAltCoord_t DeltaLLA(geoCSetupPBMsg.deltalatitudedegrees(),
                                              geoCSetupPBMsg.deltalongitudedegrees(), 0, true);
                    std::vector<LatLonAltCoord_t> latLonList(2);
                    latLonList[0] = CenterLatLonAlt + DeltaLLA;
                    latLonList[1] = CenterLatLonAlt - DeltaLLA;
                    error = gcsPtr->SetupGeoCoordinateSystem(latLonList, conversionType);
                } else
                {
                    error = gcsPtr->SetupGeoCoordinateSystem(geoCSetupPBMsg.centerlatitudedegrees(),
                                                             geoCSetupPBMsg.centerlongitudedegrees(),
                                                             geoCSetupPBMsg.groundaltitudemsl(), true,
                                                             conversionType);
                }
                if (error)
                {
                    respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
                    respMsg->set_cmdresponsemessage("[HANDLER] GeoCoordinateSystemSetupPBMsg Setup Error.");
                    LOGERROR("GeoCoordinateSystemSetupPBMsg Setup Error.  ConversionType=" << conversionType
                                                                                           << " CenterLat="
                                                                                           << geoCSetupPBMsg.centerlatitudedegrees()
                                                                                           << " CenterLon="
                                                                                           << geoCSetupPBMsg.centerlongitudedegrees()
                                                                                           << " GroundAlt="
                                                                                           << geoCSetupPBMsg.groundaltitudemsl());
                } else
                {
                    respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
                    respMsg->set_cmdresponsemessage("[HANDLER] SetupGeoCoordinateSystemCmdHandler");
                    LOGERROR("GeoCoordinateSystemSetupPBMsg Setup Error.  ConversionType=" << conversionType
                                                                                           << " CenterLat="
                                                                                           << geoCSetupPBMsg.centerlatitudedegrees()
                                                                                           << " CenterLon="
                                                                                           << geoCSetupPBMsg.centerlongitudedegrees()
                                                                                           << " GroundAlt="
                                                                                           << geoCSetupPBMsg.groundaltitudemsl());
                }
            }
            catch (std::exception &e)
            {
                respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
                respMsg->set_cmdresponsemessage("[HANDLER] GeoCoordinateSystemSetupPBMsg Setup Error.");
                LOGERROR("GeoCoordinateSystemSetupPBMsg Exception: " << e.what());
            }
        }
        else
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
            respMsg->set_cmdresponsemessage("[HANDLER] GeoCoordinateSystemSetupPBMsg No GeoCoord Setup Message Passed in.");
            LOGWARN("GeoCoordinateSystemSetupPBMsg No GeoCoord Setup Message Passed in.");
        }
    }


    void CommandResponseMessageHandler::GetGeoCoordinateSetupHandler(VisionCommandPBMsg *cmdMsg,
                                                               VisionResponsePBMsg *respMsg)
    {
        bool mgrNameFound = false;
        vision_messages::GeoCoordinateSystemSetupPBMsg geoCSetupPBMsg;

        GeoCoordinateSystem * gcsPtr = GeoCoordinateSystem::GetGeoCoordinateSystemReference();

        geoCSetupPBMsg.set_geocoordinatesystemconversiontype(vision_messages::GeoCoordinateSystemSetupPBMsg_GeoCoordinateSystemConversionType_e_Linear);

        if( gcsPtr->GetConversionType() == GeoCoordinateSystemConversionType_e::WGS84_Relative)
        {
            geoCSetupPBMsg.set_geocoordinatesystemconversiontype(vision_messages::GeoCoordinateSystemSetupPBMsg_GeoCoordinateSystemConversionType_e_WGS84_Relative);
        }
        else if( gcsPtr->GetConversionType() == GeoCoordinateSystemConversionType_e::WGS84_Map)
        {
            geoCSetupPBMsg.set_geocoordinatesystemconversiontype(vision_messages::GeoCoordinateSystemSetupPBMsg_GeoCoordinateSystemConversionType_e_WGS84_Map);
        }

        LatLonAltCoord_t lla = gcsPtr->ReferenceLatLonAltLocation();
        geoCSetupPBMsg.set_centerlatitudedegrees(lla.LatitudeDegrees());
        geoCSetupPBMsg.set_centerlongitudedegrees(lla.LongitudeDegrees());
        geoCSetupPBMsg.set_groundaltitudemsl(lla.Altitude());

        respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
        respMsg->set_cmdresponsemessage("[HANDLER] Get GeoCoordinate Setup");
        respMsg->set_cmdresponsedata(geoCSetupPBMsg.SerializeAsString());
    }

    //This is primarily for test purposes... a Lat/Lon value will
    //be converted to the X-Y coordinate... or an X-Y coordinate
    //will be converted to the Lat/Lon position.  Lat/Lon are in Degrees.
    void CommandResponseMessageHandler::LatLonXYConversionHandler(VisionCommandPBMsg *cmdMsg,
                                                                     VisionResponsePBMsg *respMsg)
    {
        //The result will be returned in the same message... the missing values
        //will be filled in.
        vision_messages::LatLonXYConversionPBMsg latLonXYPBMsg;
        LatLonAltCoord_t latLonCoord;
        XYZCoord_t xyzCoord;
        if (cmdMsg->has_cmddata())
        {
            try
            {
                latLonXYPBMsg.ParseFromString(cmdMsg->cmddata());
                GeoCoordinateSystem * gcsPtr = GeoCoordinateSystem::GetGeoCoordinateSystemReference();
                if(latLonXYPBMsg.latlontoxyconversion())
                {
                    latLonCoord.SetLatitudeDegrees(latLonXYPBMsg.latitudedegrees());
                    latLonCoord.SetLongitudeDegrees(latLonXYPBMsg.longitudedegrees());
                    xyzCoord = gcsPtr->LatLonAltToXYZ(latLonCoord);
                    latLonXYPBMsg.set_x_posmeters(xyzCoord.x);
                    latLonXYPBMsg.set_y_posmeters(xyzCoord.y);
                }
                else
                {
                    xyzCoord.x = latLonXYPBMsg.x_posmeters();
                    xyzCoord.y = latLonXYPBMsg.y_posmeters();
                    latLonCoord = gcsPtr->XYZToLatLonAlt(xyzCoord);
                    latLonXYPBMsg.set_latitudedegrees(latLonCoord.LatitudeDegrees());
                    latLonXYPBMsg.set_longitudedegrees(latLonCoord.LongitudeDegrees());
                }
                respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
                respMsg->set_cmdresponsemessage("[HANDLER] LatLonXY Conversion");
                respMsg->set_cmdresponsedata(latLonXYPBMsg.SerializeAsString());
            }
            catch (std::exception &e)
            {
                respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
                respMsg->set_cmdresponsemessage("[HANDLER] LatLonXYConversionHandler Error.");
                LOGERROR("LatLonXYConversionHandler Exception: " << e.what());
            }
        }
        else
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
            respMsg->set_cmdresponsemessage("[HANDLER] LatLonXYConversionHandler No LatLonXY Message Passed in.");
            LOGWARN("LatLonXYConversionHandler No LatLonXY Message Passed in.");
        }

    }


    //Process Incoming Target Parameters Messages.
    void CommandResponseMessageHandler::TargetParametersMsgHandler(VisionCommandPBMsg *cmdMsg,
                                                                  VisionResponsePBMsg *respMsg)
    {
        //The result will be returned in the same message... the missing values
        //will be filled in.
        vision_messages::TargetParametersPBMsg targetInfoPBMsg;
        if (cmdMsg->has_cmddata())
        {
            try
            {
                targetInfoPBMsg.ParseFromString(cmdMsg->cmddata());
                if(targetInfoPBMsg.targettypecode() == 1)
                {
                    _targetType1ParamsMsg->SetMessage(targetInfoPBMsg);
                    _targetType1ParamsMsg->PostMessage();
                    respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
                    respMsg->set_cmdresponsemessage("[HANDLER] Target Type 1 Parameters Set");
                }
                else if(targetInfoPBMsg.targettypecode() == 2)
                {
                    _targetType2ParamsMsg->SetMessage(targetInfoPBMsg);
                    _targetType2ParamsMsg->PostMessage();
                    respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
                    respMsg->set_cmdresponsemessage("[HANDLER] Target Type 2 Parameters Set");
                }
                else if(targetInfoPBMsg.targettypecode() == 3)
                {
                    _targetType3ParamsMsg->SetMessage(targetInfoPBMsg);
                    _targetType3ParamsMsg->PostMessage();
                    respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
                    respMsg->set_cmdresponsemessage("[HANDLER] Target Type 3 Parameters Set");
                }
                else if(targetInfoPBMsg.targettypecode() == 4)
                {
                    _targetType4ParamsMsg->SetMessage(targetInfoPBMsg);
                    _targetType4ParamsMsg->PostMessage();
                    respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
                    respMsg->set_cmdresponsemessage("[HANDLER] Target Type 4 Parameters Set");
                }
                else
                {
                    respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
                    respMsg->set_cmdresponsemessage("[HANDLER] Target Type Not Supported");
                    LOGWARN("Target Type Not Supported: " << targetInfoPBMsg.targettypecode());
                }
             }
            catch (std::exception &e)
            {
                respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
                respMsg->set_cmdresponsemessage("[HANDLER] TargetParametersMsgHandler Error.");
                LOGERROR("LatLonXYConversionHandler Exception: " << e.what());
            }
        }
        else
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
            respMsg->set_cmdresponsemessage("[HANDLER] TargetParametersPBMsg No Message Passed in.");
            LOGWARN("TargetParametersPBMsg No Message Passed in.");
        }

    }

    void CommandResponseMessageHandler::CameraCalCommandCmdMsgHandler(VisionCommandPBMsg *cmdMsg,
                                                                      VisionResponsePBMsg *respMsg)
    {
        vision_messages::CameraCalControlPBMsg cCalCtrlPBMsg;
        std::shared_ptr<Rabit::ManagerControlMessage> mgrCtrlMsg;

        bool ctrlMsgSet = false;
        if( cmdMsg->has_cmddata() )
        {
            try
            {
                cCalCtrlPBMsg.ParseFromString(cmdMsg->cmddata());

                _cameraCalControlMsg->CameraCalibrationType = (CameraCalibrationType_e)cCalCtrlPBMsg.cameracalibrationtype();
                _cameraCalControlMsg->CameraCalCmd = (CameraCalCmd_e)cCalCtrlPBMsg.cameracalcmd();
                _cameraCalControlMsg->CameraCalBaseFilename = cCalCtrlPBMsg.cameracalbasefilename();
                _cameraCalControlMsg->NumberOfCols = cCalCtrlPBMsg.numberofcols();
                _cameraCalControlMsg->NumberOfRows = cCalCtrlPBMsg.numberofrows();
                _cameraCalControlMsg->SquareSizeMilliMeters = cCalCtrlPBMsg.squaresizemillimeters();
                _cameraCalControlMsg->YawCorrectionDegrees = cCalCtrlPBMsg.yawcorrectiondegrees();
                _cameraCalControlMsg->PitchCorrectionDegrees = cCalCtrlPBMsg.pitchcorrectiondegrees();
                _cameraCalControlMsg->RollCorrectionDegrees = cCalCtrlPBMsg.rollcorrectiondegrees();
                _cameraCalControlMsg->DelXCorrectionCentiMeters = cCalCtrlPBMsg.delxcorrectioncentimeters();
                _cameraCalControlMsg->DelYCorrectionCentiMeters = cCalCtrlPBMsg.delycorrectioncentimeters();
                _cameraCalControlMsg->DelZCorrectionCentiMeters = cCalCtrlPBMsg.delzcorrectioncentimeters();
                _cameraCalControlMsg->PostMessage();
                ctrlMsgSet = true;
            }
            catch (std::exception &e)
            {
                LOGERROR("CameraCalCommandCmdHandler Exception: " << e.what());
            }
        }
        if(ctrlMsgSet )
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
            respMsg->set_cmdresponsemessage("[HANDLER] OK");
        }
        else
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
            respMsg->set_cmdresponsemessage("[HANDLER] Could Not read CameraCalControl Message");
        }
    }

    void CommandResponseMessageHandler::FeatureMatchProcCtrlMsgHandler(VisionCommandPBMsg *cmdMsg,
                                                                      VisionResponsePBMsg *respMsg)
    {
        vision_messages::FeatureMatchProcCtrlPBMsg fmCtrlPBMsg;
        std::shared_ptr<Rabit::ManagerControlMessage> mgrCtrlMsg;

        bool ctrlMsgSet = false;
        if( cmdMsg->has_cmddata() )
        {
            try
            {
                fmCtrlPBMsg.ParseFromString(cmdMsg->cmddata());

                _featureMatchProcCtrlMsg->FeatureMatchingProcCmd = (FeatureMatchingProcCmd_e)fmCtrlPBMsg.featurematchingproccmd();
                _featureMatchProcCtrlMsg->FeatureExtractionTypeRoutine = (FeatureExtractionTypeRoutine_e)fmCtrlPBMsg.featureextractiontyperoutine();
                _featureMatchProcCtrlMsg->FeatureMatchTypeRoutine = (FeatureMatchTypeRoutine_e)fmCtrlPBMsg.featurematchtyperoutine();
                _featureMatchProcCtrlMsg->FMImagePostProcessMethod = (FMImagePostProcessMethod_e)fmCtrlPBMsg.fmimagepostprocessmethod();
                _featureMatchProcCtrlMsg->UseGPUProcessing = fmCtrlPBMsg.usegpuprocessing();

                _featureMatchProcCtrlMsg->ParamI_1 = fmCtrlPBMsg.parami_1();
                _featureMatchProcCtrlMsg->ParamI_2 = fmCtrlPBMsg.parami_2();
                _featureMatchProcCtrlMsg->ParamI_3 = fmCtrlPBMsg.parami_3();
                _featureMatchProcCtrlMsg->ParamI_4 = fmCtrlPBMsg.parami_4();
                _featureMatchProcCtrlMsg->ParamI_5 = fmCtrlPBMsg.parami_5();
                _featureMatchProcCtrlMsg->ParamI_6 = fmCtrlPBMsg.parami_6();
                _featureMatchProcCtrlMsg->ParamI_7 = fmCtrlPBMsg.parami_7();
                _featureMatchProcCtrlMsg->ParamI_8 = fmCtrlPBMsg.parami_8();
                _featureMatchProcCtrlMsg->ParamI_9 = fmCtrlPBMsg.parami_9();

                _featureMatchProcCtrlMsg->ParamF_10 = fmCtrlPBMsg.paramf_10();
                _featureMatchProcCtrlMsg->ParamF_11 = fmCtrlPBMsg.paramf_11();
                _featureMatchProcCtrlMsg->ParamF_12 = fmCtrlPBMsg.paramf_12();
                _featureMatchProcCtrlMsg->ParamF_13 = fmCtrlPBMsg.paramf_13();
                _featureMatchProcCtrlMsg->ParamF_14 = fmCtrlPBMsg.paramf_14();
                _featureMatchProcCtrlMsg->ParamF_15 = fmCtrlPBMsg.paramf_15();
                _featureMatchProcCtrlMsg->ParamF_16 = fmCtrlPBMsg.paramf_16();
                _featureMatchProcCtrlMsg->ParamF_17 = fmCtrlPBMsg.paramf_17();
                _featureMatchProcCtrlMsg->ParamF_18 = fmCtrlPBMsg.paramf_18();
                _featureMatchProcCtrlMsg->ParamF_19 = fmCtrlPBMsg.paramf_19();

                _featureMatchProcCtrlMsg->PostMessage();
                ctrlMsgSet = true;
            }
            catch (std::exception &e)
            {
                LOGERROR("FeatureMatchProcCtrlMsgHandler Exception: " << e.what());
            }
        }
        if(ctrlMsgSet )
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
            respMsg->set_cmdresponsemessage("[HANDLER] OK");
        }
        else
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
            respMsg->set_cmdresponsemessage("[HANDLER] Could Not read CameraCalControl Message");
        }
    }




    void CommandResponseMessageHandler::VehicleInertialStatesMsgHandler(VisionCommandPBMsg *cmdMsg,
                                                                      VisionResponsePBMsg *respMsg)
    {
        vision_messages::VehicleInertialStatePBMsg visPBMsg;
        std::shared_ptr <Rabit::ManagerControlMessage> mgrCtrlMsg;

        bool ctrlMsgSet = false;
        if (cmdMsg->has_cmddata())
        {
            try
            {
                visPBMsg.ParseFromString(cmdMsg->cmddata());
                VehicleInertialStatesPBMsgToVISMsg(visPBMsg, _vehicleInertialStatesMsg);
                _vehicleInertialStatesMsg->PostMessage();
                ctrlMsgSet = true;
            }
            catch (std::exception &e)
            {
                LOGERROR("VehicleInertialStatesMsgHandler Exception: " << e.what());
            }
        }
        if (ctrlMsgSet)
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
            respMsg->set_cmdresponsemessage("[HANDLER] OK");
        } else
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
            respMsg->set_cmdresponsemessage("[HANDLER] Could Not read Vehicle Inertial States Message");
        }
    }

    void CommandResponseMessageHandler::CameraOrientationMsgHandler(VisionCommandPBMsg *cmdMsg,
                                                                        VisionResponsePBMsg *respMsg)
    {
        vision_messages::CameraSteeringPBMsg csPBMsg;
        std::shared_ptr <Rabit::ManagerControlMessage> mgrCtrlMsg;

        bool ctrlMsgSet = false;
        if (cmdMsg->has_cmddata())
        {
            try
            {
                csPBMsg.ParseFromString(cmdMsg->cmddata());
                CameraSteeringPBMsgToCOMsg(csPBMsg, _cameraOrientationMsg);
                _cameraOrientationMsg->PostMessage();
                ctrlMsgSet = true;
            }
            catch (std::exception &e)
            {
                LOGERROR("CameraOrientationMsgHandler Exception: " << e.what());
            }
        }
        if (ctrlMsgSet)
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
            respMsg->set_cmdresponsemessage("[HANDLER] OK");
        } else
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
            respMsg->set_cmdresponsemessage("[HANDLER] Could Not read Camera Orientation Message");
        }
    }

    void CommandResponseMessageHandler::CameraParametersSetupMsgHandler(VisionCommandPBMsg *cmdMsg,
                                                                    VisionResponsePBMsg *respMsg)
    {
        vision_messages::CameraParametersSetupPBMsg csPBMsg;

        bool ctrlMsgSet = false;
        if (cmdMsg->has_cmddata())
        {
            try
            {
                csPBMsg.ParseFromString(cmdMsg->cmddata());
                _cameraParametersSetupMsg->Clear();
                _cameraParametersSetupMsg->ImageCaptureFormat = (ImageCaptureFormat_e)csPBMsg.imagecaptureformat();
                _cameraParametersSetupMsg->Mode = csPBMsg.mode();
                _cameraParametersSetupMsg->FrameWidth = csPBMsg.framewidth();
                _cameraParametersSetupMsg->FrameHeight = csPBMsg.frameheight();
                _cameraParametersSetupMsg->FrameRateFPS = csPBMsg.frameratefps();
                _cameraParametersSetupMsg->Autofocus = csPBMsg.autofocus();
                _cameraParametersSetupMsg->FocusValue = csPBMsg.focusvalue();
                _cameraParametersSetupMsg->Brightness = csPBMsg.brightness();
                _cameraParametersSetupMsg->Contrast = csPBMsg.contrast();
                _cameraParametersSetupMsg->Saturation = csPBMsg.saturation();
                _cameraParametersSetupMsg->Hue = csPBMsg.hue();
                _cameraParametersSetupMsg->Gain = csPBMsg.gain();
                _cameraParametersSetupMsg->Exposure = csPBMsg.exposure();
                _cameraParametersSetupMsg->WhiteBallanceBlue = csPBMsg.whiteballanceblue();
                _cameraParametersSetupMsg->WhiteBallanceRed = csPBMsg.whiteballancered();
                _cameraParametersSetupMsg->ExternalTrigger = csPBMsg.externaltrigger();

                _cameraParametersSetupMsg->PostMessage();
                ctrlMsgSet = true;
            }
            catch (std::exception &e)
            {
                LOGERROR("CameraParametersSetupMsgHandler Exception: " << e.what());
            }
        }
        if (ctrlMsgSet)
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
            respMsg->set_cmdresponsemessage("[HANDLER] OK");
        } else
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
            respMsg->set_cmdresponsemessage("[HANDLER] Could Not read Camera ParametersSetup Message");
        }
    }

    void CommandResponseMessageHandler::GetListOfManagerNamesMsgHandler(VisionCommandPBMsg *cmdMsg,
                                         VisionResponsePBMsg *respMsg)
    {
        vision_messages::ListOfManagerNamesPBMsg listOfMgrsPBMsg;
        try
        {
            vector<string> mgrNameList = _mgrPtr->GetListOfManagerNames();
            listOfMgrsPBMsg.set_numberofmanagers(mgrNameList.size());
            for(int i = 0; i < mgrNameList.size(); i++)
            {
                listOfMgrsPBMsg.add_listofmanagernames(mgrNameList[i]);
                //listOfMgrsPBMsg.set_listofmanagernames(i, mgrNameList[i]);
            }
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
            respMsg->set_cmdresponsemessage("[HANDLER] OK");
            respMsg->set_cmdresponsedata(listOfMgrsPBMsg.SerializeAsString());

            //set_TargetCornerLocations
        }
        catch (std::exception &e)
        {
            LOGERROR("GetListOfManagerNamesMsgHandler Exception: " << e.what());
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
            respMsg->set_cmdresponsemessage("[HANDLER] Could Not Get Manager Name list.");
        }
    }



    void CommandResponseMessageHandler::HeadTrackingControlMsgHandler(VisionCommandPBMsg *cmdMsg,
                                                               VisionResponsePBMsg *respMsg)
    {
        vision_messages::HeadTrackingControlPBMsg csPBMsg;
        bool ctrlMsgSet = false;
        if (cmdMsg->has_cmddata())
        {
            try
            {
                csPBMsg.ParseFromString(cmdMsg->cmddata());
                _headTrackingControlMsg->HeadTrackingParameters.Canny_low = csPBMsg.canny_low();
                _headTrackingControlMsg->HeadTrackingParameters.Canny_high = csPBMsg.canny_high();
                _headTrackingControlMsg->HeadTrackingParameters.GlyphAreaPixels_min = csPBMsg.glyphareapixels_min();
                _headTrackingControlMsg->HeadTrackingParameters.GlyphAreaPixels_max = csPBMsg.glyphareapixels_max();
                _headTrackingControlMsg->HeadTrackingParameters.NumberOfIterations = csPBMsg.numberofiterations();
                _headTrackingControlMsg->HeadTrackingParameters.ReprojectionErrorDistance = csPBMsg.reprojectionerrordistance();
                _headTrackingControlMsg->HeadTrackingParameters.ConfidencePercent = csPBMsg.confidencepercent();
                _headTrackingControlMsg->HeadTrackingImageDisplayType = (ImageProcLibsNS::HeadTrackingImageDisplayType_e)csPBMsg.headtrackingimagedisplaytype();
                _headTrackingControlMsg->GlyphModelIndex = (int)csPBMsg.glyphmodelindex();
                 _headTrackingControlMsg->PostMessage();
                ctrlMsgSet = true;
            }
            catch (std::exception &e)
            {
                LOGERROR("BrakeControlMsgHandler Exception: " << e.what());
            }
        }
        if (ctrlMsgSet)
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
            respMsg->set_cmdresponsemessage("[HANDLER] OK");
        } else
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
            respMsg->set_cmdresponsemessage("[HANDLER] Could Not read Head Tracking Control Message");
        }
    }


    void CommandResponseMessageHandler::BrakeControlMsgHandler(VisionCommandPBMsg *cmdMsg,
                                                                    VisionResponsePBMsg *respMsg)
    {
        vision_messages::LinearActuatorPositionCtrlPBMsg csPBMsg;
        bool ctrlMsgSet = false;
        if (cmdMsg->has_cmddata())
        {
            try
            {
                csPBMsg.ParseFromString(cmdMsg->cmddata());
                _brakeCtrlMsg->FunctionType = LinearActuatorFunction_e::LA_Brake;
                _brakeCtrlMsg->ClutchEnable = csPBMsg.clutchenable();
                _brakeCtrlMsg->MotorEnable = csPBMsg.motorenable();
                _brakeCtrlMsg->ManualExtControl = csPBMsg.manualextcontrol();
                _brakeCtrlMsg->ActuatorSetupMode = csPBMsg.actuatorsetupmode();
                _brakeCtrlMsg->setPositionPercent(csPBMsg.positionpercent());
                _brakeCtrlMsg->PostMessage();
                ctrlMsgSet = true;
            }
            catch (std::exception &e)
            {
                LOGERROR("BrakeControlMsgHandler Exception: " << e.what());
            }
        }
        if (ctrlMsgSet)
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
            respMsg->set_cmdresponsemessage("[HANDLER] OK");
        } else
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
            respMsg->set_cmdresponsemessage("[HANDLER] Could Not read Brake Control Message");
        }
    }

    void CommandResponseMessageHandler::BrakeSetupMsgHandler(VisionCommandPBMsg *cmdMsg,
                                                               VisionResponsePBMsg *respMsg)
    {
        vision_messages::KarTechLinearActuatorSetupPBMsg csPBMsg;
        bool ctrlMsgSet = false;
        if (cmdMsg->has_cmddata())
        {
            try
            {
                csPBMsg.ParseFromString(cmdMsg->cmddata());
                _brakeActuatorSetuplMsg->FunctionType = LinearActuatorFunction_e::LA_Brake;
                _brakeActuatorSetuplMsg->ResetOutputs = csPBMsg.resetoutputs();
                _brakeActuatorSetuplMsg->ResetHardwareCfgs = csPBMsg.resethardwarecfgs();
                _brakeActuatorSetuplMsg->ResetUserCfgs = csPBMsg.resetusercfgs();
                _brakeActuatorSetuplMsg->ResetAll = csPBMsg.resetall();
                _brakeActuatorSetuplMsg->AutoZeroCal = csPBMsg.autozerocal();
                _brakeActuatorSetuplMsg->SetCanCommandResponsIDs = csPBMsg.setcancommandresponsids();
                _brakeActuatorSetuplMsg->PostMessage();
                ctrlMsgSet = true;
            }
            catch (std::exception &e)
            {
                LOGERROR("BrakeControlMsgHandler Exception: " << e.what());
            }
        }
        if (ctrlMsgSet)
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
            respMsg->set_cmdresponsemessage("[HANDLER] OK");
        } else
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
            respMsg->set_cmdresponsemessage("[HANDLER] Could Not read Brake Setup Message");
        }
    }

    void CommandResponseMessageHandler::SetBrakeConfigParamsMsgHandler(VisionCommandPBMsg *cmdMsg,
                                        VisionResponsePBMsg *respMsg)
    {
        vision_messages::KarTechLinearActuatorParamsPBMsg csPBMsg;
        bool ctrlMsgSet = false;
        if (cmdMsg->has_cmddata())
        {
            try
            {
                csPBMsg.ParseFromString(cmdMsg->cmddata());
                _brakeActuatorParamsControlMsg->FunctionType = LinearActuatorFunction_e::LA_Brake;
                _brakeActuatorParamsControlMsg->setMinPositionInches(csPBMsg.minpositioninches());
                _brakeActuatorParamsControlMsg->setMaxPositionInches(csPBMsg.maxpositioninches());
                _brakeActuatorParamsControlMsg->setFeedbackCtrl_KP(csPBMsg.feedbackctrl_kp());
                _brakeActuatorParamsControlMsg->setFeedbackCtrl_KI(csPBMsg.feedbackctrl_ki());
                _brakeActuatorParamsControlMsg->setFeedbackCtrl_KD(csPBMsg.feedbackctrl_kd());
                _brakeActuatorParamsControlMsg->setFeedbackCtrl_CLFreq(csPBMsg.feedbackctrl_clfreq());
                _brakeActuatorParamsControlMsg->setMotor_MinPWM(csPBMsg.motor_minpwm());
                _brakeActuatorParamsControlMsg->setMotor_MaxPWM(csPBMsg.motor_maxpwm());
                _brakeActuatorParamsControlMsg->setMotor_pwmFreq(csPBMsg.motor_pwmfreq());
                _brakeActuatorParamsControlMsg->setFeedbackCtrl_ErrDeadbandInces(csPBMsg.feedbackctrl_errdeadbandinces());
                _brakeActuatorParamsControlMsg->setPositionReachedErrorTimeMSec(csPBMsg.positionreachederrortimemsec());
                _brakeActuatorParamsControlMsg->setMotorMaxCurrentLimitAmps(csPBMsg.motormaxcurrentlimitamps());
                _brakeActuatorParamsControlMsg->PostMessage();
                ctrlMsgSet = true;
            }
            catch (std::exception &e)
            {
                LOGERROR("SetBrakeConfigParamsMsgHandler Exception: " << e.what());
            }
        }
        if (ctrlMsgSet)
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
            respMsg->set_cmdresponsemessage("[HANDLER] OK");
        } else
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
            respMsg->set_cmdresponsemessage("[HANDLER] Could Not read Brake Parameters Control Message");
        }
    }


    void CommandResponseMessageHandler::GetBrakeConfigParamsMsgHandler(VisionCommandPBMsg *cmdMsg,
                                                                        VisionResponsePBMsg *respMsg)
    {
        vision_messages::KarTechLinearActuatorParamsPBMsg ktlaPramsPBMsg;
        try
        {
            _brakeActuatorParamsFeedbackMsg->FetchMessage();
            ktlaPramsPBMsg.set_minpositioninches(_brakeActuatorParamsFeedbackMsg->getMinPositionInches());
            ktlaPramsPBMsg.set_maxpositioninches(_brakeActuatorParamsFeedbackMsg->getMaxPositionInches());

            ktlaPramsPBMsg.set_feedbackctrl_kp(_brakeActuatorParamsFeedbackMsg->getFeedbackCtrl_KP());
            ktlaPramsPBMsg.set_feedbackctrl_ki(_brakeActuatorParamsFeedbackMsg->getFeedbackCtrl_KI());
            ktlaPramsPBMsg.set_feedbackctrl_kd(_brakeActuatorParamsFeedbackMsg->getFeedbackCtrl_KD());
            ktlaPramsPBMsg.set_feedbackctrl_clfreq(_brakeActuatorParamsFeedbackMsg->getFeedbackCtrl_CLFreq());

            ktlaPramsPBMsg.set_motor_minpwm(_brakeActuatorParamsFeedbackMsg->getMotor_MinPWM());
            ktlaPramsPBMsg.set_motor_maxpwm(_brakeActuatorParamsFeedbackMsg->getMotor_MaxPWM());
            ktlaPramsPBMsg.set_motor_pwmfreq(_brakeActuatorParamsFeedbackMsg->getMotor_pwmFreq());

            ktlaPramsPBMsg.set_feedbackctrl_errdeadbandinces(_brakeActuatorParamsFeedbackMsg->getFeedbackCtrl_ErrDeadbandInces());
            ktlaPramsPBMsg.set_positionreachederrortimemsec(_brakeActuatorParamsFeedbackMsg->getPositionReachedErrorTimeMSec());
            ktlaPramsPBMsg.set_motormaxcurrentlimitamps(_brakeActuatorParamsFeedbackMsg->getMotorMaxCurrentLimitAmps());

            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
            respMsg->set_cmdresponsemessage("[HANDLER] OK");
            respMsg->set_cmdresponsedata(ktlaPramsPBMsg.SerializeAsString());

            //set_TargetCornerLocations
        }
        catch (std::exception &e)
        {
            LOGERROR("GetBrakeConfigParamsMsgHandler Exception: " << e.what());
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
            respMsg->set_cmdresponsemessage("[HANDLER] Could Not Get Linear Actuator Params.");
        }
    }

    void CommandResponseMessageHandler::ThrottleControlMsgHandler(VisionCommandPBMsg *cmdMsg,
                                                                VisionResponsePBMsg *respMsg)
    {
        vision_messages::LinearActuatorPositionCtrlPBMsg csPBMsg;
        bool ctrlMsgSet = false;
        if (cmdMsg->has_cmddata())
        {
            try
            {
                csPBMsg.ParseFromString(cmdMsg->cmddata());
                _throttleCtrlMsg->FunctionType = LinearActuatorFunction_e::LA_Accelerator;
                _throttleCtrlMsg->ClutchEnable = csPBMsg.clutchenable();
                _throttleCtrlMsg->MotorEnable = csPBMsg.motorenable();
                _throttleCtrlMsg->ManualExtControl = csPBMsg.manualextcontrol();
                _throttleCtrlMsg->ActuatorSetupMode = csPBMsg.actuatorsetupmode();
                _throttleCtrlMsg->setPositionPercent(csPBMsg.positionpercent());
                _throttleCtrlMsg->PostMessage();
                ctrlMsgSet = true;
            }
            catch (std::exception &e)
            {
                LOGERROR("ThrottleControlMsgHandler Exception: " << e.what());
            }
        }
        if (ctrlMsgSet)
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
            respMsg->set_cmdresponsemessage("[HANDLER] OK");
        } else
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
            respMsg->set_cmdresponsemessage("[HANDLER] Could Not read Throttle Control Message");
        }
    }

    void CommandResponseMessageHandler::ThrottleSetupMsgHandler(VisionCommandPBMsg *cmdMsg,
                                                             VisionResponsePBMsg *respMsg)
    {
        vision_messages::KarTechLinearActuatorSetupPBMsg csPBMsg;
        bool ctrlMsgSet = false;
        if (cmdMsg->has_cmddata())
        {
            try
            {
                csPBMsg.ParseFromString(cmdMsg->cmddata());
                _throttleActuatorSetuplMsg->FunctionType = LinearActuatorFunction_e::LA_Brake;
                _throttleActuatorSetuplMsg->ResetOutputs = csPBMsg.resetoutputs();
                _throttleActuatorSetuplMsg->ResetHardwareCfgs = csPBMsg.resethardwarecfgs();
                _throttleActuatorSetuplMsg->ResetUserCfgs = csPBMsg.resetusercfgs();
                _throttleActuatorSetuplMsg->ResetAll = csPBMsg.resetall();
                _throttleActuatorSetuplMsg->AutoZeroCal = csPBMsg.autozerocal();
                _throttleActuatorSetuplMsg->SetCanCommandResponsIDs = csPBMsg.setcancommandresponsids();
                _throttleActuatorSetuplMsg->PostMessage();
                ctrlMsgSet = true;
            }
            catch (std::exception &e)
            {
                LOGERROR("BrakeControlMsgHandler Exception: " << e.what());
            }
        }
        if (ctrlMsgSet)
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
            respMsg->set_cmdresponsemessage("[HANDLER] OK");
        } else
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
            respMsg->set_cmdresponsemessage("[HANDLER] Could Not read Throttle Setup Message");
        }
    }


    void CommandResponseMessageHandler::SetThrottleConfigParamsMsgHandler(VisionCommandPBMsg *cmdMsg,
                                                                       VisionResponsePBMsg *respMsg)
    {
        vision_messages::KarTechLinearActuatorParamsPBMsg csPBMsg;
        bool ctrlMsgSet = false;
        if (cmdMsg->has_cmddata())
        {
            try
            {
                csPBMsg.ParseFromString(cmdMsg->cmddata());
                _throttleActuatorParamsControlMsg->FunctionType = LinearActuatorFunction_e::LA_Accelerator;
                _throttleActuatorParamsControlMsg->setMinPositionInches(csPBMsg.minpositioninches());
                _throttleActuatorParamsControlMsg->setMaxPositionInches(csPBMsg.maxpositioninches());
                _throttleActuatorParamsControlMsg->setFeedbackCtrl_KP(csPBMsg.feedbackctrl_kp());
                _throttleActuatorParamsControlMsg->setFeedbackCtrl_KI(csPBMsg.feedbackctrl_ki());
                _throttleActuatorParamsControlMsg->setFeedbackCtrl_KD(csPBMsg.feedbackctrl_kd());
                _throttleActuatorParamsControlMsg->setFeedbackCtrl_CLFreq(csPBMsg.feedbackctrl_clfreq());
                _throttleActuatorParamsControlMsg->setMotor_MinPWM(csPBMsg.motor_minpwm());
                _throttleActuatorParamsControlMsg->setMotor_MaxPWM(csPBMsg.motor_maxpwm());
                _throttleActuatorParamsControlMsg->setMotor_pwmFreq(csPBMsg.motor_pwmfreq());
                _throttleActuatorParamsControlMsg->setFeedbackCtrl_ErrDeadbandInces(csPBMsg.feedbackctrl_errdeadbandinces());
                _throttleActuatorParamsControlMsg->setPositionReachedErrorTimeMSec(csPBMsg.positionreachederrortimemsec());
                _throttleActuatorParamsControlMsg->setMotorMaxCurrentLimitAmps(csPBMsg.motormaxcurrentlimitamps());
                _throttleActuatorParamsControlMsg->PostMessage();
                ctrlMsgSet = true;
            }
            catch (std::exception &e)
            {
                LOGERROR("SetThrottleConfigParamsMsgHandler Exception: " << e.what());
            }
        }
        if (ctrlMsgSet)
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
            respMsg->set_cmdresponsemessage("[HANDLER] OK");
        } else
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
            respMsg->set_cmdresponsemessage("[HANDLER] Could Not read Throttle Parameters Control Message");
        }
    }

    void CommandResponseMessageHandler::GetThrottleConfigParamsMsgHandler(VisionCommandPBMsg *cmdMsg,
                                                                       VisionResponsePBMsg *respMsg)
    {
        vision_messages::KarTechLinearActuatorParamsPBMsg ktlaPramsPBMsg;
        try
        {
            _throttleActuatorParamsFeedbackMsg->FetchMessage();
            ktlaPramsPBMsg.set_minpositioninches(_throttleActuatorParamsFeedbackMsg->getMinPositionInches());
            ktlaPramsPBMsg.set_maxpositioninches(_throttleActuatorParamsFeedbackMsg->getMaxPositionInches());

            ktlaPramsPBMsg.set_feedbackctrl_kp(_throttleActuatorParamsFeedbackMsg->getFeedbackCtrl_KP());
            ktlaPramsPBMsg.set_feedbackctrl_ki(_throttleActuatorParamsFeedbackMsg->getFeedbackCtrl_KI());
            ktlaPramsPBMsg.set_feedbackctrl_kd(_throttleActuatorParamsFeedbackMsg->getFeedbackCtrl_KD());
            ktlaPramsPBMsg.set_feedbackctrl_clfreq(_throttleActuatorParamsFeedbackMsg->getFeedbackCtrl_CLFreq());

            ktlaPramsPBMsg.set_motor_minpwm(_throttleActuatorParamsFeedbackMsg->getMotor_MinPWM());
            ktlaPramsPBMsg.set_motor_maxpwm(_throttleActuatorParamsFeedbackMsg->getMotor_MaxPWM());
            ktlaPramsPBMsg.set_motor_pwmfreq(_throttleActuatorParamsFeedbackMsg->getMotor_pwmFreq());

            ktlaPramsPBMsg.set_feedbackctrl_errdeadbandinces(_throttleActuatorParamsFeedbackMsg->getFeedbackCtrl_ErrDeadbandInces());
            ktlaPramsPBMsg.set_positionreachederrortimemsec(_throttleActuatorParamsFeedbackMsg->getPositionReachedErrorTimeMSec());
            ktlaPramsPBMsg.set_motormaxcurrentlimitamps(_throttleActuatorParamsFeedbackMsg->getMotorMaxCurrentLimitAmps());

            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
            respMsg->set_cmdresponsemessage("[HANDLER] OK");
            respMsg->set_cmdresponsedata(ktlaPramsPBMsg.SerializeAsString());

            //set_TargetCornerLocations
        }
        catch (std::exception &e)
        {
            LOGERROR("GetBrakeConfigParamsMsgHandler Exception: " << e.what());
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
            respMsg->set_cmdresponsemessage("[HANDLER] Could Not Get Linear Actuator Params.");
        }
    }


    void CommandResponseMessageHandler::SteeringControlMsgHandler(VisionCommandPBMsg *cmdMsg,
                                                                    VisionResponsePBMsg *respMsg)
    {
        vision_messages::SteeringTorqueCtrlPBMsg csPBMsg;
        bool ctrlMsgSet = false;
        if (cmdMsg->has_cmddata())
        {
            try
            {
                csPBMsg.ParseFromString(cmdMsg->cmddata());
                _steeringCtrlMsg->SteeringControlEnabled = csPBMsg.steeringcontrolenabled();
                _steeringCtrlMsg->ManualExtControl = csPBMsg.manualextcontrol();
                _steeringCtrlMsg->setSteeringTorquePercent(csPBMsg.steeringtorquepercent());
                _steeringCtrlMsg->setSteeringTorqueMap(csPBMsg.steeringtorquemap());
                _steeringCtrlMsg->PostMessage();
                ctrlMsgSet = true;
            }
            catch (std::exception &e)
            {
                LOGERROR("SteeringControlMsgHandler Exception: " << e.what());
            }
        }
        if (ctrlMsgSet)
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
            respMsg->set_cmdresponsemessage("[HANDLER] OK");
        } else
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
            respMsg->set_cmdresponsemessage("[HANDLER] Could Not read Steering Control Message");
        }
    }


    void CommandResponseMessageHandler::IMUCommandMessageHandler(VisionCommandPBMsg *cmdMsg,
                                                                  VisionResponsePBMsg *respMsg)
    {
        vision_messages::IMUCommandResponsePBMsg csPBMsg;
        bool ctrlMsgSet = false;
        if (cmdMsg->has_cmddata())
        {
            try
            {
                csPBMsg.ParseFromString(cmdMsg->cmddata());
                _imuCommandMessage->IMURemoteCtrlEnable = csPBMsg.imuremotectrlenable();
                _imuCommandMessage->CmdRspMsg = csPBMsg.cmdrspmsg();
                _imuCommandMessage->PostMessage();
                ctrlMsgSet = true;
            }
            catch (std::exception &e)
            {
                LOGERROR("IMUCommandMessageHandler Exception: " << e.what());
            }
        }
        if (ctrlMsgSet)
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
            respMsg->set_cmdresponsemessage("[HANDLER] OK");
        } else
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
            respMsg->set_cmdresponsemessage("[HANDLER] Could Not read IMU Command Message");
        }
    }

    void CommandResponseMessageHandler::HeadOrientationCommandHandler(VisionCommandPBMsg *cmdMsg,
                                                                 VisionResponsePBMsg *respMsg)
    {
        vision_messages::HeadOrientationControlPBMsg csPBMsg;
        bool ctrlMsgSet = false;
        if (cmdMsg->has_cmddata())
        {
            try
            {
                csPBMsg.ParseFromString(cmdMsg->cmddata());
                _headOrientationControlMsg->HeadOrientationOutputSelect = (HeadOrientationOutputSelect_e )csPBMsg.headorientationoutputselect();
                _headOrientationControlMsg->DisableHeadOrientationKalmanFilter = csPBMsg.disableheadorientationkalmanfilter();
                _headOrientationControlMsg->DisableVehicleInputToHeadOrientation = csPBMsg.disablevehicleinputtoheadorientation();
                _headOrientationControlMsg->DisableVehicleGravityFeedback = csPBMsg.disablevehiclegravityfeedback();
                _headOrientationControlMsg->SetHeadOrientation_QVar(csPBMsg.headorientation_qvar());
                _headOrientationControlMsg->SetHeadOrientation_RVar(csPBMsg.headorientation_rvar());
                _headOrientationControlMsg->SetVehicleGravityFeedbackGain(csPBMsg.vehiclegravityfeedbackgain());
                _headOrientationControlMsg->PostMessage();
                ctrlMsgSet = true;
            }
            catch (std::exception &e)
            {
                LOGERROR("HeadOrientationCommandHandler Exception: " << e.what());
            }
        }
        if (ctrlMsgSet)
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
            respMsg->set_cmdresponsemessage("[HANDLER] OK");
        } else
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
            respMsg->set_cmdresponsemessage("[HANDLER] Could Not read HeadOrientationCommand Message");
        }
    }


    void CommandResponseMessageHandler::VehicleControlParametersHandler(VisionCommandPBMsg *cmdMsg,
                                                                      VisionResponsePBMsg *respMsg)
    {
        vision_messages::VehicleControlParametersPBMsg csPBMsg;
        bool ctrlMsgSet = false;
        if (cmdMsg->has_cmddata())
        {
            try
            {
                csPBMsg.ParseFromString(cmdMsg->cmddata());
                _vehicleControlParametersMsg->SipnPuffBlowGain = csPBMsg.sipnpuffblowgain();
                _vehicleControlParametersMsg->SipnPuffSuckGain = csPBMsg.sipnpuffsuckgain();
                _vehicleControlParametersMsg->SipnPuffDeadBandPercent = csPBMsg.sipnpuffdeadbandpercent();
                _vehicleControlParametersMsg->ReverseSipnPuffThrottleBrake = csPBMsg.reversesipnpuffthrottlebrake();
                _vehicleControlParametersMsg->ThrottleSipnPuffGain = csPBMsg.throttlesipnpuffgain();
                _vehicleControlParametersMsg->BrakeSipnPuffGain = csPBMsg.brakesipnpuffgain();

                _vehicleControlParametersMsg->ThrottleBrakeHeadTiltEnable = csPBMsg.throttlebrakeheadtiltenable();
                _vehicleControlParametersMsg->ThrottleBrakeHeadTiltForwardDeadbandDegrees = csPBMsg.throttlebrakeheadtiltforwarddeadbanddegrees();
                _vehicleControlParametersMsg->ThrottleBrakeHeadTiltBackDeadbandDegrees = csPBMsg.throttlebrakeheadtiltbackdeadbanddegrees();
                _vehicleControlParametersMsg->ThrottleHeadTiltGain = csPBMsg.throttleheadtiltgain();
                _vehicleControlParametersMsg->BrakeHeadTiltGain = csPBMsg.brakeheadtiltgain();

                _vehicleControlParametersMsg->UseSteeringAngleControl = csPBMsg.usesteeringanglecontrol();
                _vehicleControlParametersMsg->SteeringDeadband = csPBMsg.steeringdeadband();
                _vehicleControlParametersMsg->SteeringControlGain = csPBMsg.steeringcontrolgain();
                _vehicleControlParametersMsg->SteeringBiasAngleDegrees = csPBMsg.steeringbiasangledegrees();
                _vehicleControlParametersMsg->RCSteeringGain = csPBMsg.rcsteeringgain();

                _vehicleControlParametersMsg->MaxLRHeadRotationDegrees = csPBMsg.maxlrheadrotationdegrees();

                _vehicleControlParametersMsg->HeadLeftRighLPFOrder = csPBMsg.headleftrighlpforder();
                _vehicleControlParametersMsg->HeadLeftRighLPFCutoffFreqHz = csPBMsg.headleftrighlpfcutofffreqhz();

                _vehicleControlParametersMsg->SteeringAngleFeedback_Kp = csPBMsg.steeringanglefeedback_kp();
                _vehicleControlParametersMsg->SteeringAngleFeedback_Kd = csPBMsg.steeringanglefeedback_kd();
                _vehicleControlParametersMsg->SteeringAngleFeedback_Ki = csPBMsg.steeringanglefeedback_ki();

                _vehicleControlParametersMsg->PostMessage();
                ctrlMsgSet = true;

                CommsManager *cmPtr = (CommsManager*)_mgrPtr;
                cmPtr->WorkSpace()->BCIControlConfigMsg->FetchMessage();
                cmPtr->WorkSpace()->BCIControlConfigMsg->BCIThrottleIntegrationGain = csPBMsg.bcigain();
                cmPtr->WorkSpace()->BCIControlConfigMsg->PostMessage();

            }
            catch (std::exception &e)
            {
                LOGERROR("HeadOrientationCommandHandler Exception: " << e.what());
            }
        }
        if (ctrlMsgSet)
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
            respMsg->set_cmdresponsemessage("[HANDLER] OK");
        } else
        {
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
            respMsg->set_cmdresponsemessage("[HANDLER] Could Not read HeadOrientationCommand Message");
        }
    }

    void CommandResponseMessageHandler::GetVehicleControlParametersHandler(VisionCommandPBMsg *cmdMsg,
                                                                        VisionResponsePBMsg *respMsg)
    {
        vision_messages::VehicleControlParametersPBMsg csPBMsg;
        try
        {
            CommsManager *cmPtr = (CommsManager*)_mgrPtr;
            cmPtr->WorkSpace()->BCIControlConfigMsg->FetchMessage();            
            _vehicleControlParametersMsg->FetchMessage();
            
            csPBMsg.set_sipnpuffblowgain(_vehicleControlParametersMsg->SipnPuffBlowGain);
            csPBMsg.set_sipnpuffsuckgain(_vehicleControlParametersMsg->SipnPuffSuckGain);
            csPBMsg.set_sipnpuffdeadbandpercent(_vehicleControlParametersMsg->SipnPuffDeadBandPercent);
            csPBMsg.set_bcigain(cmPtr->WorkSpace()->BCIControlConfigMsg->BCIThrottleIntegrationGain);

            csPBMsg.set_reversesipnpuffthrottlebrake(_vehicleControlParametersMsg->ReverseSipnPuffThrottleBrake);
            csPBMsg.set_throttlesipnpuffgain(_vehicleControlParametersMsg->ThrottleSipnPuffGain);
            csPBMsg.set_brakesipnpuffgain(_vehicleControlParametersMsg->BrakeSipnPuffGain);

            csPBMsg.set_throttlebrakeheadtiltenable(_vehicleControlParametersMsg->ThrottleBrakeHeadTiltEnable);
            csPBMsg.set_throttlebrakeheadtiltforwarddeadbanddegrees(_vehicleControlParametersMsg->ThrottleBrakeHeadTiltForwardDeadbandDegrees);
            csPBMsg.set_throttlebrakeheadtiltbackdeadbanddegrees(_vehicleControlParametersMsg->ThrottleBrakeHeadTiltBackDeadbandDegrees);
            csPBMsg.set_throttleheadtiltgain(_vehicleControlParametersMsg->ThrottleHeadTiltGain);
            csPBMsg.set_brakeheadtiltgain(_vehicleControlParametersMsg->BrakeHeadTiltGain);

            csPBMsg.set_usesteeringanglecontrol(_vehicleControlParametersMsg->UseSteeringAngleControl);
            csPBMsg.set_steeringdeadband(_vehicleControlParametersMsg->SteeringDeadband);
            csPBMsg.set_steeringcontrolgain(_vehicleControlParametersMsg->SteeringControlGain);
            csPBMsg.set_steeringbiasangledegrees(_vehicleControlParametersMsg->SteeringBiasAngleDegrees);
            csPBMsg.set_rcsteeringgain(_vehicleControlParametersMsg->RCSteeringGain);

            csPBMsg.set_maxlrheadrotationdegrees(_vehicleControlParametersMsg->MaxLRHeadRotationDegrees);
            csPBMsg.set_headleftrighlpforder(_vehicleControlParametersMsg->HeadLeftRighLPFOrder);
            csPBMsg.set_headleftrighlpfcutofffreqhz(_vehicleControlParametersMsg->HeadLeftRighLPFCutoffFreqHz);

            csPBMsg.set_steeringanglefeedback_kp(_vehicleControlParametersMsg->SteeringAngleFeedback_Kp);
            csPBMsg.set_steeringanglefeedback_kd(_vehicleControlParametersMsg->SteeringAngleFeedback_Kd);
            csPBMsg.set_steeringanglefeedback_ki(_vehicleControlParametersMsg->SteeringAngleFeedback_Ki);


            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kOK);
            respMsg->set_cmdresponsemessage("[HANDLER] OK");
            respMsg->set_cmdresponsedata(csPBMsg.SerializeAsString());

            //set_TargetCornerLocations
        }
        catch (std::exception &e)
        {
            LOGERROR("GetVehicleControlParametersHandler Exception: " << e.what());
            respMsg->set_cmdresponsetype(VisionResponsePBMsg_ResponseType_e_kError);
            respMsg->set_cmdresponsemessage("[HANDLER] Could Not Get Vehicle Control Params.");
        }
    }


}