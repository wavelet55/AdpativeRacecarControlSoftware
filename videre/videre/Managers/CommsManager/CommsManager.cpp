/* ****************************************************************
 * Athr(s): Harry Direen PhD, Randy Direen Phd.
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: May. 2016
 *
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
 *******************************************************************/

#include "CommsManager.h"
#include "CompressedImageMessage.h"
#include "ImageProcTargetInfoResultsMessage.h"
#include "AccelerometerGyroMessage.h"
#include "GPSFixMessage.h"
#include "MsgWrapper.h"

namespace videre
{

    CommsManager::CommsManager(std::string name, std::shared_ptr<ConfigData> config)
    : CommsManagerWSRMgr(name)
    {
        this->SetWakeupTimeDelayMSec(30);
        _config_sptr = config;

        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);
        //Messages

        std::string msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_VidereSystemControlManager, "ManagerStatusMessage");
        _videreSystemControMgrStatsMsg = std::make_shared<Rabit::ManagerStatusMessage>();
        this->AddPublishSubscribeMessage(msgName, _videreSystemControMgrStatsMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_HeadOrientationManager, "ManagerStatusMessage");
        _headOrientationMgrStatsMsg = std::make_shared<Rabit::ManagerStatusMessage>();
        this->AddPublishSubscribeMessage(msgName, _headOrientationMgrStatsMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_VehicleStateManager, "ManagerStatusMessage");
        _vehicleStateMgrStatsMsg = std::make_shared<Rabit::ManagerStatusMessage>();
        this->AddPublishSubscribeMessage(msgName, _vehicleStateMgrStatsMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_CommsManager, "ManagerStatusMessage");
        _commMgrStatsMsg = std::make_shared<Rabit::ManagerStatusMessage>();
        this->AddPublishSubscribeMessage(msgName, _commMgrStatsMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_ImageCaptureManager, "ManagerStatusMessage");
        _ImageCaptureMgrStatsMsg = std::make_shared<Rabit::ManagerStatusMessage>();
        this->AddPublishSubscribeMessage(msgName, _ImageCaptureMgrStatsMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_VisionProcessManager, "ManagerStatusMessage");
        _VisionProcessMgrStatsMsg = std::make_shared<Rabit::ManagerStatusMessage>();
        this->AddPublishSubscribeMessage(msgName, _VisionProcessMgrStatsMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_StreamRecordManager, "ManagerStatusMessage");
        _StreamRecordMgrStatsMsg = std::make_shared<Rabit::ManagerStatusMessage>();
        this->AddPublishSubscribeMessage(msgName, _StreamRecordMgrStatsMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_SystemInfoManager, "ManagerStatusMessage");
        _SystemInfoMgrStatsMsg = std::make_shared<Rabit::ManagerStatusMessage>();
        this->AddPublishSubscribeMessage(msgName, _SystemInfoMgrStatsMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_GPSManager, "ManagerStatusMessage");
        _GpsMgrStatsMsg = std::make_shared<Rabit::ManagerStatusMessage>();
        this->AddPublishSubscribeMessage(msgName, _GpsMgrStatsMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_IMUCommManager, "ManagerStatusMessage");
        _IMUCommMgrStatsMsg = std::make_shared<Rabit::ManagerStatusMessage>();
        this->AddPublishSubscribeMessage(msgName, _IMUCommMgrStatsMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_SipnPuffManager, "ManagerStatusMessage");
        _SipnPuffMgrStatsMsg = std::make_shared<Rabit::ManagerStatusMessage>();
        this->AddPublishSubscribeMessage(msgName, _SipnPuffMgrStatsMsg);


        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_VehicleActuatorInterfaceManager, "ManagerStatusMessage");
        _VehicalActuatorInterfaceMgrStatsMsg = std::make_shared<Rabit::ManagerStatusMessage>();
        this->AddPublishSubscribeMessage(msgName, _VehicalActuatorInterfaceMgrStatsMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_RobotArmManager, "ManagerStatusMessage");
        _RobotArmMgrStatsMsg = std::make_shared<Rabit::ManagerStatusMessage>();
        this->AddPublishSubscribeMessage(msgName, _RobotArmMgrStatsMsg);

        msgName = MakeManagerPlusStringName(VisionSystemManagers_e::VS_RemoteControlManager, "ManagerStatusMessage");
        _RemoteControlMgrStatsMsg = std::make_shared<Rabit::ManagerStatusMessage>();
        this->AddPublishSubscribeMessage(msgName, _RemoteControlMgrStatsMsg);

        _videreSystemCtrlStatusMsg = std::make_shared<VidereSystemControlMessage>();
        AddPublishSubscribeMessage("VidereSystemCtrlStatusMsg", _videreSystemCtrlStatusMsg);

        _cameraCalStatusMsg = std::make_shared<CameraCalStatusMessage>();
        this->AddPublishSubscribeMessage("CameraCalStatusMessage", _cameraCalStatusMsg);

        _featureMatchProcStatusMsg = std::make_shared<FeatureMatchProcStatusMessage>();
        this->AddPublishSubscribeMessage("FeatureMatchProcStatusMessage", _featureMatchProcStatusMsg);

        _imageProcessControlStatusMsg = std::make_shared<ImageProcessControlMessage>();
        this->AddPublishSubscribeMessage("ImageProcessControlStatusMessage", _imageProcessControlStatusMsg);

        _steeringStatusMsg = std::make_shared<DceEPASteeringStatusMessage>();
        this->AddPublishSubscribeMessage("SteeringStatusMsg", _steeringStatusMsg);

        _throttlePositionFeedbackMsg = std::make_shared<LinearActuatorPositionCtrlMessage>();
        this->AddPublishSubscribeMessage("ThrottleLAPositionFeedbackMsg", _throttlePositionFeedbackMsg);

        _brakePositionFeedbackMsg = std::make_shared<LinearActuatorPositionCtrlMessage>();
        this->AddPublishSubscribeMessage("BrakeLAPositionFeedbackMsg", _brakePositionFeedbackMsg);

        _imuHeadOrientationMsg = std::make_shared<HeadOrientationMessage>();
        this->AddPublishSubscribeMessage("IMUHeadOrientationMsg", _imuHeadOrientationMsg);

        _processedHeadOrientationMsg = std::make_shared<HeadOrientationMessage>();
        this->AddPublishSubscribeMessage("ProcessedHeadOrientationMsg", _processedHeadOrientationMsg);

        _sipAndPuffStatusMsg = std::make_shared<SipnPuffMessage>();
        this->AddPublishSubscribeMessage("SipnPuffMessage", _sipAndPuffStatusMsg);

        _imuResponseMsg = std::make_shared<IMUCommandResponseMessage>();
        this->AddPublishSubscribeMessage("IMUResponseMessage", _imuResponseMsg);

        _trackHeadOrientationMsg = make_shared<TrackHeadOrientationMessage>();
        this->AddPublishSubscribeMessage("TrackHeadOrientationMessage", _trackHeadOrientationMsg);

        _vehicleSwitchInputMsg = make_shared<VehicleSwitchInputMessage>();
        this->AddPublishSubscribeMessage("VehicleSwitchInputMsg", _vehicleSwitchInputMsg);

        //Queues
        _zmqPublishOutMsgQue = std::make_shared<RabitMsgQueue>(250, "ZMQPublishOutMsgQueue");
        AddManagerMessageQueue(_zmqPublishOutMsgQue->GetMessageQueueName(),
                                _zmqPublishOutMsgQue);
        WakeUpManagerOnEnqueue(_zmqPublishOutMsgQue);

        _cmdResponseMsgHandler = CommandResponseMessageHandler();
        _telemetryMessageProcessor = TelemetryMessageProcessor();

    }

/**
 * @brief Initialize
 *
 * @throws exception if it can't get path to config data.
 */
    void CommsManager::Initialize()
    {

        LOGINFO("CommsManager: Initialization Started")

        auto host1 = _config_sptr->GetConfigStringValue("zeromq.host_reply", "tcp://*:5555");
        auto host2 = _config_sptr->GetConfigStringValue("zeromq.host_pub_results", "tcp://*:5556");
        auto host3 = _config_sptr->GetConfigStringValue("zeromq.host_pub_video", "tcp://*:5557");
        auto host4 = _config_sptr->GetConfigStringValue("zeromq.host_sub_telemetry", "tcp://*:5558");
        auto host5 = _config_sptr->GetConfigStringValue("zeromq.host_pub_monitor", "tcp://*:5559");
        auto bci_pub_port = _config_sptr->GetConfigStringValue("zeromq.host_pub_nexus_bci", "tcp://*:5560");
        auto bci_sub_port = _config_sptr->GetConfigStringValue("zeromq.host_sub_nexus_bci", "tcp://*:5561");

        try
        {
            _zmqComm.Initialize(host1, host2, host3, host4, host5, bci_pub_port, bci_sub_port);
        }
        catch (std::exception &e)
        {
            LOGERROR("Error Initializing ZeroMQ: Exception: " << e.what());
            std::cout << "Error Initializing ZeroMQ: Exception: " << e.what() << std::endl;
            std::cout << "Verify zeromq connection parameters in the VidereConfig.ini file." << std::endl;
            throw e;
        }

        //Initialize Message Queues
        try
        {
            //The Image/Vision Processing Manager Queue
            _imageStreamMsgQueue_sptr = this->GetManagerMessageQueue<std::shared_ptr<RabitMsgPtrSPSCQueue>>
                                             ("ImageStreamMsgQueue");
            _imageStreamEmptyMsgQueue_sptr = this->GetManagerMessageQueue<std::shared_ptr<RabitMsgPtrSPSCQueue>>
                                             ("ImageStreamEmptyMsgQueue");
            //Set event that will wakeup the loop when something is enqueued
            this->WakeUpManagerOnEnqueue(_imageStreamMsgQueue_sptr);

        }
        catch (Rabit::MessageNotRegisteredException &e)
        {
            LOGWARN("CommsManager: A Image Stream Message Queue does not exist.");
            std::cout << "CommsManager: A Image Stream Message Queue does not exist." << std::endl;
            _imageStreamMsgQueue_sptr = nullptr;
            _imageStreamEmptyMsgQueue_sptr = nullptr;
        }

        try
        {
            _TgtMsgEmptyQueue_sptr = this->GetManagerMessageQueue<std::shared_ptr<RabitMsgPtrSPSCQueue>>
                    ("ImageProcTgtMsgEmptyQueue");
            _TgtMsgsToBeSentQueue_sptr = this->GetManagerMessageQueue<std::shared_ptr<RabitMsgPtrSPSCQueue>>
                    ("ImageProcTgtMsgQueue");
            //Set event that will wakeup the loop when something is enqueued
            this->WakeUpManagerOnEnqueue(_TgtMsgsToBeSentQueue_sptr);

        }
        catch (Rabit::MessageNotRegisteredException &e)
        {
            LOGWARN("CommsManager: A Target Message Queue does not exist.");
            std::cout << "CommsManager: A Target Message Queue does not exist." << std::endl;
             _TgtMsgEmptyQueue_sptr = nullptr;
            _TgtMsgsToBeSentQueue_sptr = nullptr;
        }

        //Protobuf Messages
        _visionCmdMsg = vision_messages::VisionCommandPBMsg();
        _visionCmdResponseMsg = vision_messages::VisionResponsePBMsg();
        _broadcastPBMsg = vision_messages::BroadcastPBMsg();

        _cmdResponseMsgHandler.Intialize(this, &_zmqComm);

        _telemetryMessageProcessor.Intialize(this, &_zmqComm);

        LOGINFO("CommsManager: Initialization Complete");
        std::cout << "CommsManager: Initialization Complete" << std::endl;
    }

    void CommsManager::Shutdown()
    {

        _cmdResponseMsgHandler.Shutdown();
        _telemetryMessageProcessor.Shutdown();
        _zmqComm.Close();
    }

    void CommsManager::ExecuteUnitOfWork()
    {
        //  Check for a command response messages and process the message.
        _cmdResponseMsgHandler.HandleMessageFromRemote();

        //  Get telemetry from HOPS (such as GPS info) and process it
        // for the current state of the vision system. If there is no new
        // telemetry information from HOPS, update the current state based
        // on prior information.
        _telemetryMessageProcessor.HandleMessagesFromRemote();

        //  Streams video back to earth.
        if (_imageStreamMsgQueue_sptr != nullptr && _imageStreamEmptyMsgQueue_sptr != nullptr)
        {
            Rabit::RabitMessage *rMsgPtr;
            CompressedImageMessage *cImgMsgPtr;
            while (_imageStreamMsgQueue_sptr->GetMessage(rMsgPtr))
            {
                try
                {
                    cImgMsgPtr = static_cast<CompressedImageMessage *>(rMsgPtr);
                    if (!CompressedImgageMsgToTxMsg(cImgMsgPtr, _msgWrapperPBMsg))
                    {
                        _zmqComm.PublishImageMessage(_msgWrapperPBMsg.SerializeAsString());
                    }
                }
                catch (std::exception &e)
                {
                    LOGERROR("Publish Video Stream Msg: Exception: " << e.what());
                }
                //Return the Empty Message back to the Stream Record Mgr.
                _imageStreamEmptyMsgQueue_sptr->AddMessage(rMsgPtr);
            }
        }

        //Send Target Informaition to HOPS
        if (_TgtMsgsToBeSentQueue_sptr != nullptr && _TgtMsgEmptyQueue_sptr != nullptr)
        {
            Rabit::RabitMessage *rMsgPtr;
            while (_TgtMsgsToBeSentQueue_sptr->GetMessage(rMsgPtr))
            {
                try
                {
                    if(rMsgPtr->GetTypeIndex() == typeid(ImageProcTargetInfoResultsMessage))
                    {
                        ImageProcTargetInfoResultsMessage *tgtMsgPtr = static_cast<ImageProcTargetInfoResultsMessage *>(rMsgPtr);
                        if (!TargetInfoMsgToTxMsg(tgtMsgPtr, _msgWrapperPBMsg))
                        {
                            _zmqComm.PublishVisionProcResults(_msgWrapperPBMsg.SerializeAsString());
                        }
                    }
                }
                catch (std::exception &e)
                {
                    LOGERROR("Transmit Target Msg: Exception: " << e.what());
                }
                //Return the Empty Message back to the Vision Process Manager
                _TgtMsgEmptyQueue_sptr->AddMessage(rMsgPtr);
            }
        }

        //Post any messages to the ZMQ Publish Socket that are in the incoming message queue
        //Limit the number of messages sent at one time to ensure this process does not dominate
        //the other send processes.
        for(int i = 0; i < 50; i++)
        {
            shared_ptr<Rabit::RabitMessage> msg;
            if(_zmqPublishOutMsgQue->GetMessage(msg))
            {
                try
                {
                    bool sendMsg = false;
                    if(msg->GetTypeIndex() == typeid(AccelerometerGyroMessage) )
                    {
                        AccelerometerGyroMessage *agmsg = static_cast<AccelerometerGyroMessage *>(msg.get());
                        if( !IMUAccelGyroMsgToTxMsg(agmsg, _msgWrapperPBMsg) )
                        {
                            _zmqComm.PublishVisionProcResults(_msgWrapperPBMsg.SerializeAsString());
                        }
                    }
                    else if(msg->GetTypeIndex() == typeid(TrackHeadOrientationMessage))
                    {
                        TrackHeadOrientationMessage *htmsg = static_cast<TrackHeadOrientationMessage *>(msg.get());
                        if( ! TrackHeadOrientationMsgToTxMsg(htmsg, _msgWrapperPBMsg))
                        {
                            _zmqComm.PublishVisionProcResults(_msgWrapperPBMsg.SerializeAsString());
                        }
                    }
                    else if(msg->GetTypeIndex() == typeid(GPSFixMessage))
                    {
                        GPSFixMessage *gpsmsg = static_cast<GPSFixMessage *>(msg.get());
                        if( ! GPSFixeMsgToTxMsg(gpsmsg, _msgWrapperPBMsg))
                        {
                            _zmqComm.PublishVisionProcResults(_msgWrapperPBMsg.SerializeAsString());
                        }
                    }

                 }
                catch (std::exception &e)
                {
                    LOGERROR("Transmitting ZMQ Publish Msg: Exception: " << e.what());
                }
            }
        }

        //Check for updated Manager Status Info... send out the
        //Monitor stream if updated.
        if( _videreSystemControMgrStatsMsg->FetchMessage())
        {
            TransmitMgrStatsMsg(_videreSystemControMgrStatsMsg);
        }
        if( _headOrientationMgrStatsMsg->FetchMessage())
        {
            TransmitMgrStatsMsg(_headOrientationMgrStatsMsg);
        }
        if( _vehicleStateMgrStatsMsg->FetchMessage())
        {
            TransmitMgrStatsMsg(_vehicleStateMgrStatsMsg);
        }
        if( _ImageCaptureMgrStatsMsg->FetchMessage())
        {
            TransmitMgrStatsMsg(_ImageCaptureMgrStatsMsg);
        }
        if( _VisionProcessMgrStatsMsg->FetchMessage())
        {
            TransmitMgrStatsMsg(_VisionProcessMgrStatsMsg);
        }
        if( _StreamRecordMgrStatsMsg->FetchMessage())
        {
            TransmitMgrStatsMsg(_StreamRecordMgrStatsMsg);
        }
        if( _commMgrStatsMsg->FetchMessage())
        {
            TransmitMgrStatsMsg(_commMgrStatsMsg);
        }
        if( _GpsMgrStatsMsg->FetchMessage())
        {
            TransmitMgrStatsMsg(_GpsMgrStatsMsg);
        }
        if( _IMUCommMgrStatsMsg->FetchMessage())
        {
            TransmitMgrStatsMsg(_IMUCommMgrStatsMsg);
        }
        if( _SipnPuffMgrStatsMsg->FetchMessage())
        {
            TransmitMgrStatsMsg(_SipnPuffMgrStatsMsg);
        }
        if( _VehicalActuatorInterfaceMgrStatsMsg->FetchMessage())
        {
            TransmitMgrStatsMsg(_VehicalActuatorInterfaceMgrStatsMsg);
        }
        if( _RemoteControlMgrStatsMsg->FetchMessage())
        {
            TransmitMgrStatsMsg(_RemoteControlMgrStatsMsg);
        }
        if( _RobotArmMgrStatsMsg->FetchMessage())
        {
            TransmitMgrStatsMsg(_RobotArmMgrStatsMsg);
        }

        if( _cameraCalStatusMsg->FetchMessage())
        {
            TransmitCameraCalStatsMsg(_cameraCalStatusMsg);
        }
        if(_featureMatchProcStatusMsg->FetchMessage())
        {
            TransmitFeatureMatchProcStatsMsg(_featureMatchProcStatusMsg);
        }
        if( _imageProcessControlStatusMsg->FetchMessage())
        {
            TransmitImageProcessControlStatusMsg(_imageProcessControlStatusMsg);
        }

        if( _steeringStatusMsg->FetchMessage())
        {
            TransmitSteeringStatusMsg(_steeringStatusMsg);
        }
        if( _throttlePositionFeedbackMsg->FetchMessage())
        {
            _throttlePositionFeedbackMsg->FunctionType = LinearActuatorFunction_e::LA_Accelerator;
            TransmitLinearActuatorPositionStatusMsg(_throttlePositionFeedbackMsg);
        }
        if( _brakePositionFeedbackMsg->FetchMessage())
        {
            _brakePositionFeedbackMsg->FunctionType = LinearActuatorFunction_e::LA_Brake;
            TransmitLinearActuatorPositionStatusMsg(_brakePositionFeedbackMsg);
        }
        if( _processedHeadOrientationMsg->FetchMessage())
        {
            TransmitHeadOrientationMsg(_processedHeadOrientationMsg);
        }
        if( _sipAndPuffStatusMsg->FetchMessage())
        {
            TransmitSipAndPuffMsg(_sipAndPuffStatusMsg);
        }
        if( _imuResponseMsg->FetchMessage() )
        {
            TransmitIMUResponseMsg(_imuResponseMsg);
        }
        if( _trackHeadOrientationMsg->FetchMessage())
        {
            TransmitTrackHeadOrientationMsg(_trackHeadOrientationMsg);
        }
        if( _videreSystemCtrlStatusMsg->FetchMessage())
        {
            TransmitVidereSystemCtrlStatusMsg(_videreSystemCtrlStatusMsg);
        }

        //Nexus BCI Communications:
        if( WorkSpace()->SipnPuffBCIValueMsg->FetchMessage())
        {
            TransmitNexusBCIMsg(WorkSpace()->SipnPuffBCIValueMsg, "ThrottleBrakeValMsg");
        }

        if( WorkSpace()->SipnPuffConfigFeedbackMsg->FetchMessage())
        {
            TransmitNexusBCIMsg(WorkSpace()->SipnPuffConfigFeedbackMsg, "SipnPuffConfigFeedbackMsg");
        }

        if( WorkSpace()->BCIControlConfigFeedbackMsg->FetchMessage())
        {
            TransmitNexusBCIMsg(WorkSpace()->BCIControlConfigFeedbackMsg, "BCIControlConfigFeedbackMsg");
        }

        //Receive messages from the Nexus BCI (Brain Computer Interface) and process
        //the messages.
        for(int i = 0; i < 10; i++)
        {
            zmq::message_t bciMsg;
            if(_zmqComm.ReceiveNexusBCIMessage(&bciMsg))
            {
                ProcessNexusBCIRxMsg(&bciMsg);
            }
        }



    }

    void CommsManager::TransmitNexusBCIMsg(std::shared_ptr<Rabit::RabitMessage> msg,
                                                std::string msgName)
    {
        MsgWrapper mw(_txMsgBuf, MAX_TX_MSG_SIZE);
        mw.MsgName = msgName;
        mw.MsgSerializationType = MsgSerializationType_e::MST_DtiByteArray;
        int msgSize = mw.Serialize(msg);
        if( msgSize > 0)
        {
            try
            {
                _zmqComm.PublishNexusBCI(_txMsgBuf, msgSize);
            }
            catch (std::exception &e)
            {
                LOGERROR("Transmit Nexus BCI Message Exception: " << e.what());
            }
        }
    }

/************************************************************************************
*                            Telemetry
************************************************************************************/


    bool CommsManager::CheckForTelemetryFromRemote()
    {
        static zmq::message_t sub;

        if (_zmqComm.SubscribedTelemetryFromHOPS(&sub))
        {
            return true;
        }

        return false;
    }


    //Transmit a Manager Status Message out the Publish Monitor Socket
    void CommsManager::TransmitMgrStatsMsg(std::shared_ptr<Rabit::ManagerStatusMessage> mgrStatsMsg)
    {
        if( !ManagerStatsMsgToTxMsg(mgrStatsMsg, _msgWrapperPBMsg) )
        {
            std::string msgPBData = _msgWrapperPBMsg.SerializeAsString();
            //Transmit message to Monitor Stream
            try
            {
                _zmqComm.PublishMonitor(msgPBData);
            }
            catch (std::exception &e)
            {
                LOGERROR("TransmitMgrStatsMsg Exception: " << e.what());
            }
        }
    }

    //Transmit a Steering Status out the Publish Monitor Socket
    void CommsManager::TransmitSteeringStatusMsg(std::shared_ptr<DceEPASteeringStatusMessage> steeringStatsMsg)
    {
        DceEPASteeringStatusPBMsg pbMsg;

        try
        {
            pbMsg.set_motorcurrentamps(steeringStatsMsg->MotorCurrentAmps);
            pbMsg.set_pwmdutycyclepercent(steeringStatsMsg->PWMDutyCyclePercent);
            pbMsg.set_motortorquepercent(steeringStatsMsg->MotorTorquePercent);
            pbMsg.set_supplyvoltage(steeringStatsMsg->SupplyVoltage);
            pbMsg.set_tempdegc(steeringStatsMsg->TempDegC);
            pbMsg.set_steeringangledeg(steeringStatsMsg->SteeringAngleDeg);
            pbMsg.set_steeringtorquemapsetting((int)steeringStatsMsg->SteeringTorqueMapSetting);
            pbMsg.set_switchposition((int)steeringStatsMsg->SwitchPosition);
            pbMsg.set_torquea(steeringStatsMsg->TorqueA);
            pbMsg.set_torqueb(steeringStatsMsg->TorqueB);
            pbMsg.set_errorcode(steeringStatsMsg->ErrorCode);
            pbMsg.set_statusflags(steeringStatsMsg->StatusFlags);
            pbMsg.set_limitflags(steeringStatsMsg->LimitFlags);
            pbMsg.set_manualextcontrol(steeringStatsMsg->ManualExtControl);

            _msgWrapperPBMsg.Clear();
            _msgWrapperPBMsg.set_msgname("SteeringStatusMsg");
            _msgWrapperPBMsg.set_msgqualifier("None");
            _msgWrapperPBMsg.set_msgdata(pbMsg.SerializeAsString());

            std::string msgPBData = _msgWrapperPBMsg.SerializeAsString();
            _zmqComm.PublishMonitor(msgPBData);
        }
        catch (std::exception &e)
        {
            LOGERROR("TransmitSteeringStatusMsg Exception: " << e.what());
        }
    }

    //Transmit a Linear Actuator Postion Status out the Publish Monitor Socket
    void CommsManager::TransmitLinearActuatorPositionStatusMsg(std::shared_ptr<LinearActuatorPositionCtrlMessage> laPosMsg)
    {
        LinearActuatorPositionCtrlPBMsg pbMsg;

        try
        {
            pbMsg.set_positionpercent(laPosMsg->getPositionPercent());
            pbMsg.set_motorcurrentamps(laPosMsg->getMotorCurrentAmps());
            pbMsg.set_tempdegc(laPosMsg->TempDegC);
            pbMsg.set_errorflags(laPosMsg->ErrorFlags);
            pbMsg.set_errorflags(laPosMsg->ErrorFlags);
            pbMsg.set_motorenable(laPosMsg->MotorEnable);
            pbMsg.set_manualextcontrol(laPosMsg->ManualExtControl);

            _msgWrapperPBMsg.Clear();
            if(laPosMsg->FunctionType == LinearActuatorFunction_e::LA_Accelerator)
                _msgWrapperPBMsg.set_msgname("ThrottlePositionStatusMsg");
            else if(laPosMsg->FunctionType == LinearActuatorFunction_e::LA_Brake)
                _msgWrapperPBMsg.set_msgname("BrakePositionStatusMsg");
            else
                _msgWrapperPBMsg.set_msgname("UnknowLAPositionStatusMsg");

            _msgWrapperPBMsg.set_msgqualifier("None");
            _msgWrapperPBMsg.set_msgdata(pbMsg.SerializeAsString());

            std::string msgPBData = _msgWrapperPBMsg.SerializeAsString();
            _zmqComm.PublishMonitor(msgPBData);
        }
        catch (std::exception &e)
        {
            LOGERROR("TransmitSteeringStatusMsg Exception: " << e.what());
        }
    }

    void CommsManager::TransmitHeadOrientationMsg(std::shared_ptr<HeadOrientationMessage> headOrientationMsg)
    {
        HeadOrientationPBMsg pbMsg;

        try
        {
            pbMsg.set_headrolldegrees(headOrientationMsg->HeadRollPitchYawAnlges.RollDegrees());
            pbMsg.set_headpitchdegrees(headOrientationMsg->HeadRollPitchYawAnlges.PitchDegrees());
            pbMsg.set_headyawdegrees(headOrientationMsg->HeadRollPitchYawAnlges.YawPlusMinusDegrees());
            pbMsg.set_covariancenorm(headOrientationMsg->CovarianceNorm);

            _msgWrapperPBMsg.Clear();
            _msgWrapperPBMsg.set_msgname("HeadOrientationMsg");
            _msgWrapperPBMsg.set_msgqualifier("None");
            _msgWrapperPBMsg.set_msgdata(pbMsg.SerializeAsString());

            std::string msgPBData = _msgWrapperPBMsg.SerializeAsString();
            _zmqComm.PublishMonitor(msgPBData);
        }
        catch (std::exception &e)
        {
            LOGERROR("TransmitHeadOrientationMsg Exception: " << e.what());
        }
    }

    void CommsManager::TransmitSipAndPuffMsg(std::shared_ptr<SipnPuffMessage> sipAndPuffStatusMsg)
    {
        SipAndPuffPBMsg pbMsg;

        try
        {
            pbMsg.set_sipandpuffpecent(sipAndPuffStatusMsg->SipnPuffPecent);
            pbMsg.set_sipandpuffintegralpercent(sipAndPuffStatusMsg->SipnPuffIntegralPercent);

            _msgWrapperPBMsg.Clear();
            _msgWrapperPBMsg.set_msgname("SipAndPuffStatusMsg");
            _msgWrapperPBMsg.set_msgqualifier("None");
            _msgWrapperPBMsg.set_msgdata(pbMsg.SerializeAsString());

            std::string msgPBData = _msgWrapperPBMsg.SerializeAsString();
            _zmqComm.PublishMonitor(msgPBData);
        }
        catch (std::exception &e)
        {
            LOGERROR("TransmitSipAndPuffMsg Exception: " << e.what());
        }
    }


    void CommsManager::TransmitIMUResponseMsg(std::shared_ptr<IMUCommandResponseMessage> imuResponseMsg)
    {
        IMUCommandResponsePBMsg pbMsg;

        try
        {
            pbMsg.set_imuremotectrlenable(imuResponseMsg->IMURemoteCtrlEnable);
            pbMsg.set_cmdrspmsg(imuResponseMsg->CmdRspMsg);

            _msgWrapperPBMsg.Clear();
            _msgWrapperPBMsg.set_msgname("IMUResponseMsg");
            _msgWrapperPBMsg.set_msgqualifier("None");
            _msgWrapperPBMsg.set_msgdata(pbMsg.SerializeAsString());

            std::string msgPBData = _msgWrapperPBMsg.SerializeAsString();
            _zmqComm.PublishMonitor(msgPBData);
        }
        catch (std::exception &e)
        {
            LOGERROR("TransmitIMUResponseMsg Exception: " << e.what());
        }
    }


    //Transmit a Camera Calibration Status Message out the Publish Monitor Socket
    void CommsManager::TransmitCameraCalStatsMsg(std::shared_ptr<CameraCalStatusMessage> cameraCalStatsMsg)
    {
        CameraCalStatusPBMsg ccStatsPBMsg;

        try
        {
            ccStatsPBMsg.set_cameracalstate((CameraCalStatusPBMsg::CameraCalState_e)cameraCalStatsMsg->CameraCalibrationState);
            ccStatsPBMsg.set_numberofimagescaptured(cameraCalStatsMsg->NumberOfImagesCaptured);
            ccStatsPBMsg.set_cameracalstatusmsg(cameraCalStatsMsg->CameraCalStatusMsg);
            ccStatsPBMsg.set_imageok(cameraCalStatsMsg->ImageOk);

            _msgWrapperPBMsg.Clear();
            _msgWrapperPBMsg.set_msgname("CameraCalStatsMsg");
            _msgWrapperPBMsg.set_msgqualifier("None");
            _msgWrapperPBMsg.set_msgdata(ccStatsPBMsg.SerializeAsString());

            std::string msgPBData = _msgWrapperPBMsg.SerializeAsString();
            _zmqComm.PublishMonitor(msgPBData);
        }
        catch (std::exception &e)
        {
            LOGERROR("TransmitMgrStatsMsg Exception: " << e.what());
        }
     }

    //Transmit a Camera Calibration Status Message out the Publish Monitor Socket
    void CommsManager::TransmitFeatureMatchProcStatsMsg(std::shared_ptr<FeatureMatchProcStatusMessage> fmpStatsMsg)
    {
        FeatureMatchProcStatusPBMsg fmpStatsPBMsg;

        try
        {
            fmpStatsPBMsg.set_featurematchingstate((int)fmpStatsMsg->FeatureMatchingState);
            fmpStatsPBMsg.set_featureextractiontyperoutine((int)fmpStatsMsg->FeatureExtractionTypeRoutine);
            fmpStatsPBMsg.set_featurematchtyperoutine((int)fmpStatsMsg->FeatureMatchTypeRoutine);
            fmpStatsPBMsg.set_statusmessage(fmpStatsMsg->StatusMessage);
            fmpStatsPBMsg.set_numberofimagescaptured(fmpStatsMsg->NumberOfImagesCaptured);
            fmpStatsPBMsg.set_processtimer_1(fmpStatsMsg->ProcessTimer_1);
            fmpStatsPBMsg.set_processtimer_2(fmpStatsMsg->ProcessTimer_2);

            fmpStatsPBMsg.set_statusvali_1(fmpStatsMsg->StatusValI_1);
            fmpStatsPBMsg.set_statusvali_2(fmpStatsMsg->StatusValI_2);
            fmpStatsPBMsg.set_statusvali_3(fmpStatsMsg->StatusValI_3);
            fmpStatsPBMsg.set_statusvali_4(fmpStatsMsg->StatusValI_4);
            fmpStatsPBMsg.set_statusvali_5(fmpStatsMsg->StatusValI_5);
            fmpStatsPBMsg.set_statusvali_6(fmpStatsMsg->StatusValI_6);
            fmpStatsPBMsg.set_statusvali_7(fmpStatsMsg->StatusValI_7);
            fmpStatsPBMsg.set_statusvali_8(fmpStatsMsg->StatusValI_8);
            fmpStatsPBMsg.set_statusvali_9(fmpStatsMsg->StatusValI_9);

            fmpStatsPBMsg.set_statusvalf_10(fmpStatsMsg->StatusValF_10);
            fmpStatsPBMsg.set_statusvalf_11(fmpStatsMsg->StatusValF_11);
            fmpStatsPBMsg.set_statusvalf_12(fmpStatsMsg->StatusValF_12);
            fmpStatsPBMsg.set_statusvalf_13(fmpStatsMsg->StatusValF_13);
            fmpStatsPBMsg.set_statusvalf_14(fmpStatsMsg->StatusValF_14);
            fmpStatsPBMsg.set_statusvalf_15(fmpStatsMsg->StatusValF_15);
            fmpStatsPBMsg.set_statusvalf_16(fmpStatsMsg->StatusValF_16);
            fmpStatsPBMsg.set_statusvalf_17(fmpStatsMsg->StatusValF_17);
            fmpStatsPBMsg.set_statusvalf_18(fmpStatsMsg->StatusValF_18);
            fmpStatsPBMsg.set_statusvalf_19(fmpStatsMsg->StatusValF_19);

            _msgWrapperPBMsg.Clear();
            _msgWrapperPBMsg.set_msgname("FeatureMatchProcStatsMsg");
            _msgWrapperPBMsg.set_msgqualifier("None");
            _msgWrapperPBMsg.set_msgdata(fmpStatsPBMsg.SerializeAsString());

            std::string msgPBData = _msgWrapperPBMsg.SerializeAsString();
            _zmqComm.PublishMonitor(msgPBData);
        }
        catch (std::exception &e)
        {
            LOGERROR("TransmitMgrStatsMsg Exception: " << e.what());
        }
    }


    //Transmit a Camera Calibration Status Message out the Publish Monitor Socket
    void CommsManager::TransmitImageProcessControlStatusMsg(std::shared_ptr<ImageProcessControlMessage> statsMsg)
    {
        vision_messages::VisionProcessingControlPBMsg visionCtrlPBMsg;

        try
        {
            visionCtrlPBMsg.set_targetimageprocessingenabled(statsMsg->TargetImageProcessingEnabled);
            visionCtrlPBMsg.set_gpsdeniedprocessingenabled(statsMsg->GPSDeniedProcessingEnabled);
            visionCtrlPBMsg.set_gpuprocessingenabled(statsMsg->GPUProcessingEnabled);
            visionCtrlPBMsg.set_visionprocessingmode((::vision_messages::VisionProcessingControlPBMsg_VisionProcessingModeEnum)statsMsg->VisionProcessingMode);
            visionCtrlPBMsg.set_targetprocessingmode((::vision_messages::VisionProcessingControlPBMsg_TargetProcessingModeEnum)statsMsg->TargetProcessingMode);
            visionCtrlPBMsg.set_gpsdeniedprocessingmode((::vision_messages::VisionProcessingControlPBMsg_GPSDeniedProcessingModeEnum)statsMsg->GPSDeniedProcessingMode);

            _msgWrapperPBMsg.Clear();
            _msgWrapperPBMsg.set_msgname("ImageProcessingStatsMsg");
            _msgWrapperPBMsg.set_msgqualifier("None");
            _msgWrapperPBMsg.set_msgdata(visionCtrlPBMsg.SerializeAsString());

            std::string msgPBData = _msgWrapperPBMsg.SerializeAsString();
            _zmqComm.PublishMonitor(msgPBData);
        }
        catch (std::exception &e)
        {
            LOGERROR("TransmitImageProcessControlStatusMsg Exception: " << e.what());
        }
    }

    void CommsManager::TransmitTrackHeadOrientationMsg(std::shared_ptr<TrackHeadOrientationMessage> thoMsg)
    {
        vision_messages::TrackHeadOrientationPBMsg thoPBMsg;

        try
        {
            thoPBMsg.set_headorientationquaternion_w(thoMsg->TrackHeadOrientationData.HeadOrientationQuaternion.qScale);
            thoPBMsg.set_headorientationquaternion_x(thoMsg->TrackHeadOrientationData.HeadOrientationQuaternion.qVec.x);
            thoPBMsg.set_headorientationquaternion_y(thoMsg->TrackHeadOrientationData.HeadOrientationQuaternion.qVec.y);
            thoPBMsg.set_headorientationquaternion_z(thoMsg->TrackHeadOrientationData.HeadOrientationQuaternion.qVec.z);

            thoPBMsg.set_headtranslationvec_x(thoMsg->TrackHeadOrientationData.HeadTranslationVec.x);
            thoPBMsg.set_headtranslationvec_y(thoMsg->TrackHeadOrientationData.HeadTranslationVec.y);
            thoPBMsg.set_headtranslationvec_z(thoMsg->TrackHeadOrientationData.HeadTranslationVec.z);
            thoPBMsg.set_isdatavalid(thoMsg->TrackHeadOrientationData.IsDataValid);

            _msgWrapperPBMsg.Clear();
            _msgWrapperPBMsg.set_msgname("TrackHeadOrientationMsg");
            _msgWrapperPBMsg.set_msgqualifier("None");
            _msgWrapperPBMsg.set_msgdata(thoPBMsg.SerializeAsString());

            std::string msgPBData = _msgWrapperPBMsg.SerializeAsString();
            _zmqComm.PublishMonitor(msgPBData);
        }
        catch (std::exception &e)
        {
            LOGERROR("TransmitImageProcessControlStatusMsg Exception: " << e.what());
        }
    }

    void CommsManager::TransmitVidereSystemCtrlStatusMsg(std::shared_ptr<VidereSystemControlMessage> thoMsg)
    {
        vision_messages::VidereSystemControlPBMsg thoPBMsg;
        _vehicleSwitchInputMsg->FetchMessage();

        try
        {
            thoPBMsg.set_systemstate((uint32_t)thoMsg->SystemState);
            thoPBMsg.set_startprocess(thoMsg->StartProcess);
            thoPBMsg.set_pauseproces(thoMsg->PauseProces);
            thoPBMsg.set_stopprocess(thoMsg->StopProcess);
            thoPBMsg.set_headcontrolenable(thoMsg->HeadControlEnable);
            thoPBMsg.set_throttlecontrolenable(thoMsg->ThrottleControlEnable);
            thoPBMsg.set_brakecontrolenable(thoMsg->BrakeControlEnable);
            thoPBMsg.set_systemstatus((uint32_t)thoMsg->SystemStatus);
            thoPBMsg.set_statuscounter(thoMsg->StatusCounter);
            thoPBMsg.set_driverenableswitch(_vehicleSwitchInputMsg->DriverControlEnabled);
            thoPBMsg.set_nexusbcicontrolenabled(thoMsg->BCIControlEnable);
            thoPBMsg.set_nexusbcithrottleenable(WorkSpace()->BCIThrottleControlMsg->ThrottleOn);

            _msgWrapperPBMsg.Clear();
            _msgWrapperPBMsg.set_msgname("VidereSystemStatusMsg");
            _msgWrapperPBMsg.set_msgqualifier("None");
            _msgWrapperPBMsg.set_msgdata(thoPBMsg.SerializeAsString());

            std::string msgPBData = _msgWrapperPBMsg.SerializeAsString();
            _zmqComm.PublishMonitor(msgPBData);
        }
        catch (std::exception &e)
        {
            LOGERROR("TransmitImageProcessControlStatusMsg Exception: " << e.what());
        }
    }


    void CommsManager::ProcessNexusBCIRxMsg(zmq::message_t *bciMsgPtr)
    {
        MsgWrapper msgWrapper;
        if(msgWrapper.DeSerializeHeader((uint8_t*)bciMsgPtr->data(), bciMsgPtr->size()) )
        {
            try
            {
                if( msgWrapper.MsgName == "BCIThrottleControlMsg" )
                {
                    WorkSpace()->BCIThrottleControlMsg->DeSerialize(msgWrapper.MsgData, msgWrapper.MsgDataSize);
                    WorkSpace()->BCIThrottleControlMsg->PostMessage();
                    LOGINFO("BCIThrottleControlMsg: " << WorkSpace()->BCIThrottleControlMsg->ToString());
                }
                else if( msgWrapper.MsgName == "SipnPuffConfigMsg" )
                {
                    WorkSpace()->SipnPuffConfigMsg->DeSerialize(msgWrapper.MsgData, msgWrapper.MsgDataSize);
                    WorkSpace()->SipnPuffConfigMsg->PostMessage();
                    LOGINFO("SipnPuffConfigMsg: " << WorkSpace()->SipnPuffConfigMsg->ToString());
                }
                else if( msgWrapper.MsgName == "BCIControlConfigMsg" )
                {
                    WorkSpace()->BCIControlConfigMsg->DeSerialize(msgWrapper.MsgData, msgWrapper.MsgDataSize);
                    WorkSpace()->BCIControlConfigMsg->PostMessage();
                    LOGINFO("BCIControlConfigMsg: " << WorkSpace()->BCIControlConfigMsg->ToString());
                }
            }
            catch (std::exception &e)
            {
                LOGERROR("ProcessNexusBCIRxMsg Exception: " << e.what());
            }
        }
        else
        {
            LOGERROR("ProcessNexusBCIRxMsg:Error Deserializeing a message.");
        }

    }

}
