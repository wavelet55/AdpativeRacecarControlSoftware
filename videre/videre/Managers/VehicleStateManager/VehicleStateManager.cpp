/* ****************************************************************
 * Athr(s): Randy Direen, PhD
 * Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: Aug, 2018
 *
 * Vehicle State Manager
 * Generates the best estimate of the vehicle state including
 * postion, velocity, and orientation.
 * Uses GPS, IMU, Car speed and Track Mapping information to generate the 
 * vehicle state.
 *******************************************************************/


#include "VehicleStateManager.h"
#include "HeadTrackingOrientationMessage.h"

using namespace std;

namespace videre
{

    VehicleStateManager::VehicleStateManager(std::string name,
                                     std::shared_ptr<ConfigData> config)
            : VehicleStateManagerWSRMgr(name), _dataRecorder(),
              _vehicleStateDataRecord(), _dataRecorderStdHeader("Vehicle State Data Log", 0)
    {
        SetWakeupTimeDelayMSec(1000);
        _config_sptr = config;

        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        //Queues
        VehicleStateIMUMsgRxQueue = std::make_shared<RabitMsgPtrSPSCQueue>(
                25, "VehicleStateIMUMsgRxQueue");
        AddManagerMessageQueue(VehicleStateIMUMsgRxQueue->GetMessageQueueName(),
                               VehicleStateIMUMsgRxQueue);
        WakeUpManagerOnEnqueue(VehicleStateIMUMsgRxQueue);

        VehicleStateIMUEmptyMsgQueue = std::make_shared<RabitMsgPtrSPSCQueue>(
                30, "VehicleStateIMUEmptyMsgQueue");
        AddManagerMessageQueue(VehicleStateIMUEmptyMsgQueue->GetMessageQueueName(),
                               VehicleStateIMUEmptyMsgQueue);

        GPSFixMsgPtr = std::make_shared<GPSFixMessage>();
        AddPublishSubscribeMessage("GPSFixMessage", GPSFixMsgPtr);

        _vehicleInertialStatesMsg = std::make_shared<VehicleInertialStatesMessage>();
        AddPublishSubscribeMessage("VehicleInertialStatesMessage", _vehicleInertialStatesMsg);

        //Messages
        _vehicleOrientationQuaternionMsg = make_shared<QuaternionMessage>();
        AddPublishSubscribeMessage("VehicleOrientationQuaternionMsg", _vehicleOrientationQuaternionMsg);

        _headOrientationControlMsg = make_shared<HeadOrientationControlMessage>();
        AddPublishSubscribeMessage("HeadOrientationControlMessage", _headOrientationControlMsg);

        _resetOrientationStateMsg = std::make_shared<ResetOrientationStateMessage>();
        AddPublishSubscribeMessage("ResetOrientationStateMessage", _resetOrientationStateMsg);

        _loggingControlMsg = std::make_shared<ImageLoggingControlMessage>();
        AddPublishSubscribeMessage("ImageLoggingControlMessage", _loggingControlMsg);

        _dataRecorder.setDirectory(DataLogDirectory);
        string fn = config->GetConfigStringValue("VehicleState.DataLogBaseFilename", "VehicleStateDataLog");
        _dataRecorder.setBaseFilename(fn);
        _dataRecorder.setHeaderRecord(&_dataRecorderStdHeader);

        EnableVehicleStatefLogging = config->GetConfigBoolValue("VehicleState.EnableLogging", true);
    }

    void VehicleStateManager::Initialize()
    {
        LOGINFO("VehicleStateManager: Initialization Started");

        SetWakeupTimeDelayMSec(1000);
        try
        {
            //_stopWatch.reset();
        }
        catch(exception e)
        {
            LOGERROR("VehicleStateManager: Event open exception: " << e.what());
        }

    }

    //Set Default System Values
    void VehicleStateManager::SetDefaultSystemValues()
    {
        _processCounter = 0;
        _lastSampleTime = 0.0;
        _filterGainAlpha = 0.99;
        _vehicleQIStateEst.MakeIdentity();
        _vehicleQIStatePred.MakeIdentity();

        _M4x4Tmp.zeros(4, 4);
        _M3x3Tmp.zeros(3, 3);
    }

    void VehicleStateManager::ResetState()
    {
        _lastSampleTime = 0.0;
        _vehicleQIStateEst.MakeIdentity();
        _vehicleQIStatePred.MakeIdentity();
        LOGWARN("Reset Vehicle Orientation States.");
    }

    void VehicleStateManager::UpdateVehicleOrientationInertialState(AccelerometerGyroMessage* agMsgPtr)
    {
        Quaternion_t qrot;
        arma::vec4 xTmp;
        XYZCoord_t zVec(0.0, 0.0, 1.0);
        XYZCoord_t rVec, tVec;

        try
        {
            double dt = agMsgPtr->IMUTimeStampSec - _lastSampleTime;
            //There should not be any big jumps in time... if there is
            //it means there was a stop in the process or restart.
            if(dt < 0.025 && dt > 0.0)
            {
                dt = 0.5 * dt;
                qrot.qScale = 1.0;
                qrot.qVec = agMsgPtr->GyroAngularRates * dt;
                _vehicleQIStatePred = _vehicleQIStateEst * qrot;

                if(!_headOrientationControlMsg->DisableVehicleGravityFeedback) //Turn On/Off Feedback filter
                {
                    rVec = _vehicleQIStatePred.rotateVecByQuaternion(zVec);
                    tVec = zVec.CrossProd(rVec);
                    double magt = tVec.Magnitude();
                    if(fabs(magt) < 1e-6)
                    {
                        _vehicleQIStateEst = _vehicleQIStatePred;
                    }
                    else
                    {
                        tVec = tVec / magt;
                        double dot = zVec.InnerProduct(rVec);
                        dot = dot > 1.0 ? 1.0 : dot < -1.0 ? -1.0 : dot;
                        double theta = -asin(dot);
                        theta = (1.0 - _filterGainAlpha) * theta;
                        rVec = tVec * theta;
                        qrot.MakeQuaternionFromRodriguesVector(rVec);
                        _vehicleQIStateEst = qrot * _vehicleQIStatePred;
                    }
                }
                else
                {
                    _vehicleQIStateEst = _vehicleQIStatePred;
                }

                _vehicleQIStateEst.Normalize();
                _vehicleOrientationQuaternionMsg->Quaternion = _vehicleQIStateEst;
                _vehicleOrientationQuaternionMsg->PostMessage();
            }
            _lastSampleTime = agMsgPtr->IMUTimeStampSec;
        }
        catch(std::exception &e)
        {
            LOGERROR("VehicleStateManager:UpdateVehicleOrientationInertialState: " << e.what());
        }
    }



    void VehicleStateManager::Startup()
    {

    }

    void VehicleStateManager::Shutdown()
    {
        _dataRecorder.closeLogFile();
        LOGINFO("VehicleStateManager shutdown");
    }

    AccelerometerGyroMessage* VehicleStateManager::GetNextAccelGyroMsg()
    {
        AccelerometerGyroMessage* agMsgPtr = nullptr;
        RabitMessage *rMsgPtr;
        if(VehicleStateIMUMsgRxQueue->GetMessage(rMsgPtr))
        {
            try
            {
                agMsgPtr = static_cast<AccelerometerGyroMessage *>(rMsgPtr);
            }
            catch(std::exception &e)
            {
                LOGERROR("VehicleStateManager:GetNextAccelGyroMsg Invalid message type received: " << e.what());
                //Return message to the IMU
                VehicleStateIMUEmptyMsgQueue->AddMessage(rMsgPtr);
                agMsgPtr = nullptr;
            }
        }
        return agMsgPtr;
    }

    void VehicleStateManager::ExecuteUnitOfWork()
    {
        AccelerometerGyroMessage* agMsgPtr;

        if( _headOrientationControlMsg->FetchMessage() )
        {
            //If changed... restart/re-initialized the car orientation
            //state.  New parameters will be picked up at this time.
            ResetState();
        }

        if( _resetOrientationStateMsg->FetchMessage())
        {
            if(_resetOrientationStateMsg->ResetVehicleOrientationState)
            {
                ResetState();
            }
        }

        //Get Accel-Gyro message from IMU... process the message
        //and then return the message back to the IMU.  The message
        //must be returned to the IMU or the pool of messages will quickly become empty.
        agMsgPtr = GetNextAccelGyroMsg();
        if( _headOrientationControlMsg->FetchMessage() )
        {
            //Reset the state... also picks up new filter gain
            _lastSampleTime = 0.0;
            _filterGainAlpha = _headOrientationControlMsg->VehicleGravityFeedbackGain;
            _vehicleQIStateEst.MakeIdentity();
            _vehicleQIStatePred.MakeIdentity();
        }

        while( agMsgPtr != nullptr )
        {
            try
            {
                UpdateVehicleOrientationInertialState(agMsgPtr);

                if( _headOrientationControlMsg->HeadOrientationOutputSelect == HeadOrientationOutputSelect_e::VehicleOrientation)
                {
                    std::shared_ptr<TrackHeadOrientationMessage> htOutMsg;
                    htOutMsg = std::make_shared<TrackHeadOrientationMessage>();
                    htOutMsg->TrackHeadOrientationData.HeadTranslationVec.Clear();
                    htOutMsg->TrackHeadOrientationData.HeadOrientationQuaternion = _vehicleOrientationQuaternionMsg->Quaternion;
                    htOutMsg->TrackHeadOrientationData.IsDataValid = true;
                    auto rmsgPtr = dynamic_pointer_cast<Rabit::RabitMessage, TrackHeadOrientationMessage>(htOutMsg);
                    AddMessageToQueue("ZMQPublishOutMsgQueue", rmsgPtr);
                }

                /**********************************
                if( ++_processCounter % 100 == 0)
                {
                    //Print out the Vehicle state Quaterion relative to the down axis
                    XYZCoord_t vz(0, 0, 1.0);
                    vz = _vehicleQIStateEst.rotateVecByQuaternion(vz);

                    std::cout << "VehicleStateManager Vec down (x, y, z): " << vz.x
                              << ", " << vz.y
                            << ", " << vz.z << std::endl;
                }
                 ************************************/

                //Generate the Vehicle Inertial State Information Message.
                //At this time the primary info will come from the GPS... as we improve
                //the vehicle model we need to add in the fixed IMU info.
                GPSFixMsgPtr->FetchMessage();
                _vehicleInertialStatesMsg->HeightAGL = 0;
                _vehicleInertialStatesMsg->LatLonAlt.SetLatitudeDegrees(GPSFixMsgPtr->GPSFixData.latitude);
                _vehicleInertialStatesMsg->LatLonAlt.SetLongitudeDegrees(GPSFixMsgPtr->GPSFixData.longitude);
                _vehicleInertialStatesMsg->XYZCoordinates = GPSFixMsgPtr->XYZPositionMeters;
                _vehicleInertialStatesMsg->XYZVelocities = GPSFixMsgPtr->XYZVelocityMetersPerSec;

                XYZCoord_t eAnglesRad = _vehicleOrientationQuaternionMsg->Quaternion.toEulerAngles(false);
                _vehicleInertialStatesMsg->RollPitchYaw.SetRollRadians(eAnglesRad.x);
                _vehicleInertialStatesMsg->RollPitchYaw.SetPitchRadians(eAnglesRad.y);
                _vehicleInertialStatesMsg->RollPitchYaw.SetYawRadians(eAnglesRad.z);

                _vehicleInertialStatesMsg->RollPitchYawRates.IsRate = true;
                _vehicleInertialStatesMsg->RollPitchYawRates.SetRollRadians(agMsgPtr->GyroAngularRates.x);
                _vehicleInertialStatesMsg->RollPitchYawRates.SetPitchRadians(agMsgPtr->GyroAngularRates.y);
                _vehicleInertialStatesMsg->RollPitchYawRates.SetYawRadians(agMsgPtr->GyroAngularRates.z);
                _vehicleInertialStatesMsg->PostMessage();
            }
            catch(std::exception &e)
            {
                LOGERROR("VehicleStateManager:Process Accel Gyro Msg Exception: " << e.what());
            }
            if( !VehicleStateIMUEmptyMsgQueue->AddMessage(agMsgPtr) )
            {
                //This should not occur... the IMU is not clearing this queue
                LOGERROR("VehicleStateManager:VehicleStateIMUEmptyMsgQueue is Full!");
            }
            agMsgPtr = GetNextAccelGyroMsg();
        }


        bool logMsgChanged = _loggingControlMsg->FetchMessage();
        if( EnableVehicleStatefLogging && _loggingControlMsg->EnableLogging)
        {
            _dataRecorder.writeDataRecord(_vehicleStateDataRecord);
        }
        else if(logMsgChanged && !_loggingControlMsg->EnableLogging)
        {
            _dataRecorder.closeLogFile();
        }


    }

}