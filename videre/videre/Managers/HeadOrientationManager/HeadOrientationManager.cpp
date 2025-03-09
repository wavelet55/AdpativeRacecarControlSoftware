/* ****************************************************************
 * Athr(s): Randy Direen, PhD
 * Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: Aug, 2018
 *
 * Head Orientation Manager
 * Generates the best estimate of the head orientation within
 * the vehicle based upon IMU input, Image Processing input
 * and vehicle state input.
 *******************************************************************/


#include "HeadOrientationManager.h"
#include "Utilities/HeadTrackingCalParamsReaderWriter.h"
#include "XYZCoord_t.h"
#include <math.h>

using namespace std;

namespace videre
{

    HeadOrientationManager::HeadOrientationManager(std::string name,
                                     std::shared_ptr<ConfigData> config)
             : HeadOrientationManagerWSRMgr(name), _dataRecorder(),
              _headOrientationDataRecord(), _dataRecorderStdHeader("Head Orientation Data Log", 0)
    {
        SetWakeupTimeDelayMSec(50);
        _config_sptr = config;

        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        //Messages
        _trackHeadOrientationMsg = make_shared<TrackHeadOrientationMessage>();
        AddPublishSubscribeMessage("TrackHeadOrientationMessage", _trackHeadOrientationMsg);
        WakeUpManagerOnMessagePost(_trackHeadOrientationMsg);

        _vehicleOrientationQuaternionMsg = make_shared<QuaternionMessage>();
        AddPublishSubscribeMessage("VehicleOrientationQuaternionMsg", _vehicleOrientationQuaternionMsg);

        _headOrientationQuaternionMsg = make_shared<QuaternionMessage>();
        AddPublishSubscribeMessage("HeadOrientationQuaternionMsg", _headOrientationQuaternionMsg);

        _headOrientationCalDataMsg = std::make_shared<HeadOrientationCalDataMessage>();
        AddPublishSubscribeMessage("HeadOrientationCalDataMsg", _headOrientationCalDataMsg);

        _headOrientationControlMsg = make_shared<HeadOrientationControlMessage>();
        AddPublishSubscribeMessage("HeadOrientationControlMessage", _headOrientationControlMsg);

        _processedHeadOrientationMsg = std::make_shared<HeadOrientationMessage>();
        AddPublishSubscribeMessage("ProcessedHeadOrientationMsg", _processedHeadOrientationMsg);

        _resetOrientationStateMsg = std::make_shared<ResetOrientationStateMessage>();
        AddPublishSubscribeMessage("ResetOrientationStateMessage", _resetOrientationStateMsg);

        //Queues
        HeadOrientationIMUMsgRxQueue = std::make_shared<RabitMsgPtrSPSCQueue>(
                                       25, "HeadOrientationIMUMsgRxQueue");
        AddManagerMessageQueue(HeadOrientationIMUMsgRxQueue->GetMessageQueueName(),
                               HeadOrientationIMUMsgRxQueue);
        WakeUpManagerOnEnqueue(HeadOrientationIMUMsgRxQueue);

        HeadOrientationIMUEmptyMsgQueue = std::make_shared<RabitMsgPtrSPSCQueue>(
                                       30, "HeadOrientationIMUEmptyMsgQueue");
        AddManagerMessageQueue(HeadOrientationIMUEmptyMsgQueue->GetMessageQueueName(),
                               HeadOrientationIMUEmptyMsgQueue);


        _loggingControlMsg = std::make_shared<ImageLoggingControlMessage>();
        AddPublishSubscribeMessage("ImageLoggingControlMessage", _loggingControlMsg);

        _dataRecorder.setDirectory(DataLogDirectory);
        string fn = config->GetConfigStringValue("HeadOrientation.DataLogBaseFilename", "HeadOrientationDataLog");
        _dataRecorder.setBaseFilename(fn);
        _dataRecorder.setHeaderRecord(&_dataRecorderStdHeader);

        EnableHeadOrientationfLogging = config->GetConfigBoolValue("HeadOrientation.EnableLogging", true);
        TxHeadOrientationRateMsgPerSec = config->GetConfigDoubleValue("HeadOrientation.EnableLogging", 30);
    }

    void HeadOrientationManager::Initialize()
    {
        LOGINFO("HeadOrientationManager: Initialization Started");

        SetWakeupTimeDelayMSec(100);
        _lastHeadOrientationDataSentTimestamp = 0;
        try
        {
            std::string calParamsFilename = _config_sptr->GetConfigStringValue("HeadTrackingParameters.HeadOrientationCalFilename", "HeadOrientationCal.ini");
            //If there is a Config file... read parameters from the config file
            try
            {
                ReadHeadOrientationCalDataFromIniFile(calParamsFilename, _headOrientationCalDataMsg->CalData);
                _headOrientationCalDataMsg->PostMessage();

                //this is handled in the System Control Resournces
                //_headOrientationControlMsg->Clear();
                //_headOrientationControlMsg->HeadOrientation_QVar = q_initial;
                //_headOrientationControlMsg->HeadOrientation_RVar = r_initial;
                //_headOrientationControlMsg->SetVehicleGravityFeedbackGain(0.99);
                //_headOrientationControlMsg->HeadOrientationOutputSelect = HeadOrientationOutputSelect_e::HeadOrientation;
                //_headOrientationControlMsg->PostMessage();
            }
            catch (exception &e)
            {
                LOGWARN("Could not read HeadOrientationCalFilename.ini Config")
            }
            //_stopWatch.reset();
            SetDefaultSystemValues();
            ResetState();
        }
        catch(exception e)
        {
            LOGERROR("HeadOrientationManager: Event open exception: " << e.what());
        }

    }

    //Set Default System Values
    void HeadOrientationManager::SetDefaultSystemValues()
    {
        _lastSampleTime = 0.0;
        _headQIStateEst.MakeIdentity();
        _headQIStatePredict.MakeIdentity();
        _headOrientationCovarianceNorm = 0;

        SetQMatrix(q_initial);
        SetRMatrix(r_initial);
        InitHeadQIStateEst();

        _Pe.eye(4,4);
        _Pe = p_initial * p_initial * _Pe;

        _Pp.zeros(4,4);
        _A.zeros(4,4);
        _G.zeros(4,3);
        _H.zeros(4,4);
        _K.zeros(4,4);

        _deltaQ.zeros(4,3);
        _deltaQ(1,0) = 1.0;
        _deltaQ(2,1) = 1.0;
        _deltaQ(3,2) = 1.0;

        _M4x4Tmp.zeros(4, 4);
        _M3x3Tmp.zeros(3, 3);

        _tmpV1.zeros(4, 1);
        _tmpV2.zeros(4, 1);

    }

    void HeadOrientationManager::ResetState()
    {
        _lastSampleTime = 0.0;
        _QcameraMeasCar.MakeIdentity();
        InitHeadQIStateEst();
        SetQMatrix(_headOrientationControlMsg->HeadOrientation_QVar);
        SetRMatrix(_headOrientationControlMsg->HeadOrientation_RVar);
        _headOrientationCovarianceNorm = 0;

        _NoHeadGyroOrImageDataCount = 0;
        _headGyroUpdateCount = 0;
        _headNoGyroUpdateCount = 0;
        _headGyroUpdateRateSamplesPerSecond = 0.0;

        LOGWARN("Reset Head Orientation States.");

        _updateTimer.reset();
        _updateTimer.start();
    }

    void HeadOrientationManager::SetQMatrix(double var)
    {
        _Q.eye(3,3);
        _Q = var * var * _Q;

        _Qnomeas.eye(4,4);
        _Qnomeas = 1000.0 * var * var * _Qnomeas;
    }

    void HeadOrientationManager::SetRMatrix(double var)
    {
        _R.eye(4,4);
        _R = var * var * _R;
    }

    void HeadOrientationManager::InitHeadQIStateEst()
    {
        _headOrientationCalDataMsg->FetchMessage();
        _headQIStateEst = _headOrientationCalDataMsg->CalData.GyroToHeadQ;
        _headQIStatePredict = _headOrientationCalDataMsg->CalData.GyroToHeadQ;
    }

    void HeadOrientationManager::UpdateHeadOrientationInertialState(AccelerometerGyroMessage* agMsgPtr)
    {
        Quaternion_t qrot;
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
                //Make the A matrix
                qrot.toLeftMatrix(_A);

                //Make the G matrix
                Quaternion_t qh = _headQIStateEst * dt;
                qh.toRightMatrix(_M4x4Tmp);
                _G = _M4x4Tmp * _deltaQ;

                //compute the Predicted new state
                _headQIStatePredict = _headQIStateEst * qrot;

                if(_headQIStatePredict.CheckNAN())
                {
                    ++_headQIStateErrorCount;
                    LOGERROR("The headQIStateEst is invalid!... Reseting the head state.  Count: " << _headQIStateErrorCount);
                    ResetState();
                }
                else
                {
                    //Compute the new Coveriance Matrix.
                    _Pp = _A * _Pe * _A.t();
                    _Pp += _G * _Q * _G.t();
                }
            }
            _lastSampleTime = agMsgPtr->IMUTimeStampSec;
        }
        catch(std::exception &e)
        {
            LOGERROR("HeadOrientationManager:UpdateHeadOrientationInertialState: " << e.what());
        }
    }

    void HeadOrientationManager::UpdateHeadOrientationNoChange()
    {
        //The Estimated state does not change because we do not have any input
        //from the gyro... this will add noise to the covarience matrix.
        _headQIStatePredict = _headQIStateEst;
        _Pp = _Pe;
    }


    void HeadOrientationManager::UpdateHeadOrientationCoevarienceMatrix(double deltaTimeSec)
    {
        //The Estimated state does not change because we do not have any input
        //from the gyro... this will add noise to the covarience matrix.
        _headQIStatePredict = _headQIStateEst;
        _M4x4Tmp = 1000.0 * deltaTimeSec * _Qnomeas;
        _Pp = _Pe + _M4x4Tmp;
    }

    void HeadOrientationManager::ComputeKalmanGain(arma::dmat &P, arma::dmat &H, arma::dmat &R, arma::dmat &Kout)
    {
        try
        {
            _M4x4Tmp = H * P * H.t() + R;
            Kout = P * H.t() * arma::inv(_M4x4Tmp);

        }
        catch(std::exception &e)
        {
            LOGERROR("HeadOrientationManager:ComputeKalmanGain: " << e.what());
            Kout.zeros(4, 4);
        }
    }

    void HeadOrientationManager::AddCameraMeasurement(bool imageUpdateAvailable)
    {
        Quaternion_t qHeadCamaraPerspective;
        Quaternion_t qCar;
        Quaternion_t qHC;

        try
        {
            //Need to combine with the Car orentation to get head orientation
            //with respect to the car.
            _vehicleOrientationQuaternionMsg->FetchMessage();
            if(_headOrientationControlMsg->DisableVehicleInputToHeadOrientation)
            {
                qCar.MakeIdentity();
            }
            else
            {
                qCar = _vehicleOrientationQuaternionMsg->Quaternion.getCongugate();
            }


            if(imageUpdateAvailable && _trackHeadOrientationMsg->TrackHeadOrientationData.IsDataValid
               && !_headOrientationControlMsg->DisableHeadOrientationKalmanFilter)
            //if( _processCounter % 20 && !_headOrientationControlMsg->DisableHeadOrientationKalmanFilter)
            {
                //The H Matrix comes from the Car Orientation.
                qCar.toLeftMatrix(_H);
                ComputeKalmanGain(_Pp, _H, _R, _K);

                //Rotate the Camera Measurement into the Car frame of reference.
                //the camera is fixed to the car
                qHeadCamaraPerspective = _trackHeadOrientationMsg->TrackHeadOrientationData.HeadOrientationQuaternion;
                //qHeadCamaraPerspective.SetQuaternion(0, 1, 0, 0);
                _QcameraMeasCar = _headOrientationCalDataMsg->CalData.CameraToCarQ * qHeadCamaraPerspective;
                _QcameraMeasCar = _QcameraMeasCar * _headOrientationCalDataMsg->CalData.HeadToModelQ;
                _QcameraMeasCar = _QcameraMeasCar * _headOrientationCalDataMsg->CalData.GyroToHeadQ;

                //Rotate the Head Gyro Quaternion state into the car orientation
                qHC = qCar * _headQIStatePredict;
                Quaternion_t deltaCPH = _QcameraMeasCar - qHC;
                deltaCPH.toVector(_tmpV1);
                _tmpV2 = _K * _tmpV1;
                deltaCPH.SetQuaternion( _tmpV2 );
                _headQIStateEst = _headQIStatePredict + deltaCPH;

                //Validate the Estimate:
                if(_headQIStateEst.CheckNAN())
                {
                    LOGWARN("AddCameraMeasurement the headQIStateEst is invalid!")
                    _headQIStateEst = _headQIStatePredict;
                    _Pe = _Pp;
                }
                else
                {
                    //Update the Covariance matrix
                    _Pe = _Pp - _K * _H * _Pp;
                }
            }
            else
            {
                _headQIStateEst = _headQIStatePredict;
                _Pe = _Pp;
            }

            //Normialize Head Orientation State
            _headQIStateEst.Normalize();

            //Compute and output the Covariance Norm.
            double covarianceNorm = 0;
            for(int i = 0; i < 4; i++)
            {
                double cv = _Pe(i,i);
                covarianceNorm += cv * cv;
            }
            _headOrientationCovarianceNorm = sqrt(covarianceNorm);

            //Head Orientation relative to the car.
            _headOrientationQuaternionMsg->Quaternion =  qCar * _headQIStateEst * _headOrientationCalDataMsg->CalData.GyroToHeadQ;
            //_headOrientationQuaternionMsg->Quaternion = _QcameraMeasCar;
            _headOrientationQuaternionMsg->PostMessage();

            //Compute the Head Orientation and Post message: _processedHeadOrientationMsg
            //Vector facing to the front of the head.
            //Note:  The positive x-axis points out the front of the head/face.
            //The positive y-axis points out the left ear
            //The positive z-axis points out the top of the head.

            XYZCoord_t eulerAngles = _headOrientationQuaternionMsg->Quaternion.toEulerAngles(true);
            _processedHeadOrientationMsg->HeadRollPitchYawAnlges.SetRollDegrees(eulerAngles.x);
            _processedHeadOrientationMsg->HeadRollPitchYawAnlges.SetPitchDegrees(eulerAngles.y);
            _processedHeadOrientationMsg->HeadRollPitchYawAnlges.SetYawDegrees(eulerAngles.z);

            _processedHeadOrientationMsg->CovarianceNorm = _headOrientationCovarianceNorm;
            _processedHeadOrientationMsg->PostMessage();

        }
        catch(std::exception &e)
        {
            LOGERROR("HeadOrientationManager:AddCameraMeasurement: " << e.what());
            _K.zeros(4, 4);
        }
    }


    void HeadOrientationManager::Startup()
    {

    }

    void HeadOrientationManager::Shutdown()
    {
        _dataRecorder.closeLogFile();
        LOGINFO("HeadOrientationManager shutdown");
    }


    AccelerometerGyroMessage* HeadOrientationManager::GetNextAccelGyroMsg()
    {
        AccelerometerGyroMessage* agMsgPtr = nullptr;
        RabitMessage *rMsgPtr;
        if(HeadOrientationIMUMsgRxQueue->GetMessage(rMsgPtr))
        {
            try
            {
                agMsgPtr = static_cast<AccelerometerGyroMessage *>(rMsgPtr);
            }
            catch(std::exception &e)
            {
                LOGERROR("HeadOrientationManager:GetNextAccelGyroMsg Invalid message type received: " << e.what());
                //Return message to the IMU
                HeadOrientationIMUEmptyMsgQueue->AddMessage(rMsgPtr);
                agMsgPtr = nullptr;
            }
        }
        return agMsgPtr;
    }

    void HeadOrientationManager::ExecuteUnitOfWork()
    {
        AccelerometerGyroMessage* agMsgPtr;
        double deltaTimeSec = 0;

        //Ensure we have the latest Config parameters.
        _headOrientationCalDataMsg->FetchMessage();

        if( _headOrientationControlMsg->FetchMessage() )
        {
            //If changed... restart/re-initialized the head orientation
            //state.  New parameters will be picked up at this time.
            ResetState();
            _updateTimer.reset();
            _updateTimer.start();
        }

        if( _resetOrientationStateMsg->FetchMessage())
        {
            if(_resetOrientationStateMsg->ResetHeadOrientationState)
            {
                ResetState();
                _updateTimer.reset();
                _updateTimer.start();
            }
        }


        ++_processCounter;

        //Get Accel-Gyro message from IMU... process the message
        //and then return the message back to the IMU.  The message
        //must be returned to the IMU or the pool of messages will quickly become empty.
        agMsgPtr = GetNextAccelGyroMsg();
        bool imageProcHeadUpdate = _trackHeadOrientationMsg->FetchMessage();
        if( agMsgPtr == nullptr && ! imageProcHeadUpdate)
        {
            ++_NoHeadGyroOrImageDataCount;
            _updateTimer.captureTime();
            deltaTimeSec = _updateTimer.getTimeElapsed();
            if( _NoHeadGyroOrImageDataCount > 3 || deltaTimeSec > 0.100 )
            {
                //Todo:  stop car head control... there is a problem.
                LOGERROR("No Head Gyro or Image Data, Count: " << _NoHeadGyroOrImageDataCount
                                                              << " Delta Time sec=" << deltaTimeSec);
            }
        }
        else
        {
            while(agMsgPtr != nullptr || imageProcHeadUpdate)
            {
                _NoHeadGyroOrImageDataCount = 0;
                _updateTimer.tick();
                deltaTimeSec = _updateTimer.getTimeElapsed();

                try
                {
                    if(agMsgPtr != nullptr)
                    {
                        UpdateHeadOrientationInertialState(agMsgPtr);
                        ++_headGyroUpdateCount;
                    }
                    else if(deltaTimeSec > 0.015)
                    {
                        //We are slow or not getting updates from the head gyro.
                        //Update the Covarience matrix
                        UpdateHeadOrientationCoevarienceMatrix(deltaTimeSec);
                        ++_headNoGyroUpdateCount;
                        LOGINFO("UpdateHeadOrientationCoevarienceMatrix Count: " << _headNoGyroUpdateCount);
                    }
                    else
                    {
                        UpdateHeadOrientationNoChange();
                    }

                    AddCameraMeasurement(imageProcHeadUpdate);

                    double currentTimestamp = Rabit::SystemTimeClock::GetSystemTimeClock()->GetCurrentGpsTimeInSeconds();
                    double delTsec = currentTimestamp - _lastHeadOrientationDataSentTimestamp;

                    if(delTsec >= (1.0 / TxHeadOrientationRateMsgPerSec)
                       && _headOrientationControlMsg->HeadOrientationOutputSelect ==
                          HeadOrientationOutputSelect_e::HeadOrientation)
                    {
                        std::shared_ptr<TrackHeadOrientationMessage> htOutMsg;
                        htOutMsg = std::make_shared<TrackHeadOrientationMessage>();
                        htOutMsg->TrackHeadOrientationData.HeadTranslationVec.Clear();
                        htOutMsg->TrackHeadOrientationData.HeadOrientationQuaternion = _headOrientationQuaternionMsg->Quaternion;
                        htOutMsg->CovarianceNorm = _headOrientationCovarianceNorm;
                        htOutMsg->TrackHeadOrientationData.IsDataValid = true;
                        auto rmsgPtr = dynamic_pointer_cast<Rabit::RabitMessage, TrackHeadOrientationMessage>(htOutMsg);
                        AddMessageToQueue("ZMQPublishOutMsgQueue", rmsgPtr);

                        _lastHeadOrientationDataSentTimestamp = currentTimestamp;
                    }

                    //Process the message.
                    /******************************************
                    if( ++_processCounter % 250 == 0)
                    {
                        XYZCoord_t Angles = _headQIStateEst.toEulerAngles(true);
                        std::cout << "HeadOrientationAngles (Degs): "
                                                                            <<  Angles.x
                                                                            << ", " << Angles.y
                                                                            << ", " << Angles.z << std::endl;
                    }
                     ********************************/

                }
                catch(std::exception &e)
                {
                    LOGERROR("HeadOrientationManager:Process Accel Gyro Msg Exception: " << e.what());
                }
                if(agMsgPtr != nullptr)
                {
                    if(!HeadOrientationIMUEmptyMsgQueue->AddMessage(agMsgPtr))
                    {
                        //This should not occur... the IMU is not clearing this queue
                        LOGERROR("HeadOrientationManager:VehicleStateIMUEmptyMsgQueue is Full!");
                    }
                }
                agMsgPtr = GetNextAccelGyroMsg();
                imageProcHeadUpdate = _trackHeadOrientationMsg->FetchMessage();
            }
        }

        /******************************
        bool logMsgChanged = _loggingControlMsg->FetchMessage();
        if( EnableHeadOrientationfLogging && _loggingControlMsg->EnableLogging)
        {
            _dataRecorder.writeDataRecord(_headOrientationDataRecord);
        }
        else if(logMsgChanged && !_loggingControlMsg->EnableLogging)
        {
            _dataRecorder.closeLogFile();
        }
         *******************************/


    }

}