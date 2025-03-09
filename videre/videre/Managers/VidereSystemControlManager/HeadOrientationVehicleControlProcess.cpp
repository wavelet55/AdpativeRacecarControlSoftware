/* ****************************************************************
 * Athr(s): Harry Direen PhD,
 * Randy Direen, PhD
 *
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: Aug, 2018
 *
 * Head Orientation Vehicle Control Process.
 *******************************************************************/

#include "HeadOrientationVehicleControlProcess.h"
#include <math.h>

using namespace Rabit;
using namespace std;
using namespace MathLibsNS;

namespace videre
{

    HeadOrientationVehicleControlProcess::HeadOrientationVehicleControlProcess(Rabit::RabitManager* mgrPtr,
                                            std::shared_ptr<ConfigData> config,
                                                    SystemControlResources *SCRPtr)
        :_headLRLPF()
    {
        _mgrPtr = mgrPtr;
        _config_sptr = config;
        _SCRPtr = SCRPtr;

        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        double runRateHz = _config_sptr->GetConfigDoubleValue("HOVehicleControl.SystemControlMgrUpdateRateHz", 100.0);
        runRateHz = runRateHz < 10 ? 10 : runRateHz > 1000 ? 1000 : runRateHz;
        int delayms = (int)( (1000.0 / runRateHz) + 0.5 );

    }

    bool HeadOrientationVehicleControlProcess::Initialize()
    {
        bool error = false;

        Reset();

        SetupHeadLeftRighLPF();

        return error;
    }

    void HeadOrientationVehicleControlProcess::Shutdown()
    {}


    //Reset should be called when leaving and before
    //entering the HeadOrientationVehicleControlProcess state of operation.
    void HeadOrientationVehicleControlProcess::Reset()
    {
        _throttleCtrlPercent = 0.0;
        _brakeCtrlPercent = 0.0;
        _steeringTorqueControlPercent = 0.0;
        _throttleBrakeIntegral = 0;

        _headLRLPF.setInitialStateOrFilterOuput(0.0);

        _SCRPtr->ResetSteeringAngleFeedbackLoop();

        _stopwatch.stop();
        _stopwatch.reset();

        _SCRPtr->SipnPuffCtrlMsg->EnableSipnPuffIntegration = false;
        _SCRPtr->SipnPuffCtrlMsg->SipnPuffDeadBandPercent = _SCRPtr->VehicleControlParametersMsg->SipnPuffDeadBandPercent;
        _SCRPtr->SipnPuffCtrlMsg->SipnPuffBlowGain = _SCRPtr->VehicleControlParametersMsg->SipnPuffBlowGain;
        _SCRPtr->SipnPuffCtrlMsg->SipnPuffSuckGain = _SCRPtr->VehicleControlParametersMsg->SipnPuffSuckGain;
        _SCRPtr->SipnPuffCtrlMsg->PostMessage();
    }

    void HeadOrientationVehicleControlProcess::SetupHeadLeftRighLPF()
    {

        _headLRLPF.createButterworthFilter(_SCRPtr->VehicleControlParametersMsg->HeadLeftRighLPFOrder,
                                           _SCRPtr->VehicleControlParametersMsg->HeadLeftRighLPFCutoffFreqHz,
                                           _SCRPtr->ControlLoopRunRateHz );
    }

    SystemStateResponse_t  HeadOrientationVehicleControlProcess::ExecuteUnitOfWork()
    {
        double deltaTimeSec;
        double headAngleCtrlDeg;
        SystemStateResponse_t ssResponse;
        ssResponse.Clear();

        bool msgChanged = _SCRPtr->VehicleControlParametersMsg->FetchMessage();
        if(msgChanged)
        {
            //The filter parameters may have changed... so setup the LPF.
            SetupHeadLeftRighLPF();
        }

        if( msgChanged || (!_SCRPtr->SipnPuffCtrlMsg->EnableSipnPuffIntegration
                && !_SCRPtr->VehicleControlParametersMsg->ThrottleBrakeHeadTiltEnable ) )
        {
            _SCRPtr->SipnPuffCtrlMsg->EnableSipnPuffIntegration = true;
            _SCRPtr->SipnPuffCtrlMsg->SipnPuffDeadBandPercent = _SCRPtr->VehicleControlParametersMsg->SipnPuffDeadBandPercent;
            _SCRPtr->SipnPuffCtrlMsg->SipnPuffBlowGain = _SCRPtr->VehicleControlParametersMsg->SipnPuffBlowGain;
            _SCRPtr->SipnPuffCtrlMsg->SipnPuffSuckGain = _SCRPtr->VehicleControlParametersMsg->SipnPuffSuckGain;
            _SCRPtr->SipnPuffCtrlMsg->PostMessage();
        }

        _SCRPtr->HeadOrientationQuaternionMsg->FetchMessage();
        _SCRPtr->VehicleOrientationQuaternionMsg->FetchMessage();
        _SCRPtr->SipnPuffMsg->FetchMessage();
        _SCRPtr->GPSFixMsgPtr->FetchMessage();

        _stopwatch.tick();
        //Time in seconds since last tick();
        deltaTimeSec = _stopwatch.getTimeElapsed();

        deltaTimeSec = _SCRPtr->DeltaTimeSec;
        //ensure a reasonable time... there could have been a brake in action
        //or a new start.
        deltaTimeSec = deltaTimeSec < 0.001 ? 0.001 : deltaTimeSec > 0.1 ? 0.1 : deltaTimeSec;

        //X-Axis points forward from head and is the Roll Axis... positive angle is head rolled right
        //Y-Axis point out the left ear and is the pitch axis... positive angle is head down
        //Z-Axis points stright up from the head, the yaw axix... positive angle is head to the left.
        XYZCoord_t eulerAnglesDegrees = _SCRPtr->HeadOrientationQuaternionMsg->Quaternion.toEulerAngles(true);

        //Change the sign of the yaw axis to turning the head to the right give is a positive
        //angle.  Change the sign of the pitch axis to pointing the head up is a positive angle.
        eulerAnglesDegrees.z = -eulerAnglesDegrees.z;
        eulerAnglesDegrees.y = -eulerAnglesDegrees.y;

        _SCRPtr->SystemControlDataRecord.HeadRollAngleDegees =  eulerAnglesDegrees.x;
        _SCRPtr->SystemControlDataRecord.HeadPitchAngleDegrees =  eulerAnglesDegrees.y;
        _SCRPtr->SystemControlDataRecord.HeadYawAngleDegrees =  eulerAnglesDegrees.z;

        //Brake & Throttle Control
        if(_SCRPtr->VehicleControlParametersMsg->ThrottleBrakeHeadTiltEnable)
        {
            //ThrottleBrakeHeadTiltSipnPuffControl(deltaTimeSec, eulerAnglesDegrees);
            ThrottleBrakeSipnPuffControl();
        }
        else
        {
            ThrottleBrakeSipnPuffControl();
        }

        SteeringHeadAngleCtrl(deltaTimeSec, eulerAnglesDegrees);


        //Now control the actuators.
        SetBrakeAcutator();
        SetThrottleAcutator();

        //Send the Steering Control last... it is used to wake-up
        //the Vehicle Actuator Manager
        SetSteeringControlActuator();

        return ssResponse;
    }


    double HeadOrientationVehicleControlProcess::SteeringHeadAngleCtrl(double dt, const XYZCoord_t &eulerAnglesDegrees)
    {
        double headLRAngleDeg;
        if(_SCRPtr->VidereSystemCtrlMsg->HeadControlEnable)
        {
            //Head Input control
            //The z or yaw axis is the head turning right or left. Right is positive.
            headLRAngleDeg = eulerAnglesDegrees.z;
            if(isnan(headLRAngleDeg) || !isfinite(headLRAngleDeg))
            {
                LOGWARN("SteeringHeadAngleCtrl LR Head angle is invalid!");
                headLRAngleDeg = 0;
            }
            double maxHeadRotDeg = _SCRPtr->VehicleControlParametersMsg->MaxLRHeadRotationDegrees;
            maxHeadRotDeg = maxHeadRotDeg < 30 ? 30 : maxHeadRotDeg > 80 ? 80 : maxHeadRotDeg;

            headLRAngleDeg += _SCRPtr->VehicleControlParametersMsg->SteeringBiasAngleDegrees;
            headLRAngleDeg =
                    headLRAngleDeg < -maxHeadRotDeg ? -maxHeadRotDeg : headLRAngleDeg > maxHeadRotDeg ? maxHeadRotDeg
                                                                                                      : headLRAngleDeg;
            if(fabs(headLRAngleDeg) < _SCRPtr->VehicleControlParametersMsg->SteeringDeadband)
            {
                headLRAngleDeg = 0.0;
            }

            _SCRPtr->SystemControlDataRecord.HeadLRAngleClamped = headLRAngleDeg;
            //Low Pass Filter the Head Angle to reduce bounce in head position.
            headLRAngleDeg = _headLRLPF.fx(headLRAngleDeg);
            _SCRPtr->SystemControlDataRecord.HeadLRAngleLPF = headLRAngleDeg;

            if(_SCRPtr->VehicleControlParametersMsg->UseSteeringAngleControl)
            {
                headLRAngleDeg = _SCRPtr->VehicleControlParametersMsg->SteeringControlGain * headLRAngleDeg;
                _steeringTorqueControlPercent = _SCRPtr->SteeringAngleFeedbackLoop(headLRAngleDeg, dt);
            }
            else
            {
                //Steering Torque control Percent.
                _steeringTorqueControlPercent =
                        100.0 * _SCRPtr->VehicleControlParametersMsg->SteeringControlGain * headLRAngleDeg /
                        maxHeadRotDeg;
            }
        }
        else
        {
            _headLRLPF.setInitialStateOrFilterOuput(0);
            _SCRPtr->ResetSteeringAngleFeedbackLoop();
            _steeringTorqueControlPercent = 0;
        }

        return _steeringTorqueControlPercent;
    }


    void HeadOrientationVehicleControlProcess::ThrottleBrakeSipnPuffControl()
    {
        if(!_SCRPtr->SipnPuffCtrlMsg->EnableSipnPuffIntegration)
        {
            _SCRPtr->SipnPuffCtrlMsg->EnableSipnPuffIntegration = true;
            _SCRPtr->SipnPuffCtrlMsg->PostMessage();
        }

        //Don't allow reversing the control at this point in time.
        _throttleBrakeIntegral = _SCRPtr->SipnPuffMsg->SipnPuffIntegralPercent;

        /*****************************
        if(_SCRPtr->VehicleControlParametersMsg->ReverseSipnPuffThrottleBrake)
        {
            _throttleBrakeIntegral = -1.0 * _SCRPtr->SipnPuffMsg->SipnPuffIntegralPercent;
        }
        else
        {
            _throttleBrakeIntegral = _SCRPtr->SipnPuffMsg->SipnPuffIntegralPercent;
        }
         ******************************/

        _SCRPtr->SystemControlDataRecord.ThrottleBrakeIntegralVal = _SCRPtr->SipnPuffMsg->SipnPuffIntegralPercent;
    }


    void HeadOrientationVehicleControlProcess::ThrottleBrakeHeadTiltSipnPuffControl(double dt, const XYZCoord_t &eulerAnglesDegrees)
    {
        //combine both sip-n-puff and head tilt forward/backwards for control
        //use an integral to apply the control.
        double spGain = _SCRPtr->SipnPuffMsg->SipnPuffPecent >= 0 ? _SCRPtr->SipnPuffCtrlMsg->SipnPuffBlowGain
                                                                  : _SCRPtr->SipnPuffCtrlMsg->SipnPuffSuckGain;

        if(fabs(_SCRPtr->SipnPuffMsg->SipnPuffPecent) >= _SCRPtr->SipnPuffCtrlMsg->SipnPuffDeadBandPercent)
        {
            if(_SCRPtr->VehicleControlParametersMsg->ReverseSipnPuffThrottleBrake)
            {
                spGain = -1.0 * spGain;
            }
            _throttleBrakeIntegral += dt * spGain * _SCRPtr->SipnPuffMsg->SipnPuffPecent;
        }

        //Head Input control
        double fbAngleDeg = eulerAnglesDegrees.y;
        if(fbAngleDeg >= 0)
        {
            //Head tilted up
            if(fbAngleDeg > _SCRPtr->VehicleControlParametersMsg->ThrottleBrakeHeadTiltBackDeadbandDegrees)
            {
                fbAngleDeg = fbAngleDeg > 45.0 ? 45.0 : fbAngleDeg;
                double range = 45.0 - _SCRPtr->VehicleControlParametersMsg->ThrottleBrakeHeadTiltBackDeadbandDegrees;
                range = range < 20 ? 20 : range;
                double amt = fbAngleDeg - _SCRPtr->VehicleControlParametersMsg->ThrottleBrakeHeadTiltBackDeadbandDegrees;
                amt = 100.0 * amt / range;
                _throttleBrakeIntegral -= dt * _SCRPtr->VehicleControlParametersMsg->BrakeHeadTiltGain * amt;
            }
        }
        else
        {
            //Head Tilted down
            fbAngleDeg = -fbAngleDeg;
            if(fbAngleDeg > _SCRPtr->VehicleControlParametersMsg->ThrottleBrakeHeadTiltForwardDeadbandDegrees)
            {
                fbAngleDeg = fbAngleDeg > 45.0 ? 45.0 : fbAngleDeg;
                double range = 45.0 - _SCRPtr->VehicleControlParametersMsg->ThrottleBrakeHeadTiltForwardDeadbandDegrees;
                range = range < 20 ? 20 : range;
                double amt = fbAngleDeg - _SCRPtr->VehicleControlParametersMsg->ThrottleBrakeHeadTiltForwardDeadbandDegrees;
                amt = 100.0 * amt / range;
                _throttleBrakeIntegral += dt * _SCRPtr->VehicleControlParametersMsg->BrakeHeadTiltGain * amt;
            }
        }
        _throttleBrakeIntegral = _throttleBrakeIntegral > 100.0 ? 100.0 :
                                 _throttleBrakeIntegral < -100.0 ? -100.0 : _throttleBrakeIntegral;

        _SCRPtr->SystemControlDataRecord.ThrottleBrakeIntegralVal = _throttleBrakeIntegral;
    }


    void HeadOrientationVehicleControlProcess::SetBrakeAcutator()
    {
        _SCRPtr->BrakeControlMsg->FunctionType = LinearActuatorFunction_e::LA_Brake;
        _SCRPtr->BrakeControlMsg->ManualExtControl = false;
        _SCRPtr->BrakeControlMsg->ActuatorSetupMode = false;
        if( _SCRPtr->VidereSystemCtrlMsg->BrakeControlEnable)
        {
            if(_throttleBrakeIntegral < 0)
            {
                _brakeCtrlPercent = -1.0 * _throttleBrakeIntegral;
                _SCRPtr->BrakeControlMsg->setPositionPercent(_brakeCtrlPercent, true);
            } else
            {
                //Return Brake actuator to the default state.
                _brakeCtrlPercent = 0;
                _SCRPtr->BrakeControlMsg->setPositionPercent(_brakeCtrlPercent, true);
            }
        }
        else
        {
            _brakeCtrlPercent = 0;
            _SCRPtr->BrakeControlMsg->setPositionPercent(_brakeCtrlPercent, false);
        }
        _SCRPtr->BrakeControlMsg->PostMessage();
        _SCRPtr->SystemControlDataRecord.BrakecontrolVal = _brakeCtrlPercent;
    }

    void HeadOrientationVehicleControlProcess::SetThrottleAcutator()
    {
        _SCRPtr->ThrottlePositionControlMsg->FunctionType = LinearActuatorFunction_e::LA_Accelerator;
        _SCRPtr->ThrottlePositionControlMsg->ManualExtControl = false;
        _SCRPtr->ThrottlePositionControlMsg->ActuatorSetupMode = false;
        if( _SCRPtr->VidereSystemCtrlMsg->ThrottleControlEnable)
        {
            if(_throttleBrakeIntegral > 0)
            {
                _throttleCtrlPercent = _throttleBrakeIntegral;
                _SCRPtr->ThrottlePositionControlMsg->setPositionPercent(_throttleCtrlPercent, true);
            } else
            {
                //Return Brake actuator to the default state.
                _throttleCtrlPercent = 0;
                _SCRPtr->ThrottlePositionControlMsg->setPositionPercent(_throttleCtrlPercent, true);
            }
        }
        else
        {
            _throttleCtrlPercent = 0;
            _SCRPtr->ThrottlePositionControlMsg->setPositionPercent(_throttleCtrlPercent, false);
        }
        _SCRPtr->ThrottlePositionControlMsg->PostMessage();
        _SCRPtr->SystemControlDataRecord.ThrottleControlVal = _throttleCtrlPercent;
    }

    //Send the Steering Control last... it is used to wake-up
    //the Vehicle Actuator Manager
    void HeadOrientationVehicleControlProcess::SetSteeringControlActuator()
    {
        if( _SCRPtr->VidereSystemCtrlMsg->HeadControlEnable)
        {
            _SCRPtr->SteeringTorqueCtrlMsg->setSteeringTorqueMap(3);
            _SCRPtr->SteeringTorqueCtrlMsg->SteeringControlEnabled = true;
            _SCRPtr->SteeringTorqueCtrlMsg->setSteeringTorquePercent(_steeringTorqueControlPercent);
        }
        else
        {
            _SCRPtr->SteeringTorqueCtrlMsg->setSteeringTorqueMap(0);
            _SCRPtr->SteeringTorqueCtrlMsg->SteeringControlEnabled = false;
            _steeringTorqueControlPercent = 0;
            _SCRPtr->SteeringTorqueCtrlMsg->setSteeringTorquePercent(_steeringTorqueControlPercent);
        }
        _SCRPtr->SteeringTorqueCtrlMsg->PostMessage();
        _SCRPtr->SystemControlDataRecord.SteeringTorqueCtrl = _steeringTorqueControlPercent;

    }


}