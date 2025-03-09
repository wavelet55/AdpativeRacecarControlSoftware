/* ****************************************************************
 * Athr(s): Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: May 8, 2018
 *
 * Sip-n-Puff Manager
 * Origin Instruments Breeze Sip/Puff
 * Ties into the Linux input system: /dev/input/eventx
 *
 * Check events from the Sip-and-puff with "evtest" (may need to install)
 * sudo evtest /dev/input/eventx
 *
 *******************************************************************/


#include "SipnPuffManager.h"
#include <linux/input.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

using namespace std;

namespace videre
{

    SipnPuffManager::SipnPuffManager(std::string name,
                                     std::shared_ptr<ConfigData> config)
            : SipnPuffManagerWSRMgr(name), _dataRecorder(),
              _sipnPuffDataRecord(), _dataRecorderStdHeader("Sip-And-Puff Data Log", 0)
    {
        this->SetWakeupTimeDelayMSec(0);
        _config_sptr = config;

        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        //Messages
        _sipnPuffDirectReadMsg = std::make_shared<SipnPuffMessage>();
        _sipnPuffReadBackMsg = std::make_shared<SipnPuffMessage>();
        _sipnPuffValueMsg = std::make_shared<SipnPuffMessage>();
        this->AddPublishSubscribeMessage("SipnPuffLCLThreadMsg", _sipnPuffDirectReadMsg);
        this->AddPublishSubscribeMessage("SipnPuffLCLThreadMsg", _sipnPuffReadBackMsg);
        this->AddPublishSubscribeMessage("SipnPuffMessage", _sipnPuffValueMsg);

        _sipnPuffCtrlMsg = std::make_shared<SipnPuffControlMessage>();
        AddPublishSubscribeMessage("SipnPuffControlMessage", _sipnPuffCtrlMsg);

        VidereSystemCtrlMsg = std::make_shared<VidereSystemControlMessage>();
        AddPublishSubscribeMessage("VidereSystemCtrlMsg", VidereSystemCtrlMsg);

        _loggingControlMsg = std::make_shared<ImageLoggingControlMessage>();
        AddPublishSubscribeMessage("ImageLoggingControlMessage", _loggingControlMsg);

        _dataRecorder.setDirectory(DataLogDirectory);
        string fn = config->GetConfigStringValue("SipnPuff.DataLogBaseFilename", "SipPuffDataLog");
        _dataRecorder.setBaseFilename(fn);
        _dataRecorder.setHeaderRecord(&_dataRecorderStdHeader);

        _sipnPuffDataRecord.SipnPuffValueMsg = _sipnPuffValueMsg;
        EnableSipnPuffLogging = config->GetConfigBoolValue("SipnPuff.EnableSipnPuffLogging", true);

        EnableNexusBCIThrottleControl = _config_sptr->GetConfigBoolValue("BCI_Throttle_Control.EnableNexusBCIThrottleControl", false);
        if(EnableNexusBCIThrottleControl)
        {
            LOGINFO("Nexus BCI Throttle Control is Enabled!");
            cout << "Nexus BCI Throttle Control is Enabled!" << endl;
        }

    }

    void SipnPuffManager::Initialize()
    {
        LOGINFO("SipnPuffManager: Initialization Started");

        string eventId = _config_sptr->GetConfigStringValue("SipnPuff.EventId", "/dev/input/by-id/usb-Origin_Instruments_Corp._Breeze-event-joystick");

        WorkSpace()->SipnPuffBCIConfigMsg->EnableSipnPuffIntegration = _config_sptr->GetConfigBoolValue("SipnPuff.SipnPuffEnabled", true);
        WorkSpace()->SipnPuffBCIConfigMsg->SipnPuffBlowGain = _config_sptr->GetConfigDoubleValue("SipnPuff.ThrottleSipnPuffGain", 0.25);
        WorkSpace()->SipnPuffBCIConfigMsg->SipnPuffSuckGain = _config_sptr->GetConfigDoubleValue("SipnPuff.BrakeSipnPuffGain", 1.0);
        WorkSpace()->SipnPuffBCIConfigMsg->SipnPuffDeadBandPercent = _config_sptr->GetConfigDoubleValue("SipnPuff.SipnPuffDeadBandPercent", 10.0);
        WorkSpace()->SipnPuffBCIConfigMsg->PostMessage();

        WorkSpace()->SipnPuffBCIConfigFeedbackMsg->CopyMessage(WorkSpace()->SipnPuffBCIConfigMsg.get());
        WorkSpace()->SipnPuffBCIConfigFeedbackMsg->PostMessage();
        _sipnPuffConfigFeedbackMsgPubTime = 0.0;

        _bciControlConfigFeedbackMsgPubTime = 0.0;


        try
        {
            _brakeThrottleCtrlState = BrakeThrottleCtrlState_e::BTCS_Neutral;
            _stopWatch.reset();

            sipnPufEventFileId = open(eventId.c_str(), O_RDONLY);

            _sipnPuffValueMsg->Clear();
            _sipnPuffValueMsg->PostMessage();

            _sipnPuffDirectReadMsg->Clear();
            _sipnPuffDirectReadMsg->PostMessage();
        }
        catch(exception e)
        {
            sipnPufEventFileId = -1;
            LOGERROR("SipnPuffManager: Event open exception: " << e.what());
        }

        if(sipnPufEventFileId >= 0)
        {
            this->SetWakeupTimeDelayMSec(10);
            LOGINFO("SipnPuffManager: Initialization Complete");
            std::cout << "SipnPuffManager: Initialization Complete" << std::endl;
        }
        else
        {
            this->SetWakeupTimeDelayMSec(10000);
            LOGINFO("SipnPuffManager: Failed to access Sip-N-Puff");
            std::cout << "SipnPuffManager: Failed to access Sip-N-Puff" << std::endl;
        }
    }

    void SipnPuffManager::Startup()
    {
        if(sipnPufEventFileId >= 0)
        {
            //Start Sip-and-Puff Read Thread
            _shutdown = false;
            _backgroundSPReadThread = std::thread(&SipnPuffManager::readSipAndPuffSensorEventsThread, this);

            this->SetWakeupTimeDelayMSec(10);
            LOGINFO("SipnPuffManager: Started Sip-N-Puff Background Read Thread.");

            _stopWatch.start();
        }
        else
        {
            LOGERROR("SipnPuffManager: Failed to start Sip-N-Puff Background Read Thread.");
        }
    }

    double deltaTime(timeval &ct, timeval lt)
    {
        double delTSec = (double)(ct.tv_sec - lt.tv_sec);
        delTSec += 0.0000001 * (double)(ct.tv_usec - lt.tv_usec);
        return delTSec;
    }


    void SipnPuffManager::ExecuteUnitOfWork()
    {
        bool vscChanged = VidereSystemCtrlMsg->FetchMessage();

        _sipnPuffCtrlMsg->FetchMessage();

        if( vscChanged || WorkSpace()->SipnPuffBCIConfigMsg->FetchMessage() )
        {
            //Clear the Throttle / Brake Integrator
            _sipnPuffValueMsg->SipnPuffIntegralPercent = 0;
            WorkSpace()->SipnPuffBCIValueMsg->SipnPuffIntegralPercent = 0;
            WorkSpace()->SipnPuffBCIConfigFeedbackMsg->CopyMessage(WorkSpace()->SipnPuffBCIConfigMsg.get());
            WorkSpace()->SipnPuffBCIConfigFeedbackMsg->PostMessage();
            _sipnPuffConfigFeedbackMsgPubTime = 0.0;
        }

        if( vscChanged || WorkSpace()->BCIControlConfigMsg->FetchMessage() )
        {
            //Clear the Throttle / Brake Integrator
            _sipnPuffValueMsg->SipnPuffIntegralPercent = 0;
            WorkSpace()->SipnPuffBCIValueMsg->SipnPuffIntegralPercent = 0;
            WorkSpace()->BCIControlConfigFeedbackMsg->CopyMessage(WorkSpace()->BCIControlConfigMsg.get());
            WorkSpace()->BCIControlConfigFeedbackMsg->PostMessage();
            _bciControlConfigFeedbackMsgPubTime = 0.0;
        }

        //Get the current sip-and-puff reading
        _sipnPuffReadBackMsg->FetchMessage();
        _sipnPuffValueMsg->SipnPuffPecent = _sipnPuffReadBackMsg->SipnPuffPecent;

        if( VidereSystemCtrlMsg->BCIControlEnable )
        {
            nexusBCISipnPuffThrottleControl();
        }
        else
        {
            standardSipnPuffThrottleControl();
        }

        bool logMsgChanged = _loggingControlMsg->FetchMessage();
        if( EnableSipnPuffLogging && _loggingControlMsg->EnableLogging)
        {
            _dataRecorder.writeDataRecord(_sipnPuffDataRecord);
        }
        else if(logMsgChanged && !_loggingControlMsg->EnableLogging)
        {
            _dataRecorder.closeLogFile();
        }
   }

   void SipnPuffManager::standardSipnPuffThrottleControl()
   {
        double integVal;
        double gain = 1.0;

        if(_sipnPuffCtrlMsg->EnableSipnPuffIntegration
                && (VidereSystemCtrlMsg->ThrottleControlEnable || VidereSystemCtrlMsg->BrakeControlEnable))
        {
            _stopWatch.tick();
            double delTsec = _stopWatch.getTimeElapsed();
            //ensure delta time is reasonable.
            delTsec = delTsec < 0.001 ? 0.001 : delTsec > 0.1 ? 0.1 : delTsec;

            if(fabs(_sipnPuffReadBackMsg->SipnPuffPecent) > _sipnPuffCtrlMsg->SipnPuffDeadBandPercent)
            {
                integVal = _sipnPuffValueMsg->SipnPuffIntegralPercent;
                switch(_brakeThrottleCtrlState)
                {
                    case BrakeThrottleCtrlState_e::BTCS_Neutral:
                        //In this state a sip or puff will control the next state.
                        _sipnPuffValueMsg->setSipnPuffIntegralPercent(0); //Start at neutral
                        if(_sipnPuffReadBackMsg->SipnPuffPecent > 0 && VidereSystemCtrlMsg->ThrottleControlEnable)
                        {
                            _brakeThrottleCtrlState = BrakeThrottleCtrlState_e::BTCS_Throttle;
                        }
                        else if(_sipnPuffReadBackMsg->SipnPuffPecent < 0 && VidereSystemCtrlMsg->BrakeControlEnable)
                        {
                            _brakeThrottleCtrlState = BrakeThrottleCtrlState_e::BTCS_Brake;
                        }
                        break;

                    case BrakeThrottleCtrlState_e::BTCS_Brake:
                        if( VidereSystemCtrlMsg->BrakeControlEnable )
                        {
                            if(_sipnPuffReadBackMsg->SipnPuffPecent > 75.0 )
                            {
                                //A strong puff will clear the brake control
                                _sipnPuffValueMsg->setSipnPuffIntegralPercent(0);
                                if(VidereSystemCtrlMsg->ThrottleControlEnable)
                                {
                                    _brakeThrottleCtrlState = BrakeThrottleCtrlState_e::BTCS_BrakeToThrottle;
                                }
                            }
                            else
                            {
                                if(integVal > 0)
                                {
                                    integVal = 0;   //Get off the gas and start applying brake
                                }
                                //The suck gain is always used for the brake.
                                gain = _sipnPuffCtrlMsg->SipnPuffSuckGain;
                                if(gain > 0)
                                {
                                    integVal += delTsec * gain * _sipnPuffReadBackMsg->SipnPuffPecent;
                                    if(integVal >= 0)
                                    {
                                        //We are off the brake.
                                        integVal = 0;
                                        if(VidereSystemCtrlMsg->ThrottleControlEnable)
                                        {
                                            _brakeThrottleCtrlState = BrakeThrottleCtrlState_e::BTCS_BrakeToThrottle;
                                        }
                                    }
                                    _sipnPuffValueMsg->setSipnPuffIntegralPercent(integVal);
                                }
                                else
                                {
                                    //a gain of zero mean ... simply apply the suck value to the integrator
                                    if(_sipnPuffReadBackMsg->SipnPuffPecent < 0)
                                    {
                                        _sipnPuffValueMsg->setSipnPuffIntegralPercent(
                                                _sipnPuffReadBackMsg->SipnPuffPecent);
                                    }
                                    else if(_sipnPuffReadBackMsg->SipnPuffPecent > 10.0 &&
                                              VidereSystemCtrlMsg->ThrottleControlEnable)
                                    {
                                        //done with braking if we have a positive puff.
                                        _sipnPuffValueMsg->setSipnPuffIntegralPercent(0);
                                        _brakeThrottleCtrlState = BrakeThrottleCtrlState_e::BTCS_BrakeToThrottle;
                                    }
                                    else
                                    {
                                        _sipnPuffValueMsg->setSipnPuffIntegralPercent(0);
                                    }
                                }
                            }
                        }
                        else
                        {
                            _brakeThrottleCtrlState = BrakeThrottleCtrlState_e::BTCS_BrakeToThrottle;
                        }
                        break;

                    case BrakeThrottleCtrlState_e::BTCS_BrakeToThrottle:
                        //In this case the driver has to back off the blow/puff to allow applying the gas
                        _sipnPuffValueMsg->setSipnPuffIntegralPercent(0);
                        if(VidereSystemCtrlMsg->ThrottleControlEnable)
                        {
                            if(_sipnPuffReadBackMsg->SipnPuffPecent < -75.0)
                            {
                                //Go to braking
                                _brakeThrottleCtrlState = BrakeThrottleCtrlState_e::BTCS_Brake;
                            }
                            else if(_sipnPuffReadBackMsg->SipnPuffPecent < 75.0)
                            {
                                _brakeThrottleCtrlState = BrakeThrottleCtrlState_e::BTCS_Throttle;
                            }
                        }
                        else
                        {
                            //we should not be here go to brake control
                            if(VidereSystemCtrlMsg->BrakeControlEnable)
                            {
                                _brakeThrottleCtrlState = BrakeThrottleCtrlState_e::BTCS_Brake;
                            }
                            else
                            {
                                _brakeThrottleCtrlState = BrakeThrottleCtrlState_e::BTCS_Neutral;
                            }
                        }
                        break;

                    case BrakeThrottleCtrlState_e::BTCS_Throttle:
                        if(VidereSystemCtrlMsg->ThrottleControlEnable)
                        {
                            if(_sipnPuffReadBackMsg->SipnPuffPecent < -75.0)
                            {
                                //A strong puff will clear the brake control
                                _sipnPuffValueMsg->setSipnPuffIntegralPercent(0);
                                if( VidereSystemCtrlMsg->BrakeControlEnable )
                                {
                                    _brakeThrottleCtrlState = BrakeThrottleCtrlState_e::BTCS_Brake;
                                }
                            }
                            else
                            {
                                if(integVal < 0)
                                {
                                    integVal = 0;   //Get off the gas and start applying brake
                                }
                                //The suck gain is always used for the brake.
                                gain = _sipnPuffCtrlMsg->SipnPuffBlowGain;
                                if(gain > 0)
                                {
                                    integVal += delTsec * gain * _sipnPuffReadBackMsg->SipnPuffPecent;
                                    if(integVal < 0)
                                    {
                                        //We are off the Gas.
                                        integVal = 0;
                                        if( VidereSystemCtrlMsg->BrakeControlEnable )
                                        {
                                            _brakeThrottleCtrlState = BrakeThrottleCtrlState_e::BTCS_Neutral;
                                        }
                                    }
                                    _sipnPuffValueMsg->setSipnPuffIntegralPercent(integVal);
                                }
                                else
                                {
                                    //a gain of zero mean ... simply apply the puff value to the integrator
                                    if(_sipnPuffReadBackMsg->SipnPuffPecent > 0)
                                    {
                                        _sipnPuffValueMsg->setSipnPuffIntegralPercent(
                                                _sipnPuffReadBackMsg->SipnPuffPecent);
                                    } else if(_sipnPuffReadBackMsg->SipnPuffPecent < -10.0)
                                    {
                                        //done with braking if we have a positive puff.
                                        _sipnPuffValueMsg->setSipnPuffIntegralPercent(0);
                                        if( VidereSystemCtrlMsg->BrakeControlEnable )
                                        {
                                            _brakeThrottleCtrlState = BrakeThrottleCtrlState_e::BTCS_Neutral;
                                        }
                                    }
                                }
                            }
                        }
                        else
                        {
                            _brakeThrottleCtrlState = BrakeThrottleCtrlState_e::BTCS_Neutral;
                        }
                        break;
                }

             }
        }
        else
        {
            //The brake-throttle control is disabled... go the the neutral position
            _sipnPuffValueMsg->setSipnPuffIntegralPercent(0);
            _brakeThrottleCtrlState = BrakeThrottleCtrlState_e::BTCS_Neutral;
        }
        _sipnPuffValueMsg->PostMessage();

   }

    void SipnPuffManager::nexusBCISipnPuffThrottleControl()
    {
        double integVal;
        double gain = 1.0;
        
        WorkSpace()->BCIThrottleControlMsg->FetchMessage();

        if(_sipnPuffCtrlMsg->EnableSipnPuffIntegration
                && (VidereSystemCtrlMsg->ThrottleControlEnable || VidereSystemCtrlMsg->BrakeControlEnable))
        {
            _stopWatch.tick();
            double delTsec = _stopWatch.getTimeElapsed();
            //ensure delta time is reasonable.
            delTsec = delTsec < 0.001 ? 0.001 : delTsec > 0.1 ? 0.1 : delTsec;

            if(fabs(_sipnPuffReadBackMsg->SipnPuffPecent) > _sipnPuffCtrlMsg->SipnPuffDeadBandPercent
                ||  WorkSpace()->BCIThrottleControlMsg->ThrottleOn)
            {
                integVal = _sipnPuffValueMsg->SipnPuffIntegralPercent;
                switch(_brakeThrottleCtrlState)
                {
                    case BrakeThrottleCtrlState_e::BTCS_Neutral:
                        //In this state a sip or puff will control the next state.
                        _sipnPuffValueMsg->setSipnPuffIntegralPercent(0); //Start at neutral
                        if(WorkSpace()->BCIThrottleControlMsg->ThrottleOn)
                        {
                            _brakeThrottleCtrlState = BrakeThrottleCtrlState_e::BTCS_Throttle;
                        }
                        else if(_sipnPuffReadBackMsg->SipnPuffPecent < 0 && VidereSystemCtrlMsg->BrakeControlEnable)
                        {
                            _brakeThrottleCtrlState = BrakeThrottleCtrlState_e::BTCS_Brake;
                        }
                        break;

                    case BrakeThrottleCtrlState_e::BTCS_Brake:
                        if( VidereSystemCtrlMsg->BrakeControlEnable )
                        {
                            if(_sipnPuffReadBackMsg->SipnPuffPecent > 75.0 )
                            {
                                //A strong puff will clear the brake control
                                _sipnPuffValueMsg->setSipnPuffIntegralPercent(0);
                                //if(VidereSystemCtrlMsg->ThrottleControlEnable)
                                {
                                    _brakeThrottleCtrlState = BrakeThrottleCtrlState_e::BTCS_BrakeToThrottle;
                                }
                            }
                            else
                            {
                                if(integVal > 0)
                                {
                                    integVal = 0;   //Get off the gas and start applying brake
                                }
                                //The suck gain is always used for the brake.
                                gain = _sipnPuffCtrlMsg->SipnPuffSuckGain;
                                gain = gain < 0.1 ? 0.1 : gain;  //A gain of zero is not allowed
                                if(gain > 0)
                                {
                                    if( _sipnPuffReadBackMsg->SipnPuffPecent >= 0
                                            && WorkSpace()->BCIThrottleControlMsg->ThrottleOn)
                                    {
                                        double pval = WorkSpace()->BCIControlConfigMsg->BCIThrottleIntegrationGain;
                                        pval += _sipnPuffCtrlMsg->SipnPuffBlowGain * _sipnPuffReadBackMsg->SipnPuffPecent;
                                        integVal += delTsec * pval;
                                    }
                                    else 
                                    {
                                        integVal += delTsec * gain * _sipnPuffReadBackMsg->SipnPuffPecent;
                                    }
                                    if(integVal >= 0)
                                    {
                                        //We are off the brake.
                                        integVal = 0;
                                        if(VidereSystemCtrlMsg->ThrottleControlEnable)
                                        {
                                            _brakeThrottleCtrlState = BrakeThrottleCtrlState_e::BTCS_BrakeToThrottle;
                                        }
                                    }
                                    _sipnPuffValueMsg->setSipnPuffIntegralPercent(integVal);
                                }
                            }
                        }
                        else
                        {
                            _brakeThrottleCtrlState = BrakeThrottleCtrlState_e::BTCS_BrakeToThrottle;
                        }
                        break;

                    case BrakeThrottleCtrlState_e::BTCS_BrakeToThrottle:
                        //In this case the driver has to back off the blow/puff to allow applying the gas
                        _sipnPuffValueMsg->setSipnPuffIntegralPercent(0);
                        if(VidereSystemCtrlMsg->ThrottleControlEnable)
                        {
                            if(_sipnPuffReadBackMsg->SipnPuffPecent < -75.0)
                            {
                                //Go to braking
                                _brakeThrottleCtrlState = BrakeThrottleCtrlState_e::BTCS_Brake;
                            }
                            else if(_sipnPuffReadBackMsg->SipnPuffPecent < 75.0)
                            {
                                _brakeThrottleCtrlState = BrakeThrottleCtrlState_e::BTCS_Throttle;
                            }
                        }
                        else
                        {
                            //we should not be here go to brake control
                            if(VidereSystemCtrlMsg->BrakeControlEnable)
                            {
                                _brakeThrottleCtrlState = BrakeThrottleCtrlState_e::BTCS_Brake;
                            }
                            else
                            {
                                _brakeThrottleCtrlState = BrakeThrottleCtrlState_e::BTCS_Neutral;
                            }
                        }
                        break;

                    case BrakeThrottleCtrlState_e::BTCS_Throttle:
                        if(VidereSystemCtrlMsg->ThrottleControlEnable)
                        {
                            if(_sipnPuffReadBackMsg->SipnPuffPecent < -75.0)
                            {
                                //A strong puff will clear the brake control
                                _sipnPuffValueMsg->setSipnPuffIntegralPercent(0);
                                if( VidereSystemCtrlMsg->BrakeControlEnable )
                                {
                                    _brakeThrottleCtrlState = BrakeThrottleCtrlState_e::BTCS_Brake;
                                }
                            }
                            else
                            {
                                if(integVal < 0)
                                {
                                    integVal = 0;   //Get off the gas and start applying brake
                                }
                                if( _sipnPuffReadBackMsg->SipnPuffPecent < - _sipnPuffCtrlMsg->SipnPuffDeadBandPercent )
                                {
                                    //The driver is trying to slow down, this take presidence over the Mind throttle On control
                                    gain = _sipnPuffCtrlMsg->SipnPuffSuckGain;
                                    gain = gain < 0.1 ? 0.1 : gain;
                                    integVal += delTsec * gain * _sipnPuffReadBackMsg->SipnPuffPecent;
                                }
                                else if( WorkSpace()->BCIThrottleControlMsg->ThrottleOn )
                                {
                                    double pval = WorkSpace()->BCIControlConfigMsg->BCIThrottleIntegrationGain;
                                    if( _sipnPuffReadBackMsg->SipnPuffPecent > _sipnPuffCtrlMsg->SipnPuffDeadBandPercent )
                                    {
                                        gain = _sipnPuffCtrlMsg->SipnPuffBlowGain;
                                        gain = gain < 0.0 ? 0.0 : gain;
                                        pval +=  gain * _sipnPuffReadBackMsg->SipnPuffPecent;
                                    }
                                    integVal += delTsec * pval;
                                }

                                if(integVal < 0)
                                {
                                    //We are off the Gas.
                                    integVal = 0;
                                    if( VidereSystemCtrlMsg->BrakeControlEnable )
                                    {
                                        _brakeThrottleCtrlState = BrakeThrottleCtrlState_e::BTCS_Neutral;
                                    }
                                }
                                _sipnPuffValueMsg->setSipnPuffIntegralPercent(integVal);
                            }
                        }
                        else
                        {
                            _brakeThrottleCtrlState = BrakeThrottleCtrlState_e::BTCS_Neutral;
                        }
                        break;
                }

             }
        }
        else
        {
            //The brake-throttle control is disabled... go the the neutral position
            _sipnPuffValueMsg->setSipnPuffIntegralPercent(0);
            _brakeThrottleCtrlState = BrakeThrottleCtrlState_e::BTCS_Neutral;
        }
        _sipnPuffValueMsg->PostMessage();
    }


    void SipnPuffManager::readSipAndPuffSensorEventsThread()
    {
        _backgroundRxThreadIsRunning = true;
        struct input_event ev;
        while(!_shutdown)
        {
            try
            {
                //This is a blocking call to read the event
                //This is not a normal for ExecuteUnitOfWork in a manager
                //but is ok since this is all the manager is doing and
                //it is on its own thread.
                read(sipnPufEventFileId, &ev, sizeof(struct input_event));
                if(ev.type == 3 && ev.code == 0)
                {
                    _sipAndPuffVal = (100.0 / 127.0) * (double)ev.value;
                    _sipnPuffDirectReadMsg->setSipnPuffPercent(_sipAndPuffVal);
                    _sipnPuffDirectReadMsg->SipnPuffIntegralPercent = 0;
                    _sipnPuffDirectReadMsg->PostMessage();

                    //cout << "SPVal: " << _sipAndPuffVal << std::endl;
                }
            }
            catch(exception e)
            {
                LOGERROR("readSipAndPuffSensorEventsThread: Sip-and-Puff Read Exception: " << e.what());
            }
        }
        _backgroundRxThreadIsRunning = false;
    }



    void SipnPuffManager::Shutdown()
    {
        _shutdown = true;
        _dataRecorder.closeLogFile();
        usleep(500000);
        if(sipnPufEventFileId >= 0)
        {
            try
            {
                close(sipnPufEventFileId);
            }
            catch(exception e)
            {
                LOGWARN("readSipAndPuffSensorEventsThread: Shutdown Exception: " << e.what());
            }
        }
        _backgroundSPReadThread.join();
        LOGINFO("SipnPuffManager shutdown");
    }

}