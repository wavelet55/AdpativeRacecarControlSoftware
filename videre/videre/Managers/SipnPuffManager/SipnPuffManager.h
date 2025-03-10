/* ****************************************************************
 * Athr(s): Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: May 8, 2018
 *
 * Sip-n-Puff Manager
 * Origin Instruments Breeze Sip/Puff
 * Ties into the Linux input system: /dev/input/eventx
 *******************************************************************/

#ifndef VIDERE_DEV_SIPNPUFFMANAGER_H
#define VIDERE_DEV_SIPNPUFFMANAGER_H

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <RabitManager.h>
#include <RabitStopWatch.h>
#include <RabitMessageQueue.h>
#include <ManagerStatusMessage.h>
#include <ManagerControlMessage.h>
#include <ManagerStats.h>
#include <ManagerStatusMessage.h>
#include "global_defines.h"
#include "all_manager_message.h"
#include "../../Utils/config_data.h"
#include "../../Utils/logger.h"
#include "../../Utils/timing.h"
#include "SipnPuffMessage.h"
#include "LinearActuatorPositionCtrlMessage.h"
#include "DceEPASteeringStatusMessage.h"
#include "DataRecorder.h"
#include "DataRecorderStdHeader.h"
#include "SipnPuffDataRecord.h"
#include "ImageLoggingControlMessage.h"
#include "SipnPuffControlMessage.h"
#include "VidereSystemControlMessage.h"

// Manually include this file that has been autogenerated
#include "SipnPuffManagerWSRMgr.h"

using namespace Rabit;

enum BrakeThrottleCtrlState_e
{
    BTCS_Neutral,
    BTCS_Brake,
    BTCS_BrakeToThrottle,
    BTCS_Throttle
};

namespace videre
{
    class SipnPuffManager : public SipnPuffManagerWSRMgr
    {
    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        std::shared_ptr<ConfigData> _config_sptr;

        int sipnPufEventFileId = -1;

        double integrationRateFactor = 2.0;   //100% for 1 sec --> full actuation

        double _sipAndPuffVal = 0.0;

        RabitStopWatch _stopWatch;

        timeval _lastCaptureTime;

        std::thread _backgroundSPReadThread;
        bool _backgroundRxThreadIsRunning = false;
        bool _shutdown = false;

        double _sipnPuffConfigFeedbackMsgPubTime = 0.0;
        double _bciControlConfigFeedbackMsgPubTime = 0.0;

        //Messages
        std::shared_ptr<SipnPuffMessage> _sipnPuffValueMsg;

        std::shared_ptr<SipnPuffMessage> _sipnPuffDirectReadMsg;
        std::shared_ptr<SipnPuffMessage> _sipnPuffReadBackMsg;

        std::shared_ptr<SipnPuffControlMessage> _sipnPuffCtrlMsg;

        std::shared_ptr<VidereSystemControlMessage> VidereSystemCtrlMsg;


        std::shared_ptr<ImageLoggingControlMessage> _loggingControlMsg;

        DataRecorder _dataRecorder;
        DataRecorderStdHeader _dataRecorderStdHeader;
        SipnPuffDataRecord _sipnPuffDataRecord;

        bool EnableSipnPuffLogging = true;

        bool EnableNexusBCIThrottleControl = false;

        BrakeThrottleCtrlState_e _brakeThrottleCtrlState;

    public:
        SipnPuffManager(std::string name, std::shared_ptr<ConfigData> config);

        void readSipAndPuffSensorEventsThread();

        virtual void Initialize();

        virtual void ExecuteUnitOfWork() final;

        //The Startup method is called once when the manager thread is first
        //started.  It can be used for any necessary initialization processess
        //that have to be done after the manager's constructor.
        virtual void Startup() final;

        virtual void Shutdown() final;

        void standardSipnPuffThrottleControl();

        void nexusBCISipnPuffThrottleControl();

    };

}
#endif //VIDERE_DEV_SIPNPUFFMANAGER_H
