/* ****************************************************************
 * Athr(s): Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: Jan. 13, 2018
 *
 * NeuroGroove IMU Communications Interface
 *******************************************************************/

#ifndef VIDERE_DEV_REMOTECONTROLRXMESSAGEPARSER_H
#define VIDERE_DEV_REMOTECONTROLRXMESSAGEPARSER_H

#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <memory>
#include <RabitManager.h>
#include "config_data.h"
#include "logger.h"
#include "RS232Comm.h"
#include "SerialCommMessage.h"
#include "RemoteControlInputMessage.h"

namespace videre
{

#define REMOTECONTROL_MAXMESSAGESIZE 64


    class RemoteControlRxMessageParser
    {
    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        std::shared_ptr<ConfigData> _config_sptr;

        //The reference to the manager is used primarily for
        //setup purposes.
        Rabit::RabitManager* _mgrPtr;

        char _receiveRS232CommMsgBuf[REMOTECONTROL_MAXMESSAGESIZE];

        std::shared_ptr<RemoteControlInputMessage> _remoteControlInputMsg;

    public:
        //Control Parameters
        bool ReverseThrottleBrake = false;
        double ThrottleBrakeDeadBandPercent = 5.0;
        bool ReverseSteeringControl = false;
        double SteeringDeadbandPercent = 2.5;


        RemoteControlRxMessageParser(Rabit::RabitManager* mgrPtr,
                            std::shared_ptr<ConfigData> config);

        bool Initialize();

        void rxMsgHandler(dtiUtils::SerialCommMessage_t &msg);

        int TestRS232CommMsgSize = 0;

    };

}
#endif //VIDERE_DEV_REMOTECONTROLRXMESSAGEPARSER_H
