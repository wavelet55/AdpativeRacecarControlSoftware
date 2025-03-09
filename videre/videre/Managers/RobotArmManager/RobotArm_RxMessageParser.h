/* ****************************************************************
 * Athr(s): Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: Jan. 13, 2018
 *
 * NeuroGroove IMU Communications Interface
 *******************************************************************/

#ifndef VIDERE_DEV_ROBOTARM_RXMESSAGEPARSER_H
#define VIDERE_DEV_ROBOTARM_RXMESSAGEPARSER_H

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

namespace videre
{

#define ROBOTARM_MAXMESSAGESIZE 128


    class RobotArm_RxMessageParser
    {
    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        std::shared_ptr<ConfigData> _config_sptr;

        //The reference to the manager is used primarily for
        //setup purposes.
        Rabit::RabitManager* _mgrPtr;

        char _receiveRS232CommMsgBuf[ROBOTARM_MAXMESSAGESIZE];

    public:
        RobotArm_RxMessageParser(Rabit::RabitManager* mgrPtr,
                            std::shared_ptr<ConfigData> config);

        void rxMsgHandler(dtiUtils::SerialCommMessage_t &msg);

        int TestRS232CommMsgSize = 0;

    };

}
#endif //VIDERE_DEV_ROBOTARM_RXMESSAGEPARSER_H
