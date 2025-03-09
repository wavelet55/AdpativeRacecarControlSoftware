/* ****************************************************************
 * Athr(s): Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: Sept. 2021
 *
 * DTX IMU Message Parser
 *******************************************************************/

#ifndef VIDERE_DEV_DTX_IMU_RXMESSAGEPARSER_H
#define VIDERE_DEV_DTX_IMU_RXMESSAGEPARSER_H

#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <memory>
#include <RabitManager.h>
#include "config_data.h"
#include "global_defines.h"
#include "logger.h"
#include "RS232Comm.h"
#include "DataRecorder.h"
//#include "GF360CI_ExerciseProgressRecord.h"
//#include "GF360LateralMotorPosCtrlMonitorRecord.h"
//#include "DataRecorderStdHeader.h"
#include "SerialCommMessage.h"
#include "message_pool.h"


namespace videre
{
    //Manager Workspace type:
    class DTX_IMU_InterfaceManagerWSRMgrWorkSpace;

    class DTX_IMU_RxMessageParser
    {
    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        std::shared_ptr<ConfigData> _config_sptr;

        //The reference to the manager is used primarily for
        //setup purposes.
        DTX_IMU_InterfaceManagerWSRMgrWorkSpace* _wsPtr;

        Rabit::RabitManager* _mgrPtr;

        //DTX_DataRecords::DataRecorder _dataRecorder;
//
        //DTX_DataRecords::DataRecorderStdHeader _dataRecorderStdHeader;
        //DTX_DataRecords::gf360ExerciseDataWrapperRecord _exDataWrapperRecord;
        //DTX_DataRecords::GF360CI_ExerciseProgressRecord _exerciseProgressRecord;
        //DTX_DataRecords::GF360LateralMotorPosCtrlMonitorRecord _lateralMotorPosCtrlMonitorRecord;


        bool _recordExerciseData = false;

        //GF360_RecordControlCmd_e _currentRecordCtrlState = GF360_RecordControlCmd_e::GF360RCC_None;

        //Messages
        bool _mgrIsRunning = false;

        uint32_t _totalNumberMsgs = 0;
        uint32_t _totalNumberMsgCrcErrors = 0;

    public:
        bool MgrIsRunning = false;

        DTX_IMU_RxMessageParser(Rabit::RabitManager* mgrPtr,
                            DTX_IMU_InterfaceManagerWSRMgrWorkSpace* mgrWSPtr,
                            std::shared_ptr<ConfigData> config);

        bool Initialize();

        void Shutdown();

        bool parseOnOff(char* val);

        int TestRS232CommMsgSize = 0;

        void rxMsgHandler(dtiUtils::SerialCommMessage_t &msg);

        int checkDataRecorderProcess();

        int startNewDataRecordLogfile();

        //A 16-bit CRC is the last two characters in Hex format on the message
        //string.
        bool checkMsgCRCOk(char* pCmd, int msgSize);


    };

}
#endif //VIDERE_DEV_IMU_RXMESSAGEPARSER_H
