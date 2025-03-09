/* ****************************************************************
 * Athr(s): Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: Jan. 13, 2018
 *
 * NeuroGroove IMU Communications Interface
 *******************************************************************/

#ifndef VIDERE_DEV_IMU_RXMESSAGEPARSER_H
#define VIDERE_DEV_IMU_RXMESSAGEPARSER_H

#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <memory>
#include <RabitManager.h>
#include <message_pool.h>
#include "config_data.h"
#include "logger.h"
#include "RS232Comm.h"
#include "SerialCommMessage.h"
#include "IMU_DataTypeDefs.h"
#include "HeadOrientationMessage.h"
#include "AccelerometerGyroMessage.h"
#include "DataRecorder.h"
#include "IMU_DataRecord.h"
#include "DataRecorderStdHeader.h"
#include "ImageLoggingControlMessage.h"
#include "IMUCommandResponseMessage.h"
#include "HeadOrientationControlMessage.h"


using namespace videre;

namespace IMU_SensorNS
{

    class IMU_RxMessageParser
    {
    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        std::shared_ptr<ConfigData> _config_sptr;

        //The reference to the manager is used primarily for
        //setup purposes.
        Rabit::RabitManager* _mgrPtr;

        char _receiveRS232CommMsgBuf[IMU_MAXMESSAGESIZE];

        //Messages
        std::shared_ptr<HeadOrientationMessage> _headOrientationMsg;

        std::shared_ptr<AccelerometerGyroMessage> _accelGyroHeadMsg;

        std::shared_ptr<AccelerometerGyroMessage> _accelGyroFixedMsg;

        std::shared_ptr<ImageLoggingControlMessage> _loggingControlMsg;

        std::shared_ptr<IMUCommandResponseMessage> _imuResponseMessageMsg;

        std::shared_ptr<HeadOrientationControlMessage> _headOrientationControlMsg;

        DataRecorder _dataRecorder;

        DataRecorderStdHeader _dataRecorderStdHeader;
        IMU_HeadOrientationRecord _headOrientationRecord;
        IMU_AccelGyroRecord _accelGyroRecord;

        bool _mgrIsRunning = false;

        bool SendHighSpeedIMUDataOut = false;

        uint32_t _headAccelGyroMsgCount = 0;
        uint32_t _fixedAccelGyroMsgCount = 0;

        uint32_t _headAccelGyroInvalidDataCount = 0;
        uint32_t _fixedAccelGyroInvalidDataCount = 0;

        uint32_t _totalNumberImuDataMsgs = 0;
        uint32_t _totalNumberImuDataMsgCrcErrors = 0;


        //The IMUCommManager keeps of pool of AccelerometerGyroMessage
        //messages.  The IMUCommManager receives and sends out to other managers
        //a large number of these messages every second.  To prevent the constant
        //creating and distroying of these messages, a message pool is used.
        //The pool size should be large enough to keep the queues to the different
        //managers full or close to full.  This way if 1 manager is not responding
        //or keeping up, there are still messages for the other managers.
        //This pool feeds messages to the VehicleStateManager, the HeadOrientationManager,
        //and the CommsManager.
        const int AccelGyroMsgPoolSize = 55;
        std::unique_ptr<MessagePool> _accelerometerGyroMsgPool;

        //Queues
        std::shared_ptr<RabitMsgPtrSPSCQueue> HeadOrientationIMUMsgRxQueue;
        std::shared_ptr<RabitMsgPtrSPSCQueue> HeadOrientationIMUEmptyMsgQueue;

        std::shared_ptr<RabitMsgPtrSPSCQueue> VehicleStateIMUMsgRxQueue;
        std::shared_ptr<RabitMsgPtrSPSCQueue> VehicleStateIMUEmptyMsgQueue;

        Data_AccelGyro_t _maxAccelGyroInputVals;


    public:
        bool EnableHeadOrientationLogging = true;
        bool EnableAccelGyroLogging = true;

        bool RemoteCtrlEnabled = false;

        bool MgrIsRunning = false;


        IMU_RxMessageParser(Rabit::RabitManager* mgrPtr,
                            std::shared_ptr<ConfigData> config);

        bool Initialize();

        void Shutdown();

        void ReturnEmptyMsgsToPool();

        AccelerometerGyroMessage* CheckoutAccelerometerGyroMsgFromPool();

        bool parseOnOff(char* val);

        Imu_SensorId_e parseImuSensorID(int sId);

        void setBinaryFlagsForSensor(Imu_SensorId_e sId, bool *fixedSenFlag, bool *headSenFlag, bool bVal);

        bool getBinaryFlagForSensor(Imu_SensorId_e sId, bool fixedSenFlag, bool headSenFlag);

        void setIntValForSensor(Imu_SensorId_e sId, int *fixedVal, int *headVal, int value);

        int getIntValForSensor(Imu_SensorId_e sId, int fixedSenFlag, int headSenFlag);

        int TestRS232CommMsgSize = 0;

        void rxMsgHandler(dtiUtils::SerialCommMessage_t &msg);

        void parseCmdResponse(char *pCmd, char *pValue);

        void parseIMUData(char *pCmd, char *pValue);

        //A 16-bit CRC is the last two characters in Hex format on the message
        //string.
        bool checkMsgCRCOk(char* pCmd, int msgSize);


    };

}
#endif //VIDERE_DEV_IMU_RXMESSAGEPARSER_H
