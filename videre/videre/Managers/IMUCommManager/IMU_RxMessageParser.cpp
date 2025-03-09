/* ****************************************************************
 * Athr(s): Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: Jan. 13, 2018
 *
 * NeuroGroove IMU Communications Interface
 *******************************************************************/

#include "IMU_RxMessageParser.h"
#include "global_defines.h"
#include "IMU_DataTypeDefs.h"
#include <RabitMessage.h>
#include <CRC_Calculator.h>
#include <Base64Conversions.h>


using namespace VidereUtils;
using namespace std;

namespace IMU_SensorNS
{

    IMU_RxMessageParser::IMU_RxMessageParser(Rabit::RabitManager* mgrPtr,
                                             std::shared_ptr<ConfigData> config)
        : _dataRecorder(), _headOrientationRecord(), _accelGyroRecord(),
          _dataRecorderStdHeader("IMU Data Log", 0)
    {
        _mgrPtr = mgrPtr;
        _config_sptr = config;
        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        _headOrientationMsg = std::make_shared<HeadOrientationMessage>();
        _mgrPtr->AddPublishSubscribeMessage("IMUHeadOrientationMsg", _headOrientationMsg);

        _accelGyroHeadMsg = std::make_shared<AccelerometerGyroMessage>();
        _mgrPtr->AddPublishSubscribeMessage("AccelerometerGyroHeadMsg", _accelGyroHeadMsg);
        _accelGyroHeadMsg->IMU_SensorID = Imu_SensorId_e::IMUSensor_Head;

        _accelGyroFixedMsg = std::make_shared<AccelerometerGyroMessage>();
        _mgrPtr->AddPublishSubscribeMessage("AccelerometerGyroFixedMsg", _accelGyroFixedMsg);
        _accelGyroFixedMsg->IMU_SensorID = Imu_SensorId_e::IMUSensor_Fixed;

        _loggingControlMsg = std::make_shared<ImageLoggingControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage("ImageLoggingControlMessage", _loggingControlMsg);

        _imuResponseMessageMsg = std::make_shared<IMUCommandResponseMessage>();
        _mgrPtr->AddPublishSubscribeMessage("IMUResponseMessage", _imuResponseMessageMsg);

        _headOrientationControlMsg = make_shared<HeadOrientationControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage("HeadOrientationControlMessage", _headOrientationControlMsg);


        _dataRecorder.setDirectory(DataLogDirectory);

        string fn = config->GetConfigStringValue("IMUComm.DataLogBaseFilename", "IMUDataLog");
        _dataRecorder.setBaseFilename(fn);
        _dataRecorder.setHeaderRecord(&_dataRecorderStdHeader);

        _headOrientationRecord.HeadOrientationMsg = _headOrientationMsg;
        _accelGyroRecord.AccelGyroMsg = _accelGyroFixedMsg;

        EnableHeadOrientationLogging = config->GetConfigBoolValue("IMUComm.EnableHeadOrientationLogging", true);
        EnableAccelGyroLogging = config->GetConfigBoolValue("IMUComm.EnableAccelGyroLogging", true);

        SendHighSpeedIMUDataOut = config->GetConfigBoolValue("IMUComm.SendHighSpeedIMUDataOut", false);

        //Set up the _acccelerometerGyroMsgPool
        //We temporarity need to create a AccelerometerGyroMessage Message required to setup the pool.
        //The message is only needed for the setup process and can then be discarded.
        AccelerometerGyroMessage agMsg;
        _accelerometerGyroMsgPool = unique_ptr<MessagePool>(new MessagePool(&agMsg, AccelGyroMsgPoolSize + 1));

        //Set Max Input Accel/Gyro Value... use to ensure valid data from IMU
        // The numbers are on the high value... mainly to ensure garbage data is not received.
        _maxAccelGyroInputVals.Ax = 100.0;  //Meters/sec^2
        _maxAccelGyroInputVals.Ay = 100.0;  //Meters/sec^2
        _maxAccelGyroInputVals.Az = 100.0;  //Meters/sec^2
        _maxAccelGyroInputVals.Gx = 10.0;   //Radians/sec
        _maxAccelGyroInputVals.Gy = 10.0;   //Radians/sec
        _maxAccelGyroInputVals.Gz = 10.0;   //Radians/sec
        _maxAccelGyroInputVals.tstampSec = 24.0 * 3600.0;  //1 day

     }

    bool IMU_RxMessageParser::Initialize()
    {
        bool error = false;
        //Get local copies of the following message queues... keeps from always having
        //to look these queues up... since they have high usage.
        _headAccelGyroMsgCount = 0;
        _fixedAccelGyroMsgCount = 0;

        try
        {
            HeadOrientationIMUMsgRxQueue = _mgrPtr->GetManagerMessageQueue<std::shared_ptr<RabitMsgPtrSPSCQueue>>("HeadOrientationIMUMsgRxQueue");
            HeadOrientationIMUEmptyMsgQueue = _mgrPtr->GetManagerMessageQueue<std::shared_ptr<RabitMsgPtrSPSCQueue>>("HeadOrientationIMUEmptyMsgQueue");
            VehicleStateIMUMsgRxQueue = _mgrPtr->GetManagerMessageQueue<std::shared_ptr<RabitMsgPtrSPSCQueue>>("VehicleStateIMUMsgRxQueue");
            VehicleStateIMUEmptyMsgQueue = _mgrPtr->GetManagerMessageQueue<std::shared_ptr<RabitMsgPtrSPSCQueue>>("VehicleStateIMUEmptyMsgQueue");
        }
        catch (MessageNotRegisteredException &e)
        {
            LOGERROR("IMU_RxMessageParser: The IMU message queue does not exist.");
            cout << "IMU_RxMessageParser: The IMU message queue does not exist." << endl;
            error = true;
        }
        return error;
    }


    void IMU_RxMessageParser::Shutdown()
    {
        _dataRecorder.closeLogFile();
    }

    bool IMU_RxMessageParser::parseOnOff(char* val)
    {
        bool On = false;
        if(val[1] == 'N')
            On = true;
        return On;
    }

    void IMU_RxMessageParser::ReturnEmptyMsgsToPool()
    {
        //Check return empty queues and return all messaages to the message pool.
        //Note: at this time all messages are assumed to be of the same type...
        //If at a later time... different types of messages are in the empy queues,
        //then the messages must be sorted by type and returned to the proper pool...
        //otherwize all hell brakes loose.
        RabitMessage *emptyMsg;
        while( HeadOrientationIMUEmptyMsgQueue->GetMessage(emptyMsg))
        {
            _accelerometerGyroMsgPool->CheckInMessage(emptyMsg);
        }
        while( VehicleStateIMUEmptyMsgQueue->GetMessage(emptyMsg))
        {
            _accelerometerGyroMsgPool->CheckInMessage(emptyMsg);
        }
    }

    AccelerometerGyroMessage* IMU_RxMessageParser::CheckoutAccelerometerGyroMsgFromPool()
    {
        AccelerometerGyroMessage* agMsg = nullptr;
        RabitMessage *emptyMsgPtr = _accelerometerGyroMsgPool->CheckOutMessage();
        if (emptyMsgPtr != nullptr)
        {
            agMsg = static_cast<AccelerometerGyroMessage *>(emptyMsgPtr);
        }
        else
        {
            LOGERROR("IMU_RxMessageParser AccelerometerGyro Message Pool is Empty!");
        }
        return agMsg;
    }


    void IMU_RxMessageParser::setBinaryFlagsForSensor(Imu_SensorId_e sId, bool *fixedSenFlag, bool *headSenFlag, bool bVal)
    {
        switch(sId)
        {
            case IMUSensor_NA:
                break;		//No change
            case IMUSensor_Fixed:
                *fixedSenFlag = bVal;
                break;
            case IMUSensor_Head:
                *headSenFlag = bVal;
                break;
            case IMUSensor_Both:
                *fixedSenFlag = bVal;
                *headSenFlag = bVal;
                break;
        }
    }

    bool IMU_RxMessageParser::getBinaryFlagForSensor(Imu_SensorId_e sId, bool fixedSenFlag, bool headSenFlag)
    {
        bool bVal = false;
        switch(sId)
        {
            case IMUSensor_NA:
                break;		//No change
            case IMUSensor_Fixed:
                bVal = fixedSenFlag;
                break;
            case IMUSensor_Head:
                bVal = headSenFlag;
                break;
            case IMUSensor_Both:
                bVal = fixedSenFlag;	//Assumes both are the same.
                break;
        }
        return bVal;
    }

    void IMU_RxMessageParser::setIntValForSensor(Imu_SensorId_e sId, int *fixedVal, int *headVal, int value)
    {
        switch(sId)
        {
            case IMUSensor_NA:
                break;		//No change
            case IMUSensor_Fixed:
                *fixedVal = value;
                break;
            case IMUSensor_Head:
                *headVal = value;
                break;
            case IMUSensor_Both:
                *fixedVal = value;
                *headVal = value;
                break;
        }
    }

    int IMU_RxMessageParser::getIntValForSensor(Imu_SensorId_e sId, int fixedSenFlag, int headSenFlag)
    {
        int value = 0;
        switch(sId)
        {
            case IMUSensor_NA:
                break;		//No change
            case IMUSensor_Fixed:
                value = fixedSenFlag;
                break;
            case IMUSensor_Head:
                value = headSenFlag;
                break;
            case IMUSensor_Both:
                value = fixedSenFlag;	//Assumes both are the same.
                break;
        }
        return value;
    }


    Imu_SensorId_e IMU_RxMessageParser::parseImuSensorID(int sId)
    {
        Imu_SensorId_e sensorID = IMUSensor_NA;
        if(sId >= 0 && sId < 4)
        {
            sensorID = (Imu_SensorId_e)sId;
        }
        return sensorID;
    }



    void IMU_RxMessageParser::rxMsgHandler(dtiUtils::SerialCommMessage_t &msg)
    {
        char* pCmd = _receiveRS232CommMsgBuf;
        char* pValue;
        int msgSize = msg.getMsg((u_char *)_receiveRS232CommMsgBuf);
        pValue = strchr(pCmd, '=');
        if(MgrIsRunning && msgSize > 4 && pValue != NULL)
        {
            int cmdSize = pValue - pCmd;
            ++pValue;
            if(pCmd[0] >= 'A' && pCmd[0] <= 'Z')
            {
                //This is a command response from the IMU
                parseCmdResponse(pCmd, pValue);
            }
            else
            {
                ++_totalNumberImuDataMsgs;
                if(checkMsgCRCOk(pCmd, msgSize))
                {
                    parseIMUData(pCmd, pValue);
                }
                else
                {
                    ++_totalNumberImuDataMsgCrcErrors;
                    if(_totalNumberImuDataMsgCrcErrors < 10
                            || _totalNumberImuDataMsgCrcErrors % 100 == 0)
                    {
                        LOGWARN("IMU Msg CRC Error: " << _receiveRS232CommMsgBuf
                             <<  " NoMsgs=" << _totalNumberImuDataMsgs
                        << " NoCRC Errs=" << _totalNumberImuDataMsgCrcErrors);
                    }
                }
            }
        }
        else
        {
            LOGWARN("Invalid IMU Msg: " << _receiveRS232CommMsgBuf);
        }
    }

    //A 16-bit CRC is the last three characters in B64 format on the message
    //string.
    bool IMU_RxMessageParser::checkMsgCRCOk(char* pCmd, int msgSize)
    {
        bool ok = false;
        msgSize -= 3;
        if(msgSize > 0)
        {
            uint16_t msgCRC = base64ToUInt16(pCmd + msgSize);
            uint16_t cmpCRC = Compute_CRC16(pCmd, msgSize);
            ok = msgCRC == cmpCRC;
        }
        return ok;
    }


    void IMU_RxMessageParser::parseCmdResponse(char *pCmd, char *pValue)
    {
        int8_t NoDigits;
        uint8_t smValue;
        uint16_t lgValue;
        int16_t lgiValue;
        bool bVal;
        int intVal = 0;
        Imu_SensorId_e sensorID = IMUSensor_NA;
        bool questionMark = false;
        int cmdNval = -1;

        int cmdSize = pValue - pCmd - 1;
        if(cmdSize < 3 || cmdSize > 4)
        {
            LOGWARN("Invalid IMU Msg: " << pCmd);
            return;
        }
        if(pValue[0] == '?')
        {
            //This should not occur
            LOGWARN("Invalid IMU Msg: " << pCmd);
            return;
        }
        if(cmdSize == 4)
        {
            cmdNval = pCmd[3] - '0';
            sensorID = parseImuSensorID(pCmd[3]);
        }

        switch(pCmd[0])
        {
            case 'A':
                if(sensorID != IMUSensor_NA)
                {
                    if(pCmd[1] == 'G' && pCmd[2] == 'E')  //AGEn=ON/OFF/?    This enables or disables the Accelerometer/Gyro sensor at the sensor
                    {
                        bVal = parseOnOff(pValue);
                        //setBinaryFlagsForSensor(sensorID, &SensorEnabledCmd_Fixed, &SensorEnabledCmd_Head, bVal);
                    }
                    else if(pCmd[1] == 'O' && pCmd[2] == 'R')  //AORn=ON/OFF/?    Accelerometer Raw Data Transmit
                    {
                        bVal = parseOnOff(pValue);
                    }
                    else if(pCmd[1] == 'O' && pCmd[2] == 'P')  //AOPn=ON/OFF/?    Accelerometer Processed Data Transmit
                    {
                        bVal = parseOnOff(pValue);
                    }
                    else if(pCmd[1] == 'O' && pCmd[2] == 'S')
                    {
                        bVal = parseOnOff(pValue);
                    }
                    else if(pCmd[1] == 'F' && pCmd[2] == 'S')  //AFSn=n  Accel Full-scale setting
                    {
                        intVal = atoi(pValue);
                        intVal =  intVal < 0 ? 0 : intVal > 3 ? 3 : intVal;
                    }
                    else if(pCmd[1] == 'G' && pCmd[2] == 'R')   //ADRn=nnn   Accelerometer / Gyro Data Rate Divider
                    {
                        intVal = atoi(pValue);
                        intVal =  intVal < 0 ? 0 : intVal > 3 ? 3 : intVal;
                    }
                    else if(pCmd[1] == 'L' && pCmd[2] == 'P')
                    {
                        intVal = atoi(pValue);
                        intVal =  intVal < 0 ? 0 : intVal > 3 ? 3 : intVal;
                    }
                }
                break;

            case 'B':
                if(sensorID != IMUSensor_NA)
                {
                    if(pCmd[1] == 'O' && pCmd[2] == 'R')  //AORn=ON/OFF/?    Accelerometer Raw Data Transmit
                    {
                        bVal = parseOnOff(pValue);
                    }
                    else if(pCmd[1] == 'O' && pCmd[2] == 'P')  //AOPn=ON/OFF/?    Accelerometer Processed Data Transmit
                    {
                        bVal = parseOnOff(pValue);
                    }
                    else if(pCmd[1] == 'T' && pCmd[2] == 'S')
                    {
                        bVal = parseOnOff(pValue);
                    }
                }
                break;

            case 'D':
                if(pCmd[1] == 'O' && pCmd[2] == 'T')  //AORn=ON/OFF/?    Accelerometer Raw Data Transmit
                {
                    bVal = pValue[0] == 'F' ? true : false;
                }
                else if(pCmd[1] == 'S' && pCmd[2] == 'N')  //AOPn=ON/OFF/?    Accelerometer Processed Data Transmit
                {
                    intVal = atoi(pValue);
                }
                else if(pCmd[1] == 'T' && pCmd[2] == 'S')
                {
                    bVal = parseOnOff(pValue);
                }
                break;

            case 'G':
                if(sensorID != IMUSensor_NA)
                {
                    if(pCmd[1] == 'O' && pCmd[2] == 'R')  //AORn=ON/OFF/?    Accelerometer Raw Data Transmit
                    {
                        bVal = parseOnOff(pValue);
                    }
                    else if(pCmd[1] == 'O' && pCmd[2] == 'P')  //AOPn=ON/OFF/?    Accelerometer Processed Data Transmit
                    {
                        bVal = parseOnOff(pValue);
                    }
                    else if(pCmd[1] == 'O' && pCmd[2] == 'S')
                    {
                        bVal = parseOnOff(pValue);
                    }
                    else if(pCmd[1] == 'F' && pCmd[2] == 'S')  //AFSn=n  Accel Full-scale setting
                    {
                        intVal = atoi(pValue);
                        intVal =  intVal < 0 ? 0 : intVal > 3 ? 3 : intVal;
                    }
                    else if(pCmd[1] == 'L' && pCmd[2] == 'P')
                    {
                        intVal = atoi(pValue);
                        intVal =  intVal < 0 ? 0 : intVal > 3 ? 3 : intVal;
                    }
                }
                break;

            case 'H':
                if(pCmd[1] == 'B' && pCmd[2] == 'T')  //Heart Beat
                {
                    intVal = atoi(pValue);
                    LOGINFO("IMU Heart Beat=" << intVal
                                              << " Fixed AG Msg Cnt: " << _fixedAccelGyroMsgCount
                                    << " Head AG Msg Cnt: " << _headAccelGyroMsgCount
                                              << " CRC Error Cnt: " << _totalNumberImuDataMsgCrcErrors);
                }
                break;

            case 'P':
                if(pCmd[1] == 'R' && pCmd[2] == 'C')  //PRC=nnn
                {
                    intVal = atoi(pValue);
                }
                else if(pCmd[1] == 'S' && pCmd[2] == 'N')  //PSN=nnn
                {
                    intVal = atoi(pValue);
                }
                else if(pCmd[1] == 'S' && pCmd[2] == 'F')  ////PSFn=ii,ffff
                {
                    //ToDo... needs work
                    intVal = atoi(pValue);
                }
                break;

            default:
                LOGWARN("Invalid IMU Msg: " << pCmd);
        }

        //Send response message out for monitoring
        //This is currenly only used for human monitoring so the use of
        //a Publish-subscibe message rather than a message Queue to the Comms Manager is ok.
        _imuResponseMessageMsg->CmdRspMsg = pCmd;
        _imuResponseMessageMsg->IMURemoteCtrlEnable = RemoteCtrlEnabled;
        _imuResponseMessageMsg->PostMessage();
     }

    void IMU_RxMessageParser::parseIMUData(char *pCmd, char *pValue)
    {
        bool logMsgChanged = _loggingControlMsg->FetchMessage();
        Imu_SensorId_e sensorID = IMUSensor_NA;
        uint16_t crcInput, crcComputed;
        if(pCmd[0] == 'd')
        {
            uint8_t bVal = B64ToByte(pCmd[1]);
            Data_Format_t dft = (Data_Format_t)(bVal & 0x0F);
            sensorID = parseImuSensorID(bVal >> 4);

            bVal = B64ToByte(pCmd[2]);
            IMU_DataType_e dataType = (IMU_DataType_e)(bVal);

            switch(dft)
            {
                case Data_Format_t::DFT_Cartesian:
                {
                    Data_Cartesian_t data;
                    data.dataType = dataType;
                    data.tstampSec = base64_48BitToTimeStampSec(pValue);
                    pValue += 8;
                    data.x = base64ToFloat(pValue);
                    pValue += 6;
                    data.y = base64ToFloat(pValue);
                    pValue += 6;
                    data.z = base64ToFloat(pValue);

                    //Test... ToDo:  There can be different message types...
                            //this assumes the this is a head orientation messsage
                     if(dataType == IMU_DataType_e::Orientation && sensorID == Imu_SensorId_e::IMUSensor_Head)
                     {
                         if( _headOrientationMsg.get() != nullptr
                             || _headOrientationMsg->GetGlobalPublishSubscribeMessageRef().get() == nullptr)
                         {
                             _headOrientationMsg->SetEulerAnglesDegrees(data.x, data.y, data.z);
                             _headOrientationMsg->IMUTimeStampSec = data.tstampSec;
                             _headOrientationMsg->PostMessage();  //Posting the message sets the timestamp.

                             if(EnableHeadOrientationLogging && _loggingControlMsg->EnableLogging)
                             {
                                 _dataRecorder.writeDataRecord(_headOrientationRecord);
                             } else if(!_loggingControlMsg->EnableLogging)
                             {
                                 _dataRecorder.closeLogFile();
                             }
                         }
                         else
                         {
                             LOGERROR("_headOrientationMsg is NULL!")
                         }
                     }
                    /***********************************************
                    std::cout << "Accel: t=" << data.tstampSec
                              << " x=" << data.x
                              << " y=" << data.y
                              << " z=" << data.z
                              << std::endl;
                    ***********************************************/
                    break;
                }
                case Data_Format_t::DFT_EulerAngles:
                {
                    Data_EulerAngles_t data;
                    data.dataType = dataType;
                    data.tstampSec = base64_48BitToTimeStampSec(pValue);
                    pValue += 8;
                    data.theta = base64ToFloat(pValue);
                    pValue += 6;
                    data.phi = base64ToFloat(pValue);
                    pValue += 6;
                    data.psi = base64ToFloat(pValue);

                    //Test
                    std::cout << "Gyro: t=" << data.tstampSec
                              << " theta=" << data.theta
                              << " phi=" << data.phi
                              << " psi=" << data.psi
                              << std::endl;

                    break;
                }
                case Data_Format_t::DFT_Quaternion:
                {
                    Data_Quaterion_t data;
                    data.dataType = dataType;
                    data.tstampSec = base64_48BitToTimeStampSec(pValue);
                    pValue += 8;
                    data.ux = base64ToFloat(pValue);
                    pValue += 6;
                    data.uy = base64ToFloat(pValue);
                    pValue += 6;
                    data.uz = base64ToFloat(pValue);
                    pValue += 6;
                    data.theta = base64ToFloat(pValue);

                    break;
                }
                case Data_Format_t::DFT_AccelGyro:
                {
                    Data_AccelGyro_t data;
                    data.dataType = dataType;
                    data.tstampSec = base64_48BitToTimeStampSec(pValue);
                    pValue += 8;
                    data.Ax = base64ToFloat(pValue);
                    pValue += 6;
                    data.Ay = base64ToFloat(pValue);
                    pValue += 6;
                    data.Az = base64ToFloat(pValue);
                    pValue += 6;
                    data.Gx = base64ToFloat(pValue);
                    pValue += 6;
                    data.Gy = base64ToFloat(pValue);
                    pValue += 6;
                    data.Gz = base64ToFloat(pValue);

                    if(IsDataTypeAccelGyro(dataType)
                       && (sensorID == Imu_SensorId_e::IMUSensor_Head || sensorID == Imu_SensorId_e::IMUSensor_Fixed))
                    {
                        if(isValidAccelGyroData(data, _maxAccelGyroInputVals))
                        {
                            AccelerometerGyroMessage *accelGyroOutMsgPtr = CheckoutAccelerometerGyroMsgFromPool();
                            std::shared_ptr<AccelerometerGyroMessage> accelGyroMsg = _accelGyroFixedMsg;
                            if(sensorID == Imu_SensorId_e::IMUSensor_Head)
                            {
                                accelGyroMsg = _accelGyroHeadMsg;
                                ++_headAccelGyroMsgCount;
                            } else
                            {
                                ++_fixedAccelGyroMsgCount;
                            }
                            accelGyroMsg->IMU_SensorID = sensorID;
                            accelGyroMsg->IMUTimeStampSec = data.tstampSec;
                            accelGyroMsg->SetAccelerationRates(data.Ax, data.Ay, data.Az);
                            accelGyroMsg->SetGyroAngularRatesRadPerSec(data.Gx, data.Gy, data.Gz);
                            accelGyroMsg->PostMessage();

                            if(accelGyroOutMsgPtr != nullptr)
                            {
                                accelGyroOutMsgPtr->CopyMessage(accelGyroMsg.get());
                                if(sensorID == Imu_SensorId_e::IMUSensor_Head)
                                {
                                    if(!HeadOrientationIMUMsgRxQueue->AddMessage(accelGyroOutMsgPtr))
                                    {
                                        //Message queue is full... return message to pool
                                        LOGWARN("HeadOrientationIMUMsgRxQueue is Full.");
                                        _accelerometerGyroMsgPool->CheckInMessage(accelGyroOutMsgPtr);
                                    }
                                } else
                                {
                                    if(!VehicleStateIMUMsgRxQueue->AddMessage(accelGyroOutMsgPtr))
                                    {
                                        //Message queue is full... return message to pool
                                        LOGWARN("VehicleStateIMUMsgRxQueue is Full.");
                                        _accelerometerGyroMsgPool->CheckInMessage(accelGyroOutMsgPtr);
                                    }
                                }
                            }

                            _headOrientationControlMsg->FetchMessage();
                            if(SendHighSpeedIMUDataOut
                               && _headOrientationControlMsg->HeadOrientationOutputSelect !=
                                  HeadOrientationOutputSelect_e::NoOutput)
                            {
                                std::shared_ptr<AccelerometerGyroMessage> agOutMsg;
                                agOutMsg = std::make_shared<AccelerometerGyroMessage>();
                                agOutMsg->CopyMessage(accelGyroMsg.get());
                                auto rmsgPtr = dynamic_pointer_cast<Rabit::RabitMessage, AccelerometerGyroMessage>(
                                        agOutMsg);
                                _mgrPtr->AddMessageToQueue("ZMQPublishOutMsgQueue", rmsgPtr);
                            }

                            if(EnableAccelGyroLogging && _loggingControlMsg->EnableLogging)
                            {
                                _accelGyroRecord.AccelGyroMsg = accelGyroMsg;
                                _dataRecorder.writeDataRecord(_accelGyroRecord);
                            } else if(!_loggingControlMsg->EnableLogging)
                            {
                                _dataRecorder.closeLogFile();
                            }
                        }
                        else
                        {
                            if(sensorID == Imu_SensorId_e::IMUSensor_Head)
                            {
                                ++_headAccelGyroInvalidDataCount;
                                if(_headAccelGyroInvalidDataCount < 10 ||
                                   _headAccelGyroInvalidDataCount % 100 == 0)
                                {
                                    LOGWARN("Invalid Accel/Gyro Data from IMU Head: " << _headAccelGyroInvalidDataCount);
                                }
                            }
                            else
                            {
                                ++_fixedAccelGyroInvalidDataCount;
                                if(_fixedAccelGyroInvalidDataCount < 10 ||
                                        _fixedAccelGyroInvalidDataCount % 100 == 0)
                                {
                                    LOGWARN("Invalid Accel/Gyro Data from IMU Fixed: " << _fixedAccelGyroInvalidDataCount);
                                }
                            }
                        }
                    }
                    break;
                }


            }

        }
        else
        {
            LOGWARN("Invalid IMU Msg: " << pCmd);
        }


    }



}