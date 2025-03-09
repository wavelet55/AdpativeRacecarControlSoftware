/* ****************************************************************
 * Athr(s): Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: Jan. 13, 2018
 *
 * NeuroGroove IMU Communications Interface
 *******************************************************************/

#include "IMUCommManager.h"
#include "Base64Conversions.h"
#include "FileUtils.h"


using namespace std;
using namespace dtiRS232Comm;
using namespace IMU_SensorNS;

namespace videre
{

    //Default/Dummy Message Handler.
    void IMURecieveMsgHandler(dtiUtils::SerialCommMessage_t &msg, void *parserObj)
    {
        IMU_RxMessageParser *msgParser;
        if(parserObj != nullptr)
        {
            msgParser = (IMU_RxMessageParser *)parserObj;
            msgParser->rxMsgHandler(msg);
        }
    }



    IMUCommManager::IMUCommManager(std::string name,
                                   std::shared_ptr<ConfigData> config)
             : IMUCommManagerWSRMgr(name),
              _rs232Comm(),
              _imuRxMessageParser(this, config),
              _fixedIMUCalData(),
              _headIMUCalData()
    {
        this->SetWakeupTimeDelayMSec(25);
        _config_sptr = config;

        _rs232CommStarted = false;
        _imuRxMessageParser.MgrIsRunning = false;

        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        //Messages

        _imuCommandMessage = std::make_shared<IMUCommandResponseMessage>();
        AddPublishSubscribeMessage("IMUCommandMessage", _imuCommandMessage);

    }


    void IMUCommManager::Initialize()
    {
        LOGINFO("IMUCommManager: Initialization Started")

        bool parserSetupError = _imuRxMessageParser.Initialize();

        string commPort = _config_sptr->GetConfigStringValue("IMUComm.CommPort", "/dev/ttyUSB0");
        int baudRate = _config_sptr->GetConfigIntValue("IMUComm.BaudRate", 921600);
        int numBits = _config_sptr->GetConfigIntValue("IMUComm.NumberOfBits", 8);

        _IMUStartupSequenceState = 0;

        _rs232Comm.MessageProcessType = RS232Comm_MessageProcessType::RS232CommMPT_TextCmds;
        _rs232Comm.setRxMsgQueueSize(100);
        _rs232Comm.setMaxRxBufferSize(1024);
        _rs232Comm.setMaxRxMessageSize(IMU_MAXMESSAGESIZE);
        _rs232Comm.ReceiveMessageHandler = IMURecieveMsgHandler;
        _rs232Comm.RegisterReceivedMessageTrigger(boost::bind(&IMUCommManager::WakeUpManagerEH, this));
        _rs232Comm.setBaudRate(baudRate);
        _rs232Comm.setNumberOfBits(numBits);
        _rs232Comm.CommPort = commPort;

        _txMsgStopwatch.reset();
        _txMsgStopwatch.start();

        _imuCalDir = _config_sptr->GetConfigStringValue("IMUComm.IMUCalDirectory", "IMUCalData");
        _imuFixedCalFilename  = _config_sptr->GetConfigStringValue("IMUComm.IMUFixedCalFilename", "IMUFixedCalData.ini");
        _imuHeadCalFilename  = _config_sptr->GetConfigStringValue("IMUComm.IMUHeadCalFilename", "IMUHeadCalData.ini");
        if(_imuCalDir.length() > 0)
        {
            //Create the directory if it does not exist.
            VidereFileUtils::CreateDirectory(_imuCalDir);
            //Combine the Directory name with the Filenames
            _imuFixedCalFilename = _imuCalDir + "/" + _imuFixedCalFilename;
            _imuHeadCalFilename = _imuCalDir + "/" + _imuHeadCalFilename;
        }

        //Read the IMU Cal Data
        if( _fixedIMUCalData.ReadIMUCalibrationFromIniFile(_imuFixedCalFilename) )
        {
            LOGWARN("Could not read Fixed IMU Cal Data File: " << _imuFixedCalFilename);
        }
        if( _headIMUCalData.ReadIMUCalibrationFromIniFile(_imuHeadCalFilename) )
        {
            LOGWARN("Could not read Fixed IMU Cal Data File: " << _imuHeadCalFilename);
        }

        LOGINFO("IMUCommManager: Initialization Complete");
        std::cout << "IMUCommManager: Initialization Complete" << std::endl;
    }


    void IMUCommManager::Startup()
    {
        _IMUStartupSequenceState = 0;
        _IMUStartupComplete = false;
        //Wait to start RS-232 Receive thread until the main thread is started...
        //otherwise the RS-232 receieve queue may fillup fast
        if(_rs232Comm.start())
        {
            LOGINFO("IMU RS232 Faild to Start");
            std::cout << "IMU RS232 Faild to Start" << std::endl;
        }
        else
        {
            LOGINFO("IMU RS232 Started Up OK");
            std::cout << "IMU RS232 Started Up OK" << std::endl;
            _rs232CommStarted = true;
            _imuRxMessageParser.MgrIsRunning = true;
        }
        _txMsgStopwatch.reset();
        _txMsgStopwatch.start();
    }



    bool IMUCommManager::IMU_StartupControlSequence()
    {
        bool startupComplete = false;
        char cmdBuf[128];
        int ival;
        double dval;
        bool bval;
        IMUCalParameter_t calParam;

        _txMsgStopwatch.captureTime();
        double tsec = _txMsgStopwatch.getTimeElapsed();

        if(tsec > 0.100)
        {

            //ToDo:  In another round verify settings by feedback from the IMU.
            switch(_IMUStartupSequenceState)
            {
                case 0:
                    //Send disable to IMU incase it is currently running while the setup is occurring
                    _imuTxMsgFormatter.AccelGyroEnable(cmdBuf, Imu_SensorId_e::IMUSensor_Head, false, false);
                    _rs232Comm.transmitMessage(cmdBuf);
                    _imuTxMsgFormatter.AccelGyroEnable(cmdBuf, Imu_SensorId_e::IMUSensor_Fixed, false, false);
                    _rs232Comm.transmitMessage(cmdBuf);
                    _txMsgStopwatch.reset();
                    _txMsgStopwatch.start();
                    ++_IMUStartupSequenceState;
                    break;

                case 1:
                    //Send disable to IMU incase it is currently running while the setup is occurring
                    _imuTxMsgFormatter.BothAGRawOutputEnable(cmdBuf, Imu_SensorId_e::IMUSensor_Fixed, false, false);
                    _rs232Comm.transmitMessage(cmdBuf);
                    _imuTxMsgFormatter.BothAGRawOutputEnable(cmdBuf, Imu_SensorId_e::IMUSensor_Head, false, false);
                    _rs232Comm.transmitMessage(cmdBuf);
                    _txMsgStopwatch.reset();
                    _txMsgStopwatch.start();
                    ++_IMUStartupSequenceState;
                    break;

                case 2:
                    ival = _config_sptr->GetConfigIntValue("IMUFixedConfig.AccelFullScale", 1);
                    ival = ival < 0 ? 0 : ival > 3 ? 3 : ival;
                    _imuTxMsgFormatter.AccelSetFullScale(cmdBuf, Imu_SensorId_e::IMUSensor_Fixed, ival, false);
                    _rs232Comm.transmitMessage(cmdBuf);
                    _txMsgStopwatch.reset();
                    _txMsgStopwatch.start();
                    ++_IMUStartupSequenceState;
                    break;

                case 3:
                    ival = _config_sptr->GetConfigIntValue("IMUFixedConfig.GyroFullScale", 1);
                    ival = ival < 0 ? 0 : ival > 3 ? 3 : ival;
                    _imuTxMsgFormatter.GyroSetFullScale(cmdBuf, Imu_SensorId_e::IMUSensor_Fixed, ival, false);
                    _rs232Comm.transmitMessage(cmdBuf);
                    _txMsgStopwatch.reset();
                    _txMsgStopwatch.start();
                    ++_IMUStartupSequenceState;
                    break;

                case 4:
                    ival = _config_sptr->GetConfigIntValue("IMUFixedConfig.RateDivider", 3);
                    ival = ival < 0 ? 0 : ival > 10 ? 10 : ival;
                    _imuTxMsgFormatter.AccelGyroDataRateDivider(cmdBuf, Imu_SensorId_e::IMUSensor_Fixed, ival, false);
                    _rs232Comm.transmitMessage(cmdBuf);
                    _txMsgStopwatch.reset();
                    _txMsgStopwatch.start();
                    ++_IMUStartupSequenceState;
                    break;

                case 5:
                    ival = _config_sptr->GetConfigIntValue("IMUHeadConfig.AccelFullScale", 1);
                    ival = ival < 0 ? 0 : ival > 3 ? 3 : ival;
                    _imuTxMsgFormatter.AccelSetFullScale(cmdBuf, Imu_SensorId_e::IMUSensor_Head, ival, false);
                    _rs232Comm.transmitMessage(cmdBuf);
                    _txMsgStopwatch.reset();
                    _txMsgStopwatch.start();
                    ++_IMUStartupSequenceState;
                    break;

                case 6:
                    ival = _config_sptr->GetConfigIntValue("IMUHeadConfig.GyroFullScale", 1);
                    ival = ival < 0 ? 0 : ival > 3 ? 3 : ival;
                    _imuTxMsgFormatter.GyroSetFullScale(cmdBuf, Imu_SensorId_e::IMUSensor_Head, ival, false);
                    _rs232Comm.transmitMessage(cmdBuf);
                    _txMsgStopwatch.reset();
                    _txMsgStopwatch.start();
                    ++_IMUStartupSequenceState;
                    break;

                case 7:
                    ival = _config_sptr->GetConfigIntValue("IMUHeadConfig.RateDivider", 3);
                    ival = ival < 0 ? 0 : ival > 10 ? 10 : ival;
                    _imuTxMsgFormatter.AccelGyroDataRateDivider(cmdBuf, Imu_SensorId_e::IMUSensor_Head, ival, false);
                    _rs232Comm.transmitMessage(cmdBuf);
                    _txMsgStopwatch.reset();
                    _txMsgStopwatch.start();
                    ++_IMUStartupSequenceState;
                    break;

                case 8:
                    //Process type 0 -->  output Calibrated Accel and Gyro values
                    _imuTxMsgFormatter.SetProcessType(cmdBuf, 0);
                    _rs232Comm.transmitMessage(cmdBuf);
                    _txMsgStopwatch.reset();
                    _txMsgStopwatch.start();
                    ++_IMUStartupSequenceState;
                    break;

                case 9:
                    //Assumes process type 0
                    bval = _config_sptr->GetConfigBoolValue("IMUFixedConfig.Enabled", true);
                    if(bval)
                    {
                        _imuTxMsgFormatter.BothAGRawOutputEnable(cmdBuf, Imu_SensorId_e::IMUSensor_Fixed, true, false);
                        _rs232Comm.transmitMessage(cmdBuf);
                        _txMsgStopwatch.reset();
                        _txMsgStopwatch.start();
                    }
                    ++_IMUStartupSequenceState;
                    break;

                case 10:
                    //Assumes process type 0
                    bval = _config_sptr->GetConfigBoolValue("IMUHeadConfig.Enabled", true);
                    if(bval)
                    {
                        _imuTxMsgFormatter.BothAGRawOutputEnable(cmdBuf, Imu_SensorId_e::IMUSensor_Head, true, false);
                        _rs232Comm.transmitMessage(cmdBuf);
                        _txMsgStopwatch.reset();
                        _txMsgStopwatch.start();
                    }
                    ++_IMUStartupSequenceState;
                    break;

                case 11:
                    bval = _config_sptr->GetConfigBoolValue("IMUFixedConfig.Enabled", true);
                    if(bval)
                    {
                        _imuTxMsgFormatter.AccelGyroEnable(cmdBuf, Imu_SensorId_e::IMUSensor_Head, true, false);
                        _rs232Comm.transmitMessage(cmdBuf);
                    }

                    bval = _config_sptr->GetConfigBoolValue("IMUHeadConfig.Enabled", true);
                    if(bval)
                    {
                        _imuTxMsgFormatter.AccelGyroEnable(cmdBuf, Imu_SensorId_e::IMUSensor_Fixed, true, false);
                        _rs232Comm.transmitMessage(cmdBuf);
                    }
                    _txMsgStopwatch.reset();
                    _txMsgStopwatch.start();
                    ++_IMUStartupSequenceState;
                    break;

                case 12:
                    //Load Fixed Accel. Cal Factors
                    _fixedIMUCalData.ResetCalParamIdx();
                    bval = _config_sptr->GetConfigBoolValue("IMUFixedConfig.UseCalData", true);
                    if(bval)
                    {
                        ++_IMUStartupSequenceState;
                    }
                    else
                    {
                        //Skip to the Head Cal Data
                        _imuTxMsgFormatter.CalibrationEnable(cmdBuf, Imu_SensorId_e::IMUSensor_Fixed, false, false);
                        _rs232Comm.transmitMessage(cmdBuf);
                        _txMsgStopwatch.reset();
                        _txMsgStopwatch.start();
                        _IMUStartupSequenceState += 3;
                    }
                    break;

                case 13:
                    //Load Fixed Accel. Cal Factors
                    calParam = _fixedIMUCalData.GetNextAccelerometerCalParameter();
                    if(calParam.idx >= 0)
                    {
                        _imuTxMsgFormatter.SetParameterValue(cmdBuf, (int)ICPS_FixedAccelPSet, calParam.idx, calParam.pVal, false);
                        _rs232Comm.transmitMessage(cmdBuf);
                        _txMsgStopwatch.reset();
                        _txMsgStopwatch.start();
                    }
                    else
                    {
                        //we are done with the AccelerometerCalParameters
                        _fixedIMUCalData.ResetCalParamIdx();
                        ++_IMUStartupSequenceState;
                    }
                    break;
                case 14:
                    //Load Fixed Gyro. Cal Factors
                    calParam = _fixedIMUCalData.GetNextGyroCalParameter();
                    if(calParam.idx >= 0)
                    {
                        _imuTxMsgFormatter.SetParameterValue(cmdBuf, (int)ICPS_FixedGyroPSet, calParam.idx, calParam.pVal, false);
                        _rs232Comm.transmitMessage(cmdBuf);
                        _txMsgStopwatch.reset();
                        _txMsgStopwatch.start();
                    }
                    else
                    {
                        //we are done with the GyroCalParameters
                        _imuTxMsgFormatter.CalibrationEnable(cmdBuf, Imu_SensorId_e::IMUSensor_Fixed, true, false);
                        _rs232Comm.transmitMessage(cmdBuf);
                        _txMsgStopwatch.reset();
                        _txMsgStopwatch.start();
                        ++_IMUStartupSequenceState;
                    }
                    break;
                case 15:
                    //Load Fixed Accel. Cal Factors
                    _headIMUCalData.ResetCalParamIdx();
                    bval = _config_sptr->GetConfigBoolValue("IMUHeadConfig.UseCalData", true);
                    if(bval)
                    {
                        ++_IMUStartupSequenceState;
                    }
                    else
                    {
                        //Skip to the Head Cal Data
                        _imuTxMsgFormatter.CalibrationEnable(cmdBuf, Imu_SensorId_e::IMUSensor_Head, false, false);
                        _rs232Comm.transmitMessage(cmdBuf);
                        _txMsgStopwatch.reset();
                        _txMsgStopwatch.start();
                        _IMUStartupSequenceState += 3;
                    }
                    break;

                case 16:
                    //Load Head Accel. Cal Factors
                    calParam = _headIMUCalData.GetNextAccelerometerCalParameter();
                    if(calParam.idx >= 0)
                    {
                        _imuTxMsgFormatter.SetParameterValue(cmdBuf, (int)ICPS_HeadAccelPSet, calParam.idx, calParam.pVal, false);
                        _rs232Comm.transmitMessage(cmdBuf);
                        _txMsgStopwatch.reset();
                        _txMsgStopwatch.start();
                    }
                    else
                    {
                        //we are done with the AccelerometerCalParameters
                        _headIMUCalData.ResetCalParamIdx();
                        ++_IMUStartupSequenceState;
                    }
                    break;
                case 17:
                    //Load Fixed Gyro. Cal Factors
                    calParam = _headIMUCalData.GetNextGyroCalParameter();
                    if(calParam.idx >= 0)
                    {
                        _imuTxMsgFormatter.SetParameterValue(cmdBuf, (int)ICPS_HeadGyroPSet, calParam.idx, calParam.pVal, false);
                        _rs232Comm.transmitMessage(cmdBuf);
                        _txMsgStopwatch.reset();
                        _txMsgStopwatch.start();
                    }
                    else
                    {
                        //we are done with the GyroCalParameters
                        _imuTxMsgFormatter.CalibrationEnable(cmdBuf, Imu_SensorId_e::IMUSensor_Head, true, false);
                        _rs232Comm.transmitMessage(cmdBuf);
                        _txMsgStopwatch.reset();
                        _txMsgStopwatch.start();
                        ++_IMUStartupSequenceState;
                    }
                    break;

                default:
                    startupComplete = true;
            }
        }
        return startupComplete;
    }



    void IMUCommManager::ExecuteUnitOfWork()
    {
        char cmdBuf[128];

        if( !_IMUStartupComplete )
        {
            _IMUStartupComplete = IMU_StartupControlSequence();
        }

        _rs232Comm.processReceivedMessages(&_imuRxMessageParser);

        //Return all empty messages by to the pool
        _imuRxMessageParser.ReturnEmptyMsgsToPool();

        if( _imuCommandMessage->FetchMessage() )
        {
            _imuRxMessageParser.RemoteCtrlEnabled = _imuCommandMessage->IMURemoteCtrlEnable;
            int cmdLen = _imuCommandMessage->CmdRspMsg.length();
            if( cmdLen > 4 && cmdLen < IMU_MAXMESSAGESIZE - 2 )
            {
                Rabit::cpyStringToBuf(cmdBuf, 0, IMU_MAXMESSAGESIZE, _imuCommandMessage->CmdRspMsg.c_str(), true);
                _rs232Comm.transmitMessage(cmdBuf);
            }
        }

    }

    void IMUCommManager::Shutdown()
    {
        _imuRxMessageParser.Shutdown();
        _rs232Comm.shutdown();
        LOGINFO("IMUCommManager shutdown");
    }

}
