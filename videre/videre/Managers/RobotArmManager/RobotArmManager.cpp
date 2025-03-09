/* ****************************************************************
 * Athr(s): Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: Jan. 13, 2018
 *
 * NeuroGroove IMU Communications Interface
 *******************************************************************/

#include "RobotArmManager.h"

using namespace std;
using namespace dtiRS232Comm;

namespace videre
{

    //Default/Dummy Message Handler.
    void RobotArmRecieveMsgHandler(dtiUtils::SerialCommMessage_t &msg, void *parserObj)
    {
        RobotArm_RxMessageParser *msgParser;
        if(parserObj != nullptr)
        {
            msgParser = (RobotArm_RxMessageParser *)parserObj;
            msgParser->rxMsgHandler(msg);
        }
    }



    RobotArmManager::RobotArmManager(std::string name,
                                   std::shared_ptr<ConfigData> config)
             : RobotArmManagerWSRMgr(name),
              _rs232Comm(),
              _imuRxMessageParser(this, config)
    {
        this->SetWakeupTimeDelayMSec(100);
        _config_sptr = config;

        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);
        //Messages


    }


    void RobotArmManager::Initialize()
    {
        LOGINFO("RobotArmManager: Initialization Started")

        string commPort = _config_sptr->GetConfigStringValue("RobotArm.CommPort", "/dev/ttyUSB1");
        int baudRate = _config_sptr->GetConfigIntValue("RobotArm.BaudRate", 115200);
        int numBits = _config_sptr->GetConfigIntValue("RobotArm.NumberOfBits", 8);

        _rs232Comm.MessageProcessType = RS232Comm_MessageProcessType::RS232CommMPT_TextCmds;
        _rs232Comm.setRxMsgQueueSize(100);
        _rs232Comm.setMaxRxBufferSize(1024);
        _rs232Comm.setMaxRxMessageSize(ROBOTARM_MAXMESSAGESIZE);
        _rs232Comm.ReceiveMessageHandler = RobotArmRecieveMsgHandler;
        _rs232Comm.RegisterReceivedMessageTrigger(boost::bind(&RobotArmManager::WakeUpManagerEH, this));
        _rs232Comm.setBaudRate(baudRate);
        _rs232Comm.setNumberOfBits(numBits);
        _rs232Comm.CommPort = commPort;
        if(_rs232Comm.start())
        {
            LOGINFO("Robot Arm RS232 Faild to Start");
            std::cout << "Robot Arm RS232 Faild to Start" << std::endl;
        }
        else
        {
            LOGINFO("Robot Arm RS232 Started Up OK");
            std::cout << "Robot Arm RS232 Started Up OK" << std::endl;
        }

        _txMsgStopwatch.reset();
        _txMsgStopwatch.start();

        LOGINFO("RobotArmManager: Initialization Complete");
        std::cout << "RobotArmManager: Initialization Complete" << std::endl;
    }




    void RobotArmManager::ExecuteUnitOfWork()
    {
        _rs232Comm.processReceivedMessages(&_imuRxMessageParser);

        //Setup Transmit of any messages ready to go out.
        _txMsgStopwatch.captureTime();
        double tsec = _txMsgStopwatch.getTimeElapsed();
        if(tsec > 10.0)
        {
            if(_accelEnabled)
            {
                //_rs232Comm.transmitMessage("ACE=OFF\r\n");
                _accelEnabled = false;
            }
            else
            {
                //_rs232Comm.transmitMessage("ACE=ON\r\n");
                _accelEnabled = true;
            }

            _txMsgStopwatch.reset();
            _txMsgStopwatch.start();
        }

    }

    void RobotArmManager::Shutdown()
    {
        _rs232Comm.shutdown();
    }

}


