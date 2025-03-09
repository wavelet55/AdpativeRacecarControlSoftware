/* ****************************************************************
 * Athr(s): Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: July, 2018
 *
 *******************************************************************/

#include "RemoteControlManager.h"
#include "RemoteControlRxMessageParser.h"

using namespace std;
using namespace dtiRS232Comm;

namespace videre
{

    //Default/Dummy Message Handler.
    void RemoteControllerRecieveMsgHandler(dtiUtils::SerialCommMessage_t &msg, void *parserObj)
    {
        RemoteControlRxMessageParser *msgParser;
        if(parserObj != nullptr)
        {
            msgParser = (RemoteControlRxMessageParser *)parserObj;
            msgParser->rxMsgHandler(msg);
        }
    }



    RemoteControlManager::RemoteControlManager(std::string name,
                                   std::shared_ptr<ConfigData> config)
             : RemoteControlManagerWSRMgr(name),
              _rs232Comm(), _RxMessageParser(this, config)
    {
        this->SetWakeupTimeDelayMSec(10);
        _config_sptr = config;

        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);
        //Messages

    }


    void RemoteControlManager::Initialize()
    {
        LOGINFO("RemoteControlManager: Initialization Started")
        _RxMessageParser.Initialize();

        string commPort = _config_sptr->GetConfigStringValue("RemoteControl.CommPort", "/dev/ttyUSB0");
        int baudRate = _config_sptr->GetConfigIntValue("RemoteControl.BaudRate", 115200);
        int numBits = _config_sptr->GetConfigIntValue("RemoteControl.NumberOfBits", 8);

        _rs232Comm.MessageProcessType = RS232Comm_MessageProcessType::RS232CommMPT_TextCmds;
        _rs232Comm.setRxMsgQueueSize(100);
        _rs232Comm.setMaxRxBufferSize(1024);
        _rs232Comm.setMaxRxMessageSize(REMOTECONTROL_MAXMESSAGESIZE);
        _rs232Comm.ReceiveMessageHandler = RemoteControllerRecieveMsgHandler;
        _rs232Comm.RegisterReceivedMessageTrigger(boost::bind(&RemoteControlManager::WakeUpManagerEH, this));
        _rs232Comm.setBaudRate(baudRate);
        _rs232Comm.setNumberOfBits(numBits);
        _rs232Comm.CommPort = commPort;
        if(_rs232Comm.start())
        {
            LOGINFO("Remote Control RS232 Faild to Start");
            std::cout << "Remote Control RS232 Faild to Start" << std::endl;
        }
        else
        {
            LOGINFO("Remote Control RS232 Started Up OK");
            std::cout << "Remote Control RS232 Started Up OK" << std::endl;
        }

        LOGINFO("RemoteControlManager: Initialization Complete");
        std::cout << "RemoteControlManager: Initialization Complete" << std::endl;
    }




    void RemoteControlManager::ExecuteUnitOfWork()
    {
        _rs232Comm.processReceivedMessages(&_RxMessageParser);
    }

    void RemoteControlManager::Shutdown()
    {
        _rs232Comm.shutdown();
    }

}


