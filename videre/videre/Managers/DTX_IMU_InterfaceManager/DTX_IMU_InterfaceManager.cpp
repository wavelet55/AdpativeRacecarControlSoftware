/* ****************************************************************
 * Athr(s): Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: Sept. 2021
 *
 * DTX IMU Interface
 *******************************************************************/

#include <chrono>
#include <thread>

#include "DTX_IMU_InterfaceManager.h"
#include "FileUtils.h"
#include <fastcdr/Cdr.h>
#include "host_conn_rep.h"
#include "SystemTimeClock.h"

using namespace std;
using namespace dtiRS232Comm;
using namespace Rabit;

namespace videre
{

    //Default/Dummy Message Handler.
    void DTXRecieveMsgHandler(dtiUtils::SerialCommMessage_t &msg, void *parserObj)
    {
        DTX_IMU_RxMessageParser *msgParser;
        if(parserObj != nullptr)
        {
            msgParser = (DTX_IMU_RxMessageParser *)parserObj;
            msgParser->rxMsgHandler(msg);
        }
    }

    DTX_IMU_InterfaceManager::DTX_IMU_InterfaceManager(std::string name,
                                   std::shared_ptr<ConfigData> config)
             : DTX_IMU_InterfaceManagerWSRMgr(name),
              _dtxRxMessageParser(this, WorkSpace(), config)
    {
        this->SetWakeupTimeDelayMSec(25);
        _config_sptr = config;

        _rs232CommStarted = false;
        _dtxRxMessageParser.MgrIsRunning = false;

        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("ctli");

        _mgrIncomingMessageQueue = std::make_shared<RabitQueue>(100, name);
        AddManagerMessageQueue("DTX_IMU_InterfaceManager", _mgrIncomingMessageQueue);
        WakeUpManagerOnEnqueue(_mgrIncomingMessageQueue);

        //WorkSpace()->DTX_IMU_InterfaceMgrStatusMsg->dtx_connection_status = 0;
        //WorkSpace()->DTX_IMU_InterfaceMgrStatusMsg->mgr_status = 0;
        //WorkSpace()->DTX_IMU_InterfaceMgrStatusMsg->PostMessage();

        _timeLastMsgReceivedSec = 0.0f;
    }


    void DTX_IMU_InterfaceManager::Initialize()
    {
        LOGINFO("DTX_IMU_InterfaceManager: Initialization Started")

        _dtxRxMessageParser.Initialize();

        LOGINFO("DTX_IMU_InterfaceManager: Initialization Complete");
    }

    void DTX_IMU_InterfaceManager::Startup()
    {
        _txMsgStopwatch.reset();
        _txMsgStopwatch.start();
    }


    void DTX_IMU_InterfaceManager::ExecuteUnitOfWork()
    {
        if( checkDTXControlBdConnection() )
        {
            SetWakeupTimeDelayMSec(25);
            _uartConnectionState = ProcessMessagesToBoard();
        }
        else
        {
            SetWakeupTimeDelayMSec(250);
        }
    }

    bool DTX_IMU_InterfaceManager::checkDTXControlBdConnection()
    {
        bool connected = false;
        /*
         * The following attempts to connect to the board every couple of seconds. If it connects it maintains a
         * connection, if it loses connection, it attempts to connect again.
         */
        static int connect_attempt = 0;
        switch(_uartConnectionState)
        {
            case CONNECTING:
                _uartConnectionState = GetNewRS232Comm();
                if(_uartConnectionState == CONNECTING)
                {
                    if (connect_attempt == 0)
                    {
                        LOGERROR("DTX_IMU_Interface RS232 Failed to CONNECT to : " << _commPort);
                    }else
                    {
                        LOGDEBUG("Connection attempt: " << connect_attempt);
                    }
                    connect_attempt++;
                    std::this_thread::sleep_for(std::chrono::seconds(2)); // wait a couple seconds
                }else{
                    connect_attempt = 0;
                    LOGINFO("DTX_IMU_Interface Comm CONNECTED");
                }
                break;

            case CONNECTED:
                connected = true;
                break;
        }
        return connected;
    }

    DTX_IMU_InterfaceManager::UartConnectionState_e DTX_IMU_InterfaceManager::GetNewRS232Comm()
    {
        _commPort = _config_sptr->GetConfigStringValue("DTX_IMU.CommPort", "/dev/ttyACM0");
        int baudRate = _config_sptr->GetConfigIntValue("DTX_IMU.BaudRate", 115200);
        int numBits = 8;

        _rs232Comm_uptr = nullptr;
        _rs232Comm_uptr = std::make_unique<dtiRS232Comm::RS232Comm>();

        _rs232Comm_uptr->MessageProcessType = RS232Comm_MessageProcessType::RS232CommMPT_FastDDS;
        _rs232Comm_uptr->setRxMsgQueueSize(100);
        _rs232Comm_uptr->setMaxRxBufferSize(1024);
        _rs232Comm_uptr->setMaxRxMessageSize(256);
        _rs232Comm_uptr->ReceiveMessageHandler = DTXRecieveMsgHandler;
        _rs232Comm_uptr->RegisterReceivedMessageTrigger(boost::bind(&DTX_IMU_InterfaceManager::WakeUpManagerEH, this));
        _rs232Comm_uptr->setBaudRate(baudRate);
        _rs232Comm_uptr->setNumberOfBits(numBits);
        _rs232Comm_uptr->CommPort = _commPort;

        //Wait to start RS-232 Receive thread until the main thread is started...
        //otherwise the RS-232 receive queue may fill up fast
        enum UartConnectionState_e state = CONNECTING;
        if(_rs232Comm_uptr->start())
        {
            // Failed to connect
            WorkSpace()->DTX_IMU_InterfaceMgrStatusMsg->dtx_connection_status = 0;
            _rs232Comm_uptr = nullptr;
            state = CONNECTING;
        }
        else
        {
            // Connected
            WorkSpace()->DTX_IMU_InterfaceMgrStatusMsg->dtx_connection_status = 1;
            _rs232CommStarted = true;
            _dtxRxMessageParser.MgrIsRunning = true;
            state = CONNECTED;
        }
        WorkSpace()->DTX_IMU_InterfaceMgrStatusMsg->PostMessage();
        return state;
    }

    DTX_IMU_InterfaceManager::UartConnectionState_e DTX_IMU_InterfaceManager::ProcessMessagesToBoard()
    {
        if (_rs232Comm_uptr->getReceiveThreadIsRunning()) {

            int noMsgsProcessed = _rs232Comm_uptr->processReceivedMessages(&_dtxRxMessageParser);
            double currentTsec = SystemTimeClock::GetSystemTimeClock()->GetCurrentGpsTimeInSeconds();
            double deltaTsec = currentTsec - _timeLastMsgReceivedSec;
            if(noMsgsProcessed > 0)
            {
                _timeLastMsgReceivedSec = currentTsec;
                WorkSpace()->DTX_IMU_InterfaceMgrStatusMsg->mgr_status = 1;
                if(deltaTsec < 2.5)
                {
                     //Verify we have received a GF360SystemStatusMsg in a reasonable period of time.
                    double dt1 = currentTsec - WorkSpace()->DTX_IMU_SysStatusMsg->GetTimeStamp();
                    double dt2 = currentTsec - WorkSpace()->IMU_SensorSetValuesMsg->GetTimeStamp();
                    if(dt1 < 2.5 || dt2 < 2.5)
                    {
                        WorkSpace()->DTX_IMU_InterfaceMgrStatusMsg->dtx_connection_status = 3;
                    }
                    else
                    {
                        WorkSpace()->DTX_IMU_InterfaceMgrStatusMsg->dtx_connection_status = 2;
                    }
                }
                else
                {
                    WorkSpace()->DTX_IMU_InterfaceMgrStatusMsg->dtx_connection_status = 1;
                }
                WorkSpace()->DTX_IMU_InterfaceMgrStatusMsg->PostMessage();
            }
            else
            {
                if(deltaTsec > 2.5
                    && WorkSpace()->DTX_IMU_InterfaceMgrStatusMsg->dtx_connection_status > 1)
                {
                    WorkSpace()->DTX_IMU_InterfaceMgrStatusMsg->dtx_connection_status = 1;
                    WorkSpace()->DTX_IMU_InterfaceMgrStatusMsg->mgr_status = 0;
                    WorkSpace()->DTX_IMU_InterfaceMgrStatusMsg->PostMessage();
                }
            }

            if( WorkSpace()->DTX_IMU_SysCmdMsg->FetchMessage())
            {
                LOGINFO("DTX_IMU_SysCmdMsg: " << WorkSpace()->DTX_IMU_SysCmdMsg->ToString());
                _rs232Comm_uptr->transmitFastDDSMessage(WorkSpace()->DTX_IMU_SysCmdMsg, 80);
            }


            //Send messages to the GF360 Control Board
            std::shared_ptr<RabitMessage> rmsg;
            while (_mgrIncomingMessageQueue->GetMessage(rmsg))
            {
                if (rmsg->GetTypeIndex() == typeid(DTX_IMU_SystemCommandMsg))
                {
                    LOGINFO("DTX_IMU_SysCmdMsg: " << rmsg->ToString());
                    _rs232Comm_uptr->transmitFastDDSMessage(rmsg, 80);
                }
                else if (rmsg->GetTypeIndex() == typeid(host_conn_rep))
                {
                    LOGDEBUG("host_conn_rep: " << rmsg->ToString());
                    _rs232Comm_uptr->transmitFastDDSMessage(rmsg, 83);
                }
            }
            return CONNECTED;
        }
        else
        {
            _rs232Comm_uptr = nullptr;
            WorkSpace()->DTX_IMU_InterfaceMgrStatusMsg->dtx_connection_status = 0;
            WorkSpace()->DTX_IMU_InterfaceMgrStatusMsg->mgr_status = 0;
            WorkSpace()->DTX_IMU_InterfaceMgrStatusMsg->PostMessage();
            return CONNECTING;
        }
    }


    void DTX_IMU_InterfaceManager::Shutdown()
    {
        _rs232Comm_uptr = nullptr;
        LOGINFO("DTX_IMU_InterfaceManager shutdown");
    }


}
