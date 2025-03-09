/* ****************************************************************
 * Athr(s): Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: Jan. 13, 2018
 *
 * NeuroGroove IMU Communications Interface
 *******************************************************************/


#ifndef VIDERE_DEV_RS232COMM_H
#define VIDERE_DEV_RS232COMM_H

#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <boost/asio.hpp>
#include <boost/signals2.hpp>
#include "logger.h"
#include "LockFreeQueue.h"
#include "SerialCommMessage.h"
#include "RabitMessage.h"

namespace dtiRS232Comm
{
    //Must Match the Boost enum:
    //boost::asco::serial_port_base::stop_bits
    enum RS232Comm_NumberOfStopBits_e
    {
        RS232CommNSB_one,
        RS232CommNSB_onePointFive,
        RS232CommNSB_two
    };

    //Must Match the Boost enum:
    //boost::asco::serial_port_base::flow_control
    enum RS232Comm_FlowControl
    {
        RS232CommFC_none,
        RS232CommFC_software,
        RS232CommFC_hardware,
     };

    enum RS232Comm_MessageProcessType
    {
        RS232CommMPT_TextCmds,
        RS232CommMPT_FastDDS,
        RS232CommMPT_BinaryMT1
    };


    class RS232Comm
    {

    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        //RS-232 IO Port
        boost::asio::io_service _boostIOctx;
        boost::asio::serial_port _rs232Port;

        int _numberOfBits = 8;

        int _baudRate = 9600;

        bool _shutdown = false;

        std::thread _backgroundRxThread;
        bool _backgroundRxThreadIsRunning = false;

        uint32_t _numMessagesRecieved = 0;
        uint32_t _numInvalidMessages = 0;

#define MAXRXMSGSIZE 8192;
#define MAXRXBUFSIZE 8192;

        int _minRxMsgSize = 3;
        int _maxRxMsgSize = 128;
        int _rxBufferSize = MAXRXBUFSIZE;
        int _msgQueueSize = 128;

        //Buffer used for Serializing FastDDS Messages.
        static const int MaxTxMsgSize = 1024;
        char _txMsgSerBuf[MaxTxMsgSize];

        dtiUtils::LockFreeQueue<dtiUtils::SerialCommMessage_t> _rxMsgQueue;

        //A trigger that is fired when a new message is recieve.
        //This is typically used to wake-up the manager thread
        //so the manager will process the message(s).
        boost::signals2::signal<void()> _sigMsgReceivedTrigger;

        //A Receive message Handler.
        //The receive message handler will handle or process
        //the received message.  This will be processed on the
        //local thread.
        boost::signals2::signal<void()> _sigProcessReceivedMessage;

    public:

        std::string CommPort;

        RS232Comm_NumberOfStopBits_e NumberOfStopBits = RS232Comm_NumberOfStopBits_e::RS232CommNSB_onePointFive;
        RS232Comm_FlowControl FlowControl = RS232Comm_FlowControl::RS232CommFC_none;

        bool getReceiveThreadIsRunning() { return _backgroundRxThreadIsRunning;}

        int getBaudRate() { return _baudRate; }
        void setBaudRate(int value)
        {
            _baudRate = value < 300 ? 300 : value > 10000000 ? 10000000 : value;
        }

        int getNumberOfBits() { return _numberOfBits; }
        void setNumberOfBits(int value)
        {
            _numberOfBits = value < 5 ? 5 : value > 8 ? 8 : value;
        }

        int getMinRxMessageSize() { return _minRxMsgSize; }
        void setMinRxMessageSize(int value)
        {
            if(!_rs232Port.is_open())
            {
                int max = MAXRXMSGSIZE;
                _minRxMsgSize = value < 1 ? 1 : value > max ? max : value;
            }
        }

        int getMaxRxMessageSize() { return _maxRxMsgSize; }
        void setMaxRxMessageSize(int value)
        {
            if(!_rs232Port.is_open())
            {
                int max = MAXRXMSGSIZE;
                _maxRxMsgSize = value < 1 ? 1 : value > max ? max : value;
            }
        }

        int getMaxRxBufferSize() { return _rxBufferSize; }
        void setMaxRxBufferSize(int value)
        {
            if(!_rs232Port.is_open())
            {
                int max = MAXRXBUFSIZE;
                _rxBufferSize = value < 128 ? 128 : value > max ? max : value;
            }
        }

        int getRxMsgQueueSize() { return _msgQueueSize; }
        void setRxMsgQueueSize(int value)
        {
            if(!_rs232Port.is_open())
            {
                _msgQueueSize = value < 2 ? 2 : value > 1024 ? 1024 : value;
            }
        }

        uint32_t getNumberOfMessagesRecieved() { return _numMessagesRecieved; }
        uint32_t getNumberOfInvalidMessages() { return _numInvalidMessages; }

        RS232Comm_MessageProcessType MessageProcessType = RS232Comm_MessageProcessType::RS232CommMPT_TextCmds;

        //A pointer to a Receive message Handler.
        //The receive message handler will handle messages that are recived.
        //A parser object may be passed to the message handler.
        typedef void (*receiveMessageHandler_t)(dtiUtils::SerialCommMessage_t &msg,
                            void *parserObj);
        receiveMessageHandler_t ReceiveMessageHandler;

        //The Message Recived Callback Trigger can be used to trigger
        //or wakeup a manager or process to then call the
        //processReceivedMessages method.
        //Do not use this methed to process messages... it should be
        //a quick wake-up call.  It is called from within the receive
        //comm thread.
        typedef void (*messageRecievedTrigger_t)();
        messageRecievedTrigger_t MessageRecievedTrigger = nullptr;

        //Use this method with a method function something like:
        //_rs232Comm.RegisterReceivedMessageTrigger(boost::bind(&IMUCommManager::WakeUpManagerEH, this));
        void RegisterReceivedMessageTrigger(const boost::function<void()> &handler)
        {
            _sigMsgReceivedTrigger.connect(handler);
        }


        RS232Comm(receiveMessageHandler_t rxMsgHandler = nullptr,
                RS232Comm_MessageProcessType msgProcType = RS232Comm_MessageProcessType::RS232CommMPT_TextCmds);

        ~RS232Comm();


        //Start the background thread which runs until
        //shutdown.
        bool start();

        //Shutdown including stopping the background thread.
        void shutdown();

        //Use start();
        bool openCommPort(std::string &commPort, int baudRate = 0, bool logResults = true);

        void closeCommPort();

        bool setupRxBuffersAndQueue();

        void releaseRxBuffersAndQueue();

        bool transmitMessage(std::string msg);

        bool transmitMessage(u_char *msgBuf, int msgSize);

        //The Fast DDS Message is expected to be in the msg Buffer
        //offset by the Fast DDS Header Size (16-bytes).  The message
        //size is the size of the serialized Fast DDS Message.
        //The transmitFastDDSMessage fills in the Fast DDS Header
        //with the appropiate values including the message ID.
        bool transmitFastDDSMessage(std::shared_ptr<Rabit::RabitMessage> fastDDSMsg, uint8_t topicID);

        //Process the received messages on the internal message queue.
        //A parser object may be passed which will be forwarded to
        //the ReceiveMessageHandler. The ReceiveMessageHandler can
        //the cast the parserObj to the type required.
        //If maxNumberOfMsgsToProcess > 0, the number of messages
        //processed will be limited by the number given.
        //Returns the number of messages processed.
        //A return of less then zero indicates an error.
        int processReceivedMessages(void *parserObj = nullptr, int maxNumberOfMsgsToProcess = 0);

        bool IsMsgEndPoint(char c)
        {
            bool endpt = c == '\r' || c == '\n';
            return endpt;
        }

    //private:
        //The RS232 receive process will be handled in a separate
        //thread.  This keeps the blocking recieve function from
        //stopping the primary process. A syncronous queue is used
        //to pass receive messages back to the thread the RS232Comm
        //is running on... all messages will be processed in the
        //RS232Comm thread.
        void receiveRS232CommThread();


        //This method allows different systems to process the
        //read buffer in differnt ways.
        //This first way is assuming text messages terminated in
        //either carriage returns or newlines (/r or /n).
        void processReadBuffer(int noBytes);

        void resetProcessMsgState();

        int findNextMsgEndPoint(int bufOffset, int maxOffset);

        int findNextMsgStartPoint(int bufOffset, int maxOffset);

        void processTextReadBuffer(int noBytes);

        int findFastDDSMsgHeader(int bufOffset, int maxOffset);

        void processFastDDSReadBuffer(int noBytes);

        int testOnlyAddMsgToRxBuf(char *msgBuf, int nobytes, int rxBufOffset = 0);

        int testOnlyGetMsgFromRxQueue(char *msgBuf, bool incTail=true);

        //Parameters and variables for the RS232 Read Process.
        //It is assumed that the RS232Comm is being used for high-speed
        //reads.  For that reason a buffer is setup so more than one
        //byte can be read from the RS232 Comm Port at a time.

        int _minNumberBytesToRead = 1;

        int _currentMsgOffset = 0;
        int _currentReadBufOffset = 0;

        bool _rxBufsSetup = false;

        u_char *_readBufferPtr = nullptr;

        enum RdTextBufState_e
        {
            RTBS_start,
            RTBS_findendofmsg,
            RTBS_findstartofmsg
        };

        RdTextBufState_e _rdTextBufState = RdTextBufState_e::RTBS_start;


        enum RdFastDDSBufState_e
        {
            RFDDSBS_start,
            RFDDSBS_findmsghdrstart,
            RFDDSBS_getmsghdr,
            RFDDSBS_getmsgbodystart,
            RFDDSBS_getmsgbodyproc,
            RFDDSBS_verifymsgcrc
        };

        RdFastDDSBufState_e _rdFastDDSBufState = RdFastDDSBufState_e::RFDDSBS_start;

        union FastDDSHeader_t
        {
            u_char hdrBuf[16];
            struct __attribute__((packed))
            {
                char marker[3];
                uint8_t topic_ID;
                uint8_t seq;
                uint8_t payload_len_h;
                uint8_t payload_len_l;
                uint8_t crc_h;
                uint8_t crc_l;
            } fddHdr;
        };

        FastDDSHeader_t _fastDDSHdr;

        int getSizeOfFastDDSHeader()
        {
            return (int)sizeof(FastDDSHeader_t::fddHdr);
        }

        //Get the message size... this assumes
        //there a valid FastDDSMsg header stored.
        int getFastDDSMsgSize()
        {
            int size = _fastDDSHdr.fddHdr.payload_len_h << 8;
            size |= _fastDDSHdr.fddHdr.payload_len_l;
            return size;
        }

        //Get the message size... this assumes
        //there a valid FastDDSMsg header stored.
        uint16_t getFastDDSMsgCRC()
        {
            uint16_t crc = _fastDDSHdr.fddHdr.crc_h << 8;
            crc |= _fastDDSHdr.fddHdr.crc_l;
            return crc;
        }

        int _fastDDSHdrIdx = 0;

    };

}


#endif //VIDERE_DEV_RS232COMM_H
