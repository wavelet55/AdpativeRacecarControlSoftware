/* ****************************************************************
 * Athr(s): Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: Jan. 13, 2018
 *
 * NeuroGroove IMU Communications Interface
 *******************************************************************/


#include "RS232Comm.h"
#include "CRC_Calculator.h"
#include <memory.h>

using namespace std;
using namespace boost::asio;
using namespace dtiUtils;

namespace dtiRS232Comm
{

    RS232Comm::RS232Comm(receiveMessageHandler_t rxMsgHandler,
                         RS232Comm_MessageProcessType msgProcType)
        : _rs232Port(_boostIOctx),
          _rxMsgQueue()
    {
        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("utls");
        log4cpp_->setAdditivity(false);
        ReceiveMessageHandler = rxMsgHandler;
        MessageProcessType = msgProcType;
    }

    RS232Comm::~RS232Comm()
    {
        shutdown();
    }

    //Start the background thread which runs until
    //shutdown.
    bool RS232Comm::start()
    {
        bool error = true;
        _shutdown = false;
        if(!setupRxBuffersAndQueue())
        {
            resetProcessMsgState();
            closeCommPort();        //Ensure it is closed ... reset port.
            error = openCommPort(CommPort, _baudRate);
            if(!error)
                _backgroundRxThread = std::thread(&RS232Comm::receiveRS232CommThread, this);
        }
        return error;
    }

    //Shutdown including stopping the background thread.
    void RS232Comm::shutdown()
    {
        //Closing the comm port should cause the
        //blocking read to stop.
        if(!_shutdown)
        {
            _shutdown = true;
            closeCommPort();
            if(_backgroundRxThread.joinable())
                _backgroundRxThread.join();
            releaseRxBuffersAndQueue();
        }
    }


    //Returns false if open... false if error opening the comm port.
    bool RS232Comm::openCommPort(string &commPort, int baudRate, bool logResults)
    {
        bool error = false;
        try
        {
            if(!_rs232Port.is_open())
            {
                if(commPort.size() > 0)
                {
                    CommPort = commPort;
                }
                if(baudRate > 0)
                {
                    setBaudRate(baudRate);
                }

                //It looks like the comm port has to be open before
                //setting the parameter.
                _rs232Port.open(CommPort);

                _rs232Port.set_option(serial_port_base::baud_rate(_baudRate));

                //It is a good idea to use 1.5 or 2 stop bits.  This helps ensure
                //minor clock rates between systems do not add up to causing
                //errors.  This only effects the transmit process.
                //_rs232Port.set_option(serial_port_base::stop_bits(serial_port_base::stop_bits::onepointfive));
                        //(serial_port_base::stop_bits::type)NumberOfStopBits));

                //_rs232Port.set_option(serial_port_base::character_size(_numberOfBits));

                //_rs232Port.set_option(serial_port_base::flow_control(
                //        (serial_port_base::flow_control::type)FlowControl));

                //_rs232Port.set_option(serial_port_base::parity(
                //        serial_port_base::parity::none));

                error = !_rs232Port.is_open();
                if (logResults)
                {
                    if (!error)
                    {
                        LOGINFO("RS232 Comm Port is Open: " << commPort);
                        LOGINFO("RS232 BaudRate = " << baudRate);
                    }
                    else
                    {
                        LOGERROR("Error Opening RS232 Comm Port: " << commPort);
                    }
                }
            }
        }
        catch(std::exception &e)
        {
            LOGERROR("Error Opening RS232 Comm Port: " <<  commPort
                                                     << " Exception: " << e.what() );
            error = true;
        }
        return error;
    }

    void RS232Comm::closeCommPort()
    {
        if(_rs232Port.is_open())
        {
            _rs232Port.cancel();
            _rs232Port.close();
        }
    }

    bool RS232Comm::setupRxBuffersAndQueue()
    {
        bool error = true;
        //Cannot Change Buffers if the Comm Port is Open.
        if(!_rs232Port.is_open())
        {
            if(_rxBufsSetup)
                releaseRxBuffersAndQueue();
            try
            {
                int max = MAXRXBUFSIZE;
                _rxBufferSize = _rxBufferSize < 128 ? 128 : _rxBufferSize > max ? max : _rxBufferSize;
                _readBufferPtr = new u_char[_rxBufferSize];

                //Setup the Message Queue.
                _rxMsgQueue.init(_msgQueueSize);
                for(int i = 0; i < _msgQueueSize; i++)
                {
                    _rxMsgQueue.getQueuePtr()[i].init(_maxRxMsgSize);
                }

                _rxBufsSetup = true;
                error = false;
            }
            catch(std::exception &e)
            {
                LOGERROR("Error settin up RS-232 Rx Buffers And Queue : " << "Exception: " << e.what() );
                error = true;
            }
        }
        return error;
    }

    void RS232Comm::releaseRxBuffersAndQueue()
    {
        if(_rxBufsSetup)
        {
            if (_readBufferPtr != nullptr)
            {
                delete[] _readBufferPtr;
                _readBufferPtr = nullptr;
            }
            for (int i = 0; i < _msgQueueSize; i++)
            {
                _rxMsgQueue.getQueuePtr()[i].releaseMem();
            }
        }
        _rxBufsSetup = false;
    }

    //Messages typically should be terminated with a carriage return "\r"
    //This method does change the message or terminate the message.
    bool RS232Comm::transmitMessage(std::string msg)
    {
        bool error = true;
        if(_rs232Port.is_open())
        {
            try
            {
                _rs232Port.write_some(buffer(msg.c_str(), msg.size()));
                error = false;
            }
            catch(std::exception &e)
            {
                LOGERROR("Error writing to IMU Comm Port: " << "Exception: " << e.what() );
                error = true;
            }
        }
        return error;
    }

    //Messages typically should be terminated with a carriage return "\r"
    //This method does change the message or terminate the message.
    bool RS232Comm::transmitMessage(u_char *msgBuf, int msgSize)
    {
        bool error = true;
        if(_rs232Port.is_open())
        {
            try
            {
                _rs232Port.write_some(buffer(msgBuf, msgSize));
                error = false;
            }
            catch(std::exception &e)
            {
                LOGERROR("Error writing to IMU Comm Port: " << "Exception: " << e.what() );
                error = true;
            }
        }
        return error;
    }

    //The Fast DDS Message is expected to be in the msg Buffer
    //offset by the Fast DDS Header Size (16-bytes).  The message
    //size is the size of the serialized Fast DDS Message.
    //The transmitFastDDSMessage fills in the Fast DDS Header
    //with the appropriate values including the message ID.
    bool RS232Comm::transmitFastDDSMessage(std::shared_ptr<Rabit::RabitMessage> fastDDSMsg, uint8_t topicID)
    {
        bool error = false;
        uint16_t msgCCRC;
        int hdrSize = getSizeOfFastDDSHeader();
        //Leave room for the fastDDS Header when serializing the message.
        int msgSize = fastDDSMsg->Serialize((uint8_t *)_txMsgSerBuf + hdrSize,
                                            MaxTxMsgSize - hdrSize, 2);
        if(msgSize > 0)
        {
            msgCCRC = Rabit::Compute_FastDDSCRC16((uint8_t const *)_txMsgSerBuf + hdrSize, msgSize);
            _txMsgSerBuf[0] = '>';
            _txMsgSerBuf[1] = '>';
            _txMsgSerBuf[2] = '>';
            _txMsgSerBuf[3] = topicID;
            _txMsgSerBuf[4] = 1;
            _txMsgSerBuf[5] = (uint8_t) ((msgSize >> 8) & 0xFF);
            _txMsgSerBuf[6] = (uint8_t) (msgSize & 0xFF);
            _txMsgSerBuf[7] = (uint8_t) ((msgCCRC >> 8) & 0xFF);
            _txMsgSerBuf[8] = (uint8_t) (msgCCRC & 0xFF);

            for (int i = 9; i < hdrSize; i++) {
                _txMsgSerBuf[i] = 0;
            }
            return transmitMessage((u_char *) _txMsgSerBuf, msgSize + hdrSize);
        }
        return error;
    }

    int RS232Comm::processReceivedMessages(void *parserObj, int maxNumberOfMsgsToProcess)
    {
        int noMsgProc = 0;
        if(maxNumberOfMsgsToProcess <= 0 )
            maxNumberOfMsgsToProcess = _msgQueueSize;

        while(!_shutdown && !_rxMsgQueue.IsQueueEmpty()
            && maxNumberOfMsgsToProcess > 0)
        {
             try
            {
                SerialCommMessage_t &msg = _rxMsgQueue.getTailItemReference();
                if(ReceiveMessageHandler != nullptr)
                {
                    //Process the message... this is a blocking
                    //call until the message is processed.
                    //The _rxMsgQueue tail cannot be incremented unil
                    //the ReceiveMessageHandler is done with the message
                    //and its internal memory.

                    //Test... loop back message.
                    //transmitMessage(msg.msgPtr.get(), msg.MsgSize);

                    ReceiveMessageHandler(msg, parserObj);
                    ++noMsgProc;
                }
                _rxMsgQueue.IncrementTail();
            }
            catch (std::exception &e)
            {
                LOGERROR("Error processing received RS232 Message: " << e.what());
                noMsgProc = -1;
                maxNumberOfMsgsToProcess = 0;
                _rxMsgQueue.IncrementTail();
            }
            --maxNumberOfMsgsToProcess;
        }

        return noMsgProc;
    }


    //The RS232 receive process will be handled in a separate
    //thread.  This keeps the blocking receive function from
    //stopping the primary process. A synchronous queue is used
    //to pass receive messages back to the thread the RS232Comm
    //is running on... all messages will be processed in the
    //RS232Comm thread.
    void RS232Comm::receiveRS232CommThread()
    {
        _backgroundRxThreadIsRunning = true;
        resetProcessMsgState();
        int errorCount = 0;
        //Run until shutdown.
        while(!_shutdown && errorCount < 10)
        {
            try
            {
                if(_rs232Port.is_open())
                {
                    int noBytes = (int) boost::asio::read(_rs232Port,
                                                          boost::asio::buffer(_readBufferPtr, _rxBufferSize - 1),
                                                          boost::asio::transfer_at_least(_minNumberBytesToRead));
                    if(noBytes > 0)
                    {
                        _readBufferPtr[noBytes] = 0;
                        processReadBuffer(noBytes);
                        //LOGINFO("RS232 RX Size : Buf " << noBytes << " :" << (char *)_readBufferPtr);
                    }
                    else
                    {
                        //Something is wrong with the read process...
                        //force the number of Bytes to read to be the min value.
                        _minNumberBytesToRead = 1;
                        LOGWARN("Warn receiveRS232CommThread: Empty Read." );
                    }
                }
                else
                {
                   //Wait for a little time for the comm port to be opened.
                    std::this_thread::sleep_for(std::chrono::milliseconds(250));
                }
                errorCount = 0;
            }
            catch(std::exception &e)
            {
                LOGERROR("Error receiveRS232CommThread: " << e.what() );
                ++errorCount;
            }

        }
        _backgroundRxThreadIsRunning = false;
    }

    //This should only be called when not receiving messages.
    void RS232Comm::resetProcessMsgState()
    {
        _minNumberBytesToRead = MessageProcessType == RS232Comm_MessageProcessType::RS232CommMPT_FastDDS ?
                getSizeOfFastDDSHeader() : 1;
        _currentReadBufOffset = 0;
        _currentMsgOffset = 0;
        _rdTextBufState = RdTextBufState_e::RTBS_start;
        _rdFastDDSBufState = RdFastDDSBufState_e::RFDDSBS_start;
    }


    //Runs on the  RS232 receive process thread
    void RS232Comm::processReadBuffer(int noBytes)
    {
        if(MessageProcessType == RS232Comm_MessageProcessType::RS232CommMPT_TextCmds)
        {
            processTextReadBuffer(noBytes);
        }
        else if(MessageProcessType == RS232Comm_MessageProcessType::RS232CommMPT_FastDDS)
        {
            processFastDDSReadBuffer(noBytes);
        }
        else if(MessageProcessType == RS232Comm_MessageProcessType::RS232CommMPT_BinaryMT1)
        {
            LOGERROR("Error: Unsupported RS232Comm_MessageProcessType: BinaryMT1");
        }
        else
        {
            LOGERROR("Error: Unsupported RS232Comm_MessageProcessType: Unknown");
        }
    }


    int RS232Comm::findNextMsgEndPoint(int bufOffset, int maxOffset)
    {
        int endPt = -1;
        maxOffset = maxOffset >  _rxBufferSize ? _rxBufferSize : maxOffset;
        for(int i = bufOffset; i < maxOffset; i++)
        {
            char c = (char)_readBufferPtr[i];
            if(IsMsgEndPoint(c))
            {
                endPt = i;
                break;
            }
        }
        return endPt;
    }

    int RS232Comm::findNextMsgStartPoint(int bufOffset, int maxOffset)
    {
        int startPt = -1;
        maxOffset = maxOffset >  _rxBufferSize ? _rxBufferSize : maxOffset;
        for(int i = bufOffset; i < maxOffset; i++)
        {
            char c = (char)_readBufferPtr[i];
            if(c != '\r' && c != '\n' && c != ' ' && c != '\t' && c != 0)
            {
                startPt = i;
                break;
            }
        }
        return startPt;
    }


    //Runs on the  RS232 receive process thread
    //A text message will be all ascii characters terminated in
    //either a carriage return "/r" or newline char: "/n"
    //The Read Buffer can have 0 or more messages contained in it
    //and the last message does not have to be complete:
    //readBuf: "midMsg\r\nNextMsg\nLastPartialMsg"
    //All message bytes in the readBuf must be processed before returning.
    void RS232Comm::processTextReadBuffer(int noBytes)
    {
        int rxBufOffset = 0;
        int startPt = 0;
        int endPt = 0;
        int msgNoBytes = 0;
        bool endOfBufFound = false;
        while(!endOfBufFound)
        {
            switch (_rdTextBufState)
            {
                case RdTextBufState_e::RTBS_start:
                    _currentMsgOffset = 0;
                    _rdTextBufState = RdTextBufState_e::RTBS_findstartofmsg;
                   //break;

                case RdTextBufState_e::RTBS_findstartofmsg:
                    startPt = findNextMsgStartPoint(rxBufOffset, noBytes);
                    if(startPt >= 0)
                    {
                        rxBufOffset = startPt;
                        _rdTextBufState = RdTextBufState_e::RTBS_findendofmsg;
                    }
                    else
                    {
                        //No message in buffer
                        _minNumberBytesToRead = _minRxMsgSize;
                        _rdTextBufState = RdTextBufState_e::RTBS_start;
                        endOfBufFound = true;
                    }
                    break;

                case RdTextBufState_e::RTBS_findendofmsg:
                    if(!_rxMsgQueue.IsQueueFull())
                    {
                        endPt = findNextMsgEndPoint(rxBufOffset, noBytes);
                        if (endPt >= rxBufOffset)
                        {
                            msgNoBytes = endPt - rxBufOffset;
                            if (msgNoBytes > 0)
                            {
                                //There is more to the message... add it in.
                                _rxMsgQueue.getHeadItemReference().addMsg(_readBufferPtr,
                                                                          msgNoBytes,
                                                                          rxBufOffset,
                                                                          _currentMsgOffset);
                            }
                            //This is a complete message... so inc. queue
                            if (_rxMsgQueue.IncrementHead())
                            {
                                LOGWARN("RS232 Rx Queue is Full!");
                            }
                            rxBufOffset = endPt + 1;
                            _rdTextBufState = RdTextBufState_e::RTBS_start;
                            _sigMsgReceivedTrigger();
                        } else
                        {
                            //We reached the end of the read buffer... add partial message and quit.
                            msgNoBytes = noBytes - rxBufOffset;
                            if (msgNoBytes > 0)
                            {
                                _rxMsgQueue.getHeadItemReference().addMsg(_readBufferPtr,
                                                                          msgNoBytes,
                                                                          rxBufOffset,
                                                                          _currentMsgOffset);
                                _currentMsgOffset += msgNoBytes;
                                //This is a partial message... so don't inc. queue head.
                            }
                            //We don't know how many bytes are required to finish the
                            //message... so go small
                            _minNumberBytesToRead = 1;
                            endOfBufFound = true;
                        }
                    }
                    else
                    {
                        LOGWARN("RS232 Rx Queue is Full!");
                        //Not the best idea... buf hold off until
                        //the queue has been processed.
                        _rdTextBufState = RdTextBufState_e::RTBS_start;
                        endOfBufFound = true;
                        _sigMsgReceivedTrigger();
                    }
                    break;
            }
        }
    }

    //The start of each message is:  ">>>"
    //Returns buffer index.  A - 1 indicates the end of the buffer
    //was reached without finding the start of the header.
    int RS232Comm::findFastDDSMsgHeader(int bufOffset, int maxOffset)
    {
        int bufIdx = -1;
        _fastDDSHdrIdx = _fastDDSHdrIdx < 0 ? 0 : _fastDDSHdrIdx >= 3 ? 0 : _fastDDSHdrIdx;
        maxOffset = maxOffset >  _rxBufferSize ? _rxBufferSize : maxOffset;
        for(bufIdx = bufOffset; bufIdx < maxOffset; bufIdx++)
        {
            char c = (char)_readBufferPtr[bufIdx];
            if(c == '>')
            {
                _fastDDSHdr.hdrBuf[_fastDDSHdrIdx++] = c;
                if(_fastDDSHdrIdx >= 3)
                {
                    ++bufIdx;
                    return bufIdx;
                }
            } else {
                _fastDDSHdrIdx = 0;
           }
        }
        return bufIdx;
    }


    //The Transport Fast DDS message format coming from PX4/DTX systems
    //over a Comm port are binary messages with the format:
    //    FastDDSHeader:MessageBytes
    //    The start of each message is:  ">>>"
    void RS232Comm::processFastDDSReadBuffer(int noBytes)
    {
        int mbIdx = 0;
        int msgNoBytes = 0;
        int fddsMsgSize = 0;
        bool endOfBufFound = false;

        while(!endOfBufFound)
        {
            switch (_rdFastDDSBufState)
            {
                case RdFastDDSBufState_e::RFDDSBS_start:
                    _currentMsgOffset = 0;
                    _fastDDSHdrIdx = 0;
                    _minNumberBytesToRead = getSizeOfFastDDSHeader();
                    _rdFastDDSBufState = RdFastDDSBufState_e::RFDDSBS_findmsghdrstart;
                    //break;

                case RdFastDDSBufState_e::RFDDSBS_findmsghdrstart:
                    mbIdx = findFastDDSMsgHeader(mbIdx, noBytes);
                    if(_fastDDSHdrIdx == 3)
                    {
                        //The start of the header was found
                        _rdFastDDSBufState = RdFastDDSBufState_e::RFDDSBS_getmsghdr;
                        _currentMsgOffset = 0;
                    }
                    else if(_fastDDSHdrIdx > 0)
                    {
                        //Start of the header ">>>" found, but we are at the end
                        //of the current buffer.
                        endOfBufFound = true;
                    }
                    else
                    {
                        //No msg header found and we are at the end of the buffer.
                        _rdFastDDSBufState = RdFastDDSBufState_e::RFDDSBS_start;
                        endOfBufFound = true;
                    }
                    break;

                case RdFastDDSBufState_e::RFDDSBS_getmsghdr:
                    //Add Bytes to the header buffer until we have a full header
                    //or run out of data in the current message buffer.
                    while(mbIdx < noBytes)
                    {
                        _fastDDSHdr.hdrBuf[_fastDDSHdrIdx++] = _readBufferPtr[mbIdx++];
                        if(_fastDDSHdrIdx >= getSizeOfFastDDSHeader())
                        {
                            _rdFastDDSBufState = RdFastDDSBufState_e::RFDDSBS_getmsgbodystart;
                            break;
                        }
                    }
                    endOfBufFound = mbIdx >= noBytes;
                    if(endOfBufFound)
                    {
                        if(_rdFastDDSBufState == RdFastDDSBufState_e::RFDDSBS_getmsgbodystart)
                        {
                            int msgBodySize = getFastDDSMsgSize();
                            msgBodySize = msgBodySize < 1 ? 1 : msgBodySize;
                            int halfRxBufSize = getMaxRxBufferSize() >> 1;
                            _minNumberBytesToRead = msgBodySize > halfRxBufSize ? halfRxBufSize : msgBodySize;
                        }
                        else
                        {
                            _minNumberBytesToRead = getSizeOfFastDDSHeader() - _fastDDSHdrIdx;
                            _minNumberBytesToRead = _minNumberBytesToRead < 1 ? 1 : _minNumberBytesToRead;
                        }
                    }
                    break;

                case RdFastDDSBufState_e::RFDDSBS_getmsgbodystart:
                    msgNoBytes = noBytes - mbIdx;
                    if(msgNoBytes > 0)
                    {
                        if (!_rxMsgQueue.IsQueueFull())
                        {
                            _currentMsgOffset = 0;
                            fddsMsgSize = getFastDDSMsgSize();
                            msgNoBytes = msgNoBytes <= fddsMsgSize ? msgNoBytes : fddsMsgSize;
                            _rxMsgQueue.getHeadItemReference().MsgId = _fastDDSHdr.fddHdr.topic_ID;
                            _rxMsgQueue.getHeadItemReference().MsgSize = fddsMsgSize;

                            _rxMsgQueue.getHeadItemReference().addMsg(_readBufferPtr,
                                                                      msgNoBytes,
                                                                      mbIdx,
                                                                      _currentMsgOffset);
                            mbIdx += msgNoBytes;
                            _currentMsgOffset += msgNoBytes;
                            if(_currentMsgOffset >= fddsMsgSize)
                            {
                                //We have a complete message;
                                _rdFastDDSBufState = RdFastDDSBufState_e::RFDDSBS_verifymsgcrc;
                            }
                            else
                            {
                                //There is more to the message... we have to wait for the
                                //next buffer of data.
                                int msgBodySize = getFastDDSMsgSize() - _currentMsgOffset;
                                msgBodySize = msgBodySize < 1 ? 1 : msgBodySize;
                                int halfRxBufSize = getMaxRxBufferSize() >> 1;
                                _minNumberBytesToRead = msgBodySize > halfRxBufSize ? halfRxBufSize : msgBodySize;
                                _rdFastDDSBufState = RdFastDDSBufState_e::RFDDSBS_getmsgbodyproc;
                                endOfBufFound = mbIdx >= noBytes;
                            }

                         } else {
                            LOGWARN("RS232 Rx Queue is Full!");
                            //Not the best idea... buf hold off until
                            //the queue has been processed.
                            //We are going to loose the current message buffer of data
                            //so just start over and through away the current message.
                            _rdFastDDSBufState = RdFastDDSBufState_e::RFDDSBS_start;
                            _minNumberBytesToRead = getSizeOfFastDDSHeader();
                            endOfBufFound = true;
                            _sigMsgReceivedTrigger();
                        }
                    }
                    break;

                case RdFastDDSBufState_e::RFDDSBS_getmsgbodyproc:
                    msgNoBytes = noBytes - mbIdx;
                    if(msgNoBytes > 0)
                    {
                        fddsMsgSize = getFastDDSMsgSize();
                        msgNoBytes = fddsMsgSize - _currentMsgOffset;
                        if(msgNoBytes > 0)
                        {
                            _rxMsgQueue.getHeadItemReference().addMsg(_readBufferPtr,
                                                                      msgNoBytes,
                                                                      mbIdx,
                                                                      _currentMsgOffset);
                            mbIdx += msgNoBytes;
                            _currentMsgOffset += msgNoBytes;
                        }
                        if(_currentMsgOffset >= fddsMsgSize)
                        {
                            //We have a complete message;
                            _rdFastDDSBufState = RdFastDDSBufState_e::RFDDSBS_verifymsgcrc;
                        }
                        else
                        {
                            //There is more to the message... we have to wait for the
                            //next buffer of data.
                            int msgBodySize = getFastDDSMsgSize() - _currentMsgOffset;
                            msgBodySize = msgBodySize < 1 ? 1 : msgBodySize;
                            int halfRxBufSize = getMaxRxBufferSize() >> 1;
                            _minNumberBytesToRead = msgBodySize > halfRxBufSize ? halfRxBufSize : msgBodySize;
                            endOfBufFound = mbIdx >= noBytes;
                        }
                    }
                    break;

                case RdFastDDSBufState_e::RFDDSBS_verifymsgcrc:
                    //This is a complete message...
                    //Verify the message CRC before posting the message.
                    //If it is a bad message through it away.
                    ++_numMessagesRecieved;
                    uint16_t msgHCRC = getFastDDSMsgCRC();
                    uint16_t msgCCRC = Rabit::Compute_FastDDSCRC16((uint8_t const *)_rxMsgQueue.getHeadItemReference().msgPtr,
                                                     _rxMsgQueue.getHeadItemReference().MsgSize);
                    if(msgHCRC == msgCCRC)
                    {
                        if (_rxMsgQueue.IncrementHead())
                        {
                            LOGWARN("RS232 Rx Queue is Full!");
                        }
                        _rdFastDDSBufState = RdFastDDSBufState_e::RFDDSBS_start;
                        _sigMsgReceivedTrigger();
                    }
                    else
                    {
                        //This is an invalid message based on the CRC.
                        //Through message away... don't post it.
                        LOGWARN("Invalid Fast DDS Msg, CRCs do not Match! MsgID=" << _fastDDSHdr.fddHdr.topic_ID);
                        ++_numInvalidMessages;
                        _rdFastDDSBufState = RdFastDDSBufState_e::RFDDSBS_start;
                    }
                    _minNumberBytesToRead = getSizeOfFastDDSHeader();
                    endOfBufFound = mbIdx >= noBytes;
                    break;
            }
        }
    }

    int RS232Comm::testOnlyAddMsgToRxBuf(char *msgBuf, int nobytes, int rxBufOffset )
    {
        int msgSize = 0;
        int spaceAvialble = _rxBufferSize - rxBufOffset;
        msgSize = nobytes > spaceAvialble ? spaceAvialble : nobytes;
        for(int i = 0; i < msgSize; i++)
            _readBufferPtr[rxBufOffset + i] = msgBuf[i];

        msgSize = rxBufOffset + msgSize;
        return msgSize;
    }

    int RS232Comm::testOnlyGetMsgFromRxQueue(char *msgBuf, bool incTail)
    {
        int msgSize = 0;
        msgSize = _rxMsgQueue.getTailItemReference().getMsg((u_char*)msgBuf);
        if(incTail)
            _rxMsgQueue.IncrementTail();
        return msgSize;
    }

}