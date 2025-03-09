/* ****************************************************************
 * Athr(s): Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: May 8, 2018
 *
 * GPS Interface
 *******************************************************************/

#include "GPSManager.h"
#include "GPS_DataTypeDefs.h"
#include "GPS_RxMessageParser.h"

using namespace std;
using namespace dtiRS232Comm;
using namespace GPS_ReceiverNS;

namespace videre
{

    //Default/Dummy Message Handler.
    void GPSRecieveMsgHandler(dtiUtils::SerialCommMessage_t &msg, void *parserObj)
    {
        GPS_RxMessageParser *msgParser;
        if(parserObj != nullptr)
        {
            msgParser = (GPS_RxMessageParser *)parserObj;
            msgParser->rxMsgHandler(msg);
        }
    }



    GPSManager::GPSManager(std::string name,
                                   std::shared_ptr<ConfigData> config)
             : GPSManagerWSRMgr(name),
              _rs232Comm(),
              _gpsRxMessageParser(this, config),
              _gpsDataLogReader()
    {
        this->SetWakeupTimeDelayMSec(1000);
        _config_sptr = config;

        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        //Messages
        _playbackControlMsg = std::make_shared<PlaybackControlMessage>();
        AddPublishSubscribeMessage("PlaybackControlMessage", _playbackControlMsg);

        //This is a local message only.
        _lastPlaybackControlMsg = std::make_shared<PlaybackControlMessage>();

        GPSFixMsgPtr = std::make_shared<GPSFixMessage>();
        AddPublishSubscribeMessage("GPSFixMessage", GPSFixMsgPtr);

        EnableGPSLogging = config->GetConfigBoolValue("GPSPod.EnableGPSLogging", true);

        string fn = config->GetConfigStringValue("GPSPod.DataLogBaseFilename", "GPSDataLog");
        _gpsDataLogReader.setBaseFilename(fn);
        string playbackLogDir = config->GetConfigStringValue("GlobalParameters.PlaybackDataLogDirectory", "PlaybackDataLogs");
        _gpsDataLogReader.setDirectoryName(playbackLogDir);
        _gpsDataLogReader.LoopBackToStartOfDataRecords = false;

        _dataRecorderGpsHeaderPtr = std::make_shared<GPSDataRecordHeader>("GPS Data Log", 0);
        _gpsDataRecordPtr = std::make_shared<GPSDataRecord>();

        _gpsDataRecordPtr->GPSFixMsg = GPSFixMsgPtr;
        _gpsDataLogReader.AddRecordToRecords(_gpsDataRecordPtr);
        _gpsDataLogReader.HeaderRecordPtr = _dataRecorderGpsHeaderPtr;

    }


    void GPSManager::Initialize()
    {
        LOGINFO("GPSManager: Initialization Started")

        string commPort = _config_sptr->GetConfigStringValue("GPSPod.CommPort", "/dev/ttyUSB0");
        int baudRate = _config_sptr->GetConfigIntValue("GPSPod.BaudRate", 115200);
        int numBits = _config_sptr->GetConfigIntValue("GPSPod.NumberOfBits", 8);

        _rs232Comm.MessageProcessType = RS232Comm_MessageProcessType::RS232CommMPT_TextCmds;
        _rs232Comm.setRxMsgQueueSize(256);
        _rs232Comm.setMaxRxBufferSize(4096);
        _rs232Comm.setMaxRxMessageSize(GPS_MAXMESSAGESIZE);
        _rs232Comm.ReceiveMessageHandler = GPSRecieveMsgHandler;
        _rs232Comm.RegisterReceivedMessageTrigger(boost::bind(&GPSManager::WakeUpManagerEH, this));
        _rs232Comm.setBaudRate(baudRate);
        _rs232Comm.setNumberOfBits(numBits);
        _rs232Comm.CommPort = commPort;

        _txMsgStopwatch.reset();
        _txMsgStopwatch.start();

        LOGINFO("GPSManager: Initialization Complete");
        std::cout << "GPSManager: Initialization Complete" << std::endl;
    }

    void GPSManager::Startup()
    {
        //Don't start the RS-232 Comm until the manager thread starts.
        //The GPS Recievier is constantly senting messages that build up
        //during the Rabit manager initialization process.  Starting
        //the RS-232 comm here avoids that build up.
        if(_rs232Comm.start())
        {
            LOGINFO("GPS RS232 Faild to Start");
            std::cout << "GPS RS232 Faild to Start" << std::endl;
        }
        else
        {
            LOGINFO("GPS RS232 Started Up OK");
            std::cout << "GPS RS232 Started Up OK" << std::endl;
        }
    }

    void GPSManager::ExecuteUnitOfWork()
    {
        char cmdBuf[128];
        _rs232Comm.processReceivedMessages(&_gpsRxMessageParser);

        if(_playbackControlMsg->FetchMessage())
        {
            //The message has changed.
            if(_playbackControlMsg->EnablePlayback && !_lastPlaybackControlMsg->EnablePlayback)
            {
                _gpsDataLogReader.LoopBackToStartOfDataRecords = _playbackControlMsg->LoopBackToStartOfDataRecords;

                if(!_lastPlaybackControlMsg->EnablePlayback)
                {
                    //ensure any log files that are currently open are closed
                    //and we reset to the start of the log files.
                    _gpsDataLogReader.ResetReaderToStartOfLogFiles();

                    //Playback has been enabled... read the log files from the given directory
                    int noLogFiles = _gpsDataLogReader.GetListOfFilesFromDirectory(
                            _playbackControlMsg->DataLogDirectory);
                    if(noLogFiles > 0)
                    {
                        LOGINFO("GPS Playback found: " << noLogFiles << " GPS Log Files.");
                    } else
                    {
                        LOGWARN("GPS Playback did not find any log files!");
                    }
                }

                if(_lastPlaybackControlMsg->ResetPlayback)
                {
                    _gpsDataLogReader.ResetReaderToStartOfLogFiles();
                }


            }


        }
        //Setup Transmit of any messages ready to go out.
        //_txMsgStopwatch.captureTime();


        //double tsec = _txMsgStopwatch.getTimeElapsed();
        //_txMsgStopwatch.reset();
        //_txMsgStopwatch.start();

    }

    //Process the playback mode of operation.
    //Returns true if in the playback mode, false otherwise
    bool GPSManager::playbackProcess()
    {
        if(_playbackControlMsg->FetchMessage())
        {
            //The message has changed.
            if(_playbackControlMsg->EnablePlayback)
            {
                _gpsDataLogReader.LoopBackToStartOfDataRecords = _playbackControlMsg->LoopBackToStartOfDataRecords;

                if(!_lastPlaybackControlMsg->EnablePlayback)
                {
                    //ensure any log files that are currently open are closed
                    //and we reset to the start of the log files.
                    _gpsDataLogReader.ResetReaderToStartOfLogFiles();

                    //Playback has been enabled... read the log files from the given directory
                    int noLogFiles = _gpsDataLogReader.GetListOfFilesFromDirectory(
                            _playbackControlMsg->DataLogDirectory);
                    if(noLogFiles > 0)
                    {
                        LOGINFO("GPS Playback found: " << noLogFiles << " GPS Log Files.");
                    } else
                    {
                        LOGWARN("GPS Playback did not find any log files!");
                    }
                }

                if(_lastPlaybackControlMsg->ResetPlayback)
                {
                    _gpsDataLogReader.ResetReaderToStartOfLogFiles();
                }

            }

            //Update the _lastPlaybackControlMsg
            _lastPlaybackControlMsg->CopyMessage(_playbackControlMsg.get());
        }
        if(_playbackControlMsg->EnablePlayback && _playbackControlMsg->StartPlayback
                && !_gpsDataLogReader.IsEndOfDataRecords())
        {
            auto record = _gpsDataLogReader.ReadNextRecord();
            if(record.get() != nullptr)
            {
                //Reading a record updates the GPS Message... so post the new data
                GPSFixMsgPtr->PostMessage();
            }
        }

        return _playbackControlMsg->EnablePlayback;
    }

    void GPSManager::Shutdown()
    {
        _gpsRxMessageParser.Shutdown();
        _rs232Comm.shutdown();
        _gpsDataLogReader.closeLogFile();
    }
}
