/* ****************************************************************
 * Athr(s): Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: May, 2018
 *
 * NeuroGroove GPS Interface
 *******************************************************************/

#include "GPS_RxMessageParser.h"
#include <iostream>
#include <string>

using namespace nmea;
using namespace std;

namespace GPS_ReceiverNS
{

    GPS_RxMessageParser::GPS_RxMessageParser(Rabit::RabitManager* mgrPtr,
                                             std::shared_ptr<ConfigData> config)
            :gpsParserService(nmeaGPSParser),
             _dataRecorder(),
             _gpsDataRecord(), _dataRecorderGpsHeader("GPS Data Log", 0)
    {
        _mgrPtr = mgrPtr;
        _config_sptr = config;
        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        GPSFixMsgPtr = std::make_shared<GPSFixMessage>();
        mgrPtr->AddPublishSubscribeMessage("GPSFixMessage", GPSFixMsgPtr);

        _loggingControlMsg = std::make_shared<ImageLoggingControlMessage>();
        mgrPtr->AddPublishSubscribeMessage("ImageLoggingControlMessage", _loggingControlMsg);

        _playbackControlMsg = std::make_shared<PlaybackControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage("PlaybackControlMessage", _playbackControlMsg);

        _dataRecorder.setDirectory(DataLogDirectory);
        string fn = config->GetConfigStringValue("GPSPod.DataLogBaseFilename", "GPSDataLog");
        _dataRecorder.setBaseFilename(fn);
        _dataRecorder.setHeaderRecord(&_dataRecorderGpsHeader);
        _gpsDataRecord.GPSFixMsg = GPSFixMsgPtr;

        EnableGPSLogging = config->GetConfigBoolValue("GPSPod.EnableGPSLogging", true);
        SendGPSDataOut = config->GetConfigBoolValue("GPSPod.SendGPSDataOut", false);
    }

    void GPS_RxMessageParser::Shutdown()
    {
        _dataRecorder.closeLogFile();
    }


    void GPS_RxMessageParser::rxMsgHandler(dtiUtils::SerialCommMessage_t &msg)
    {
        NMEASentence nmea;
        bool parseError = true;
        _playbackControlMsg->FetchMessage();
        //When Playback is enabled... we stop processing incoming GPS Messages
        //The GPS Data will come from the log files.
        if(!_playbackControlMsg->EnablePlayback)
        {
            int msgSize = msg.getMsg((u_char *) _receiveRS232CommMsgBuf);
            try
            {
                std::string gpsStr(_receiveRS232CommMsgBuf, msgSize);
                parseError = nmeaGPSParser.parseGpsNmeaString(gpsStr, nmea);
                if(!parseError)
                {
                    if(gpsParserService.parseNmeaSentence(nmea))
                    {
                        processNewGpsData(gpsParserService.fix);

                        if( SendGPSDataOut )
                        {
                            std::shared_ptr<GPSFixMessage> gpsOutMsg;
                            gpsOutMsg = std::make_shared<GPSFixMessage>();
                            gpsOutMsg->CopyMessage(GPSFixMsgPtr.get());
                            auto rmsgPtr = dynamic_pointer_cast<Rabit::RabitMessage, GPSFixMessage>(gpsOutMsg);
                            _mgrPtr->AddMessageToQueue("ZMQPublishOutMsgQueue", rmsgPtr);
                        }

                        bool logMsgChanged = _loggingControlMsg->FetchMessage();
                        if(EnableGPSLogging && _loggingControlMsg->EnableLogging)
                        {
                            _dataRecorder.writeDataRecord(_gpsDataRecord);
                        }
                        else if(logMsgChanged && !_loggingControlMsg->EnableLogging)
                        {
                            _dataRecorder.closeLogFile();
                        }
                    }
                }
            }
            catch(std::exception &e)
            {
                LOGWARN("Invalid GPS NMEA Msg: " << _receiveRS232CommMsgBuf << e.what());
            }
        }
    }


    void GPS_RxMessageParser::processNewGpsData(GPSFix &gpsFix)
    {
        //LOGINFO("GPS Time:" << gpsParserService.fix.timestamp.toString());
        GPSFixMsgPtr->SetGPSFix(gpsFix);
        GPSFixMsgPtr->SetTimeNow();
        GPSFixMsgPtr->PostMessage();
    }


}