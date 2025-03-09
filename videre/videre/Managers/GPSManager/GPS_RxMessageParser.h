/* ****************************************************************
 * Athr(s): Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: May, 2018
 *
 * NeuroGroove May Interface
 *******************************************************************/

#ifndef VIDERE_DEV_GPS_RXMESSAGEPARSER_H
#define VIDERE_DEV_GPS_RXMESSAGEPARSER_H

#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <memory>
#include <RabitManager.h>
#include "config_data.h"
#include "logger.h"
#include "RS232Comm.h"
#include "SerialCommMessage.h"
#include "GPS_DataTypeDefs.h"
#include "../../NemaTodeGpsParser/NMEAParser.h"
#include "../../NemaTodeGpsParser/NMEACommand.h"
#include "../../NemaTodeGpsParser/GPSService.h"
#include "../../NemaTodeGpsParser/GPSFix.h"
#include "GPSFixMessage.h"
#include "DataRecorder.h"
#include "GPSDataRecordHeader.h"
#include "ImageLoggingControlMessage.h"
#include "GPSDataRecord.h"
#include "PlaybackControlMessage.h"


using namespace videre;

namespace GPS_ReceiverNS
{

    class GPS_RxMessageParser
    {
    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        std::shared_ptr<ConfigData> _config_sptr;

        //The reference to the manager is used primarily for
        //setup purposes.
        Rabit::RabitManager* _mgrPtr;

        char _receiveRS232CommMsgBuf[GPS_MAXMESSAGESIZE];

        nmea::NMEAParser nmeaGPSParser;
        nmea::GPSService gpsParserService;


        std::shared_ptr<ImageLoggingControlMessage> _loggingControlMsg;
        std::shared_ptr<PlaybackControlMessage> _playbackControlMsg;

        DataRecorder _dataRecorder;
        GPSDataRecordHeader _dataRecorderGpsHeader;
        GPSDataRecord _gpsDataRecord;

    public:

        bool EnableGPSLogging = true;
        bool SendGPSDataOut = false;
        std::shared_ptr<GPSFixMessage> GPSFixMsgPtr;

    public:

        void Shutdown();

        GPS_RxMessageParser(Rabit::RabitManager* mgrPtr,
                            std::shared_ptr<ConfigData> config);


        void rxMsgHandler(dtiUtils::SerialCommMessage_t &msg);

        void processNewGpsData(nmea::GPSFix &gpsFix);

    };

}
#endif //VIDERE_DEV_GPS_RXMESSAGEPARSER_H
