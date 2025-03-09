/* ****************************************************************
 * Athr(s): Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: May 8, 2018
 *
 * GPS Interface
 *******************************************************************/

#ifndef VIDERE_DEV_GPSMANAGER_H
#define VIDERE_DEV_GPSMANAGER_H

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <RabitManager.h>
#include <RabitMessageQueue.h>
#include <ManagerStatusMessage.h>
#include <ManagerControlMessage.h>
#include <ManagerStats.h>
#include <ManagerStatusMessage.h>
#include "global_defines.h"
#include "all_manager_message.h"
#include "../../Utils/config_data.h"
#include "../../Utils/logger.h"
#include "../../Utils/timing.h"
#include "RS232Comm.h"
#include "SerialCommMessage.h"
#include "GPS_RxMessageParser.h"
#include "DataRecorder.h"
#include "DataLogReader.h"
#include "ImageLoggingControlMessage.h"
#include "GPSDataRecord.h"
#include "GPSDataRecordHeader.h"
#include "PlaybackControlMessage.h"

// Manually include this file that has been autogenerated
#include "GPSManagerWSRMgr.h"

using namespace Rabit;

using namespace Rabit;

namespace videre
{

    class GPSManager : public GPSManagerWSRMgr
    {

    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        std::shared_ptr<ConfigData> _config_sptr;

        dtiRS232Comm::RS232Comm _rs232Comm;

        RabitStopWatch _txMsgStopwatch;

        GPS_ReceiverNS::GPS_RxMessageParser _gpsRxMessageParser;

        DataLogReader _gpsDataLogReader;
        std::shared_ptr<GPSDataRecordHeader> _dataRecorderGpsHeaderPtr;
        std::shared_ptr<GPSDataRecord> _gpsDataRecordPtr;


        std::shared_ptr<PlaybackControlMessage> _playbackControlMsg;
        std::shared_ptr<PlaybackControlMessage> _lastPlaybackControlMsg;
        std::shared_ptr<GPSFixMessage> GPSFixMsgPtr;

        bool EnableGPSLogging = true;


    public:
        GPSManager(std::string name, std::shared_ptr<ConfigData> config);

        virtual void Initialize();

        bool openCommPort(std::string commPort, int baudRate, bool logResults = true);

        void closeCommPort();

        virtual void ExecuteUnitOfWork() final;

        virtual void Startup() final;

        virtual void Shutdown() final;

        //Process the playback mode of operation.
        //Returns true if in the playback mode, false otherwise
        bool playbackProcess();

     };

}
#endif //VIDERE_DEV_GPSMANAGER_H
