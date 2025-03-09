/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: June 2018
 *
  *******************************************************************/

#ifndef VIDERE_DEV_DATARECORDERABSTRACTRECORD_H
#define VIDERE_DEV_DATARECORDERABSTRACTRECORD_H

#include "global_defines.h"
#include <chrono>
#include <SystemTimeClock.h>

namespace VidereUtils
{
#define DATARECORDERTYPESIZETERMSIZE 6


    //When Adding new record types add to the end of this enum...
    //otherwise this could break the proper reading of log records
    //that have been created in the past.
    enum DataRecordType_e
    {
        DRT_Header,
        IMURT_AccelGyro,
        IMURT_HeadOrientation,
        SPRT_SipnPuffVals,
        GPSRD_GPS_Header,
        GPSRT_GPS_Data,
        KTLA_Brake_Cmd,
        KTLA_Brake_Status,
        KTLA_Throttle_Cmd,
        KTLA_Throttle_Status,
        EPAS_Steering_Cmd,
        EPAS_Steering_Status,
        DRT_TrackHeadOrientation,
        DRT_SystemControl,

    };


    //An Abstract class for a Record...
    //One "line" of data that is recorded in a file.
    //The data to be recorded can be part of the concrete class
    //or it can be passed in as an object or structure.
    class DataRecorderAbstractRecord
    {
    public:
        //The record type is stored in each record line.
        //This number is set by the user to specify the specific
        //data record type.
        //This value should be set by the designer in the concreate constructor.
        DataRecordType_e RecordType = DataRecordType_e::DRT_Header;

        //TimeStamp in seconds... all records
        //must have a timestamp associated with the recorded data.
        //Within the system the timestamps must be synchronous so that
        //recorded data can be ordered by time.
        double TimeStampSec;

    protected:
        //The maximum size of the serialized record.
        //This will be know at design time for a given Data Record Type.
        //An actual record can be smaller than the max size... but not larger.
        //This value should be set by the designer in the concreate constructor.
        uint32_t _maxRecordSize = 0;

        //Contains the current/last serialized record size.
        uint32_t _currentSerializedRecordSize = 0;

        //A Buffer that will contain the RecordType, the record size,
        // and a termination: '='
        char _typeSizeTermBuf[DATARECORDERTYPESIZETERMSIZE + 1];


    public:
        DataRecorderAbstractRecord() {}

        uint8_t GetRecordType() { return RecordType; }

        uint32_t GetMaxRecordSize() { return _maxRecordSize; }

        //Get the Current SerializedRecordSize
        uint32_t GetCurrentSerializedRecordSize() { return _currentSerializedRecordSize; }

        double TimeStampSeconds() { return TimeStampSec; }

        void SetTimeNow()
        {
            //_timeStamp = std::chrono::high_resolution_clock::now();
            TimeStampSec = Rabit::SystemTimeClock::GetSystemTimeClock()->GetCurrentGpsTimeInSeconds();
        }

        void SetTimeStampSeconds(double ts)
        {
            TimeStampSec = ts;
        }

        char* getRecordTypeAndLenghtItems(int recordLength = -1);

        //Get a pointer to the Serialized Data Record
        virtual char* GetSerializedDataRecordPtr() = 0;

        //Serialize data and put results into the local buffer;
        //Returns the number of bytes of the serialized data.
        virtual char* serializeDataRecord(uint32_t *recordSizeOut) = 0;


        //Deserialized the data buffer into a data structure.
        //Returns true if error, false otherwise.
        virtual bool deserialzedDataRecord(char* dataBufPtr) = 0;

        //Read the data record from the file.  It is assumed the file is set to the
        // start of the record and the record header has been read.
        //Returns true if error, false otherwise.
        virtual bool readDataRecordFromFile(std::ifstream &logFile, int recordSize) = 0;

    };


    std::shared_ptr<DataRecorderAbstractRecord> generateDataRecord(DataRecordType_e recordType);


}
#endif //VIDERE_DEV_DATARECORDERABSTRACTRECORD_H
