/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: June 2018
 *
  *******************************************************************/

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include "GPSDataRecord.h"
#include "ByteArrayReaderWriterVidere.h"

namespace videre
{

    GPSDataRecord::GPSDataRecord()
            : DataRecorderAbstractRecord()
    {
        _maxRecordSize = GPSDATAMAXRECORDSIZE;
        RecordType = DataRecordType_e::GPSRT_GPS_Data;
        GPSFixMsg = nullptr;
    }

    //Serialize data and put results into the local buffer;
    //DataPtr is a pointer to a structure or object containing the data
    //to be serialized.  If is is null, then the data to be serialized
    //is part of the concrete object.
    //Returns the number of bytes of the serialized data.
    char* GPSDataRecord::serializeDataRecord(uint32_t *recordSizeOut)
    {
        _currentSerializedRecordSize = 0;
        if(GPSFixMsg != nullptr)
        {
            VidereUtils::ByteArrayWriterVidere bw((byte *) _recordBuf, GPSDATAMAXRECORDSIZE,
                                                  Rabit::EndianOrder_e::Endian_Big);

            TimeStampSec = GPSFixMsg->GetTimeStamp();
            bw.writeDouble(TimeStampSec);  //system timestamp
            bw.writeDouble(GPSFixMsg->GPSFixData.timestamp.rawTime);  //This is the GPS Timestamp
            bw.writeDouble(GPSFixMsg->GPSFixData.latitude);
            bw.writeDouble(GPSFixMsg->GPSFixData.longitude);
            bw.writeDouble(GPSFixMsg->XYZPositionMeters.x);
            bw.writeDouble(GPSFixMsg->XYZPositionMeters.y);
            bw.writeDouble(GPSFixMsg->XYZPositionMeters.z);
            bw.writeDouble(GPSFixMsg->XYZVelocityMetersPerSec.x);
            bw.writeDouble(GPSFixMsg->XYZVelocityMetersPerSec.y);
            //Note: the GPS does not measure vertical velocity... so it is not logged.
            bw.writeDouble(GPSFixMsg->GPSFixData.horizontalDilution);
            bw.writeDouble(GPSFixMsg->GPSFixData.verticalDilution);
            bw.writeInt32(GPSFixMsg->GPSFixData.trackingSatellites);
            bw.writeByte(GPSFixMsg->GPSFixData.type);
            bw.writeByte(GPSFixMsg->GPSFixData.quality);
            bw.writeByte(GPSFixMsg->GPSFixData.status);

            _currentSerializedRecordSize = bw.Idx;
            *recordSizeOut = _currentSerializedRecordSize;
        }
        return _recordBuf;
    }


    //Deserialized the data buffer into a data structure.
    //Returns true if error, false otherwise.
    bool GPSDataRecord::deserialzedDataRecord(char* dataBufPtr)
    {
        bool error = true;
        if(GPSFixMsg.get() == nullptr)
        {
            GPSFixMsg = std::make_shared<GPSFixMessage>();
        }
        if(GPSFixMsg != nullptr)
        {
            VidereUtils::ByteArrayReaderVidere br((byte *) dataBufPtr, GPSDATAMAXRECORDSIZE,
                                            Rabit::EndianOrder_e::Endian_Big);
            TimeStampSec = br.readDouble();
            GPSFixMsg->SetTimeStamp(TimeStampSec);
            GPSFixMsg->GPSFixData.timestamp.rawTime = br.readDouble();
            GPSFixMsg->GPSFixData.latitude = br.readDouble();
            GPSFixMsg->GPSFixData.longitude = br.readDouble();
            GPSFixMsg->XYZPositionMeters.x = br.readDouble();
            GPSFixMsg->XYZPositionMeters.y = br.readDouble();
            GPSFixMsg->XYZPositionMeters.z = br.readDouble();
            GPSFixMsg->XYZVelocityMetersPerSec.x = br.readDouble();
            GPSFixMsg->XYZVelocityMetersPerSec.y = br.readDouble();
            GPSFixMsg->GPSFixData.horizontalDilution = br.readDouble();
            GPSFixMsg->GPSFixData.verticalDilution = br.readDouble();
            GPSFixMsg->GPSFixData.trackingSatellites = br.readUInt32();
            GPSFixMsg->GPSFixData.type = br.readByte();
            GPSFixMsg->GPSFixData.quality = br.readByte();
            GPSFixMsg->GPSFixData.status = br.readByte();
            error = false;
        }
        return error;
    }

    //Read the data record from the file.  It is assumed the file is set to the
    // start of the record and the record header has been read.
    //Returns true if error, false otherwise.
    bool GPSDataRecord::readDataRecordFromFile(std::ifstream &logFile, int recordSize)
    {
        bool error = true;
        try
        {
            logFile.read(_recordBuf, recordSize);
            error = deserialzedDataRecord(_recordBuf);
        }
        catch (std::exception &e)
        {
            error = true;
        }
        return error;
    }

}