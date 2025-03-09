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
#include "IMU_DataRecord.h"
#include "ByteArrayReaderWriterVidere.h"

namespace IMU_SensorNS
{

    IMU_HeadOrientationRecord::IMU_HeadOrientationRecord()
            : DataRecorderAbstractRecord()
    {
        _maxRecordSize = IMUHEADORIENTATIONMAXRECORDSIZE;
        RecordType = DataRecordType_e::IMURT_HeadOrientation;
    }

    IMU_HeadOrientationRecord::IMU_HeadOrientationRecord(std::shared_ptr<HeadOrientationMessage> headOrientatinMsg)
        : DataRecorderAbstractRecord()
    {
        _maxRecordSize = IMUHEADORIENTATIONMAXRECORDSIZE;
        RecordType = DataRecordType_e::IMURT_HeadOrientation;
        HeadOrientationMsg = headOrientatinMsg;
    }

    //Serialize data and put results into the local buffer;
    //DataPtr is a pointer to a structure or object containing the data
    //to be serialized.  If is is null, then the data to be serialized
    //is part of the concrete object.
    //Returns the number of bytes of the serialized data.
    char* IMU_HeadOrientationRecord::serializeDataRecord(uint32_t *recordSizeOut)
    {
        _currentSerializedRecordSize = 0;
        if(HeadOrientationMsg.get() != nullptr)
        {
            VidereUtils::ByteArrayWriterVidere bw((byte *) _recordBuf, IMUHEADORIENTATIONMAXRECORDSIZE,
                                                  Rabit::EndianOrder_e::Endian_Big);

            TimeStampSec = HeadOrientationMsg->GetTimeStamp();
            bw.writeDouble(TimeStampSec);  //system timestamp
            bw.writeDouble(HeadOrientationMsg->IMUTimeStampSec);
            bw.writeDouble(HeadOrientationMsg->HeadRollPitchYawAnlges.PitchRadians());
            bw.writeDouble(HeadOrientationMsg->HeadRollPitchYawAnlges.RollRadians());
            bw.writeDouble(HeadOrientationMsg->HeadRollPitchYawAnlges.YawRadians());
            _currentSerializedRecordSize = bw.Idx;
            *recordSizeOut = _currentSerializedRecordSize;
        }
        return _recordBuf;
    }


    //Deserialized the data buffer into a data structure.
    //Returns true if error, false otherwise.
    bool IMU_HeadOrientationRecord::deserialzedDataRecord(char* dataBufPtr)
    {
        bool error = true;
        if(HeadOrientationMsg.get() == nullptr)
        {
            HeadOrientationMsg = std::make_shared<HeadOrientationMessage>();
        }
        if(HeadOrientationMsg.get() != nullptr)
        {
            VidereUtils::ByteArrayReaderVidere br((byte *) dataBufPtr, IMUHEADORIENTATIONMAXRECORDSIZE,
                                                  Rabit::EndianOrder_e::Endian_Big);
            TimeStampSec = br.readDouble();
            HeadOrientationMsg->SetTimeStamp(TimeStampSec);
            HeadOrientationMsg->IMUTimeStampSec = br.readDouble();
            HeadOrientationMsg->HeadRollPitchYawAnlges.SetPitchRadians(br.readDouble());
            HeadOrientationMsg->HeadRollPitchYawAnlges.SetRollRadians(br.readDouble());
            HeadOrientationMsg->HeadRollPitchYawAnlges.SetYawRadians(br.readDouble());
            error = false;
        }
        return error;
    }

    //Read the data record from the file.  It is assumed the file is set to the
    // start of the record and the record header has been read.
    //Returns true if error, false otherwise.
    bool IMU_HeadOrientationRecord::readDataRecordFromFile(std::ifstream &logFile, int recordSize)
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

    IMU_AccelGyroRecord::IMU_AccelGyroRecord()
            : DataRecorderAbstractRecord()
    {
        _maxRecordSize = IMUACCELGYROMAXRECORDSIZE;
        RecordType = DataRecordType_e::IMURT_AccelGyro;
        AccelGyroMsg = nullptr;
    }

    IMU_AccelGyroRecord::IMU_AccelGyroRecord(std::shared_ptr<AccelerometerGyroMessage> accelGyroMsg)
            : DataRecorderAbstractRecord()
    {
        _maxRecordSize = IMUACCELGYROMAXRECORDSIZE;
        RecordType = DataRecordType_e::IMURT_AccelGyro;
        AccelGyroMsg = accelGyroMsg;
    }

    //Serialize data and put results into the local buffer;
    //DataPtr is a pointer to a structure or object containing the data
    //to be serialized.  If is is null, then the data to be serialized
    //is part of the concrete object.
    //Returns the number of bytes of the serialized data.
    char* IMU_AccelGyroRecord::serializeDataRecord(uint32_t *recordSizeOut)
    {
        _currentSerializedRecordSize = 0;
        if(AccelGyroMsg != nullptr)
        {
            VidereUtils::ByteArrayWriterVidere bw((byte *) _recordBuf, IMUACCELGYROMAXRECORDSIZE,
                                                  Rabit::EndianOrder_e::Endian_Big);

            TimeStampSec = AccelGyroMsg->GetTimeStamp();
            bw.writeByte((byte)AccelGyroMsg->IMU_SensorID);
            bw.writeDouble(TimeStampSec);  //system timestamp
            bw.writeDouble(AccelGyroMsg->IMUTimeStampSec);
            bw.writeDouble(AccelGyroMsg->AccelerationRates.x);
            bw.writeDouble(AccelGyroMsg->AccelerationRates.y);
            bw.writeDouble(AccelGyroMsg->AccelerationRates.z);
            bw.writeDouble(AccelGyroMsg->GyroAngularRates.x);
            bw.writeDouble(AccelGyroMsg->GyroAngularRates.y);
            bw.writeDouble(AccelGyroMsg->GyroAngularRates.z);
            _currentSerializedRecordSize = bw.Idx;
            *recordSizeOut = _currentSerializedRecordSize;
        }
        return _recordBuf;
    }


    //Deserialized the data buffer into a data structure.
    //Returns true if error, false otherwise.
    bool IMU_AccelGyroRecord::deserialzedDataRecord(char* dataBufPtr)
    {
        bool error = true;
        if(AccelGyroMsg.get() == nullptr)
        {
            AccelGyroMsg = std::make_shared<AccelerometerGyroMessage>();
        }
        if(AccelGyroMsg != nullptr)
        {
            VidereUtils::ByteArrayReaderVidere br((byte *) dataBufPtr, IMUACCELGYROMAXRECORDSIZE,
                                                  Rabit::EndianOrder_e::Endian_Big);
            AccelGyroMsg->IMU_SensorID = (IMU_SensorNS::Imu_SensorId_e)br.readByte();
            TimeStampSec = br.readDouble();
            AccelGyroMsg->SetTimeStamp(TimeStampSec);
            AccelGyroMsg->IMUTimeStampSec = br.readDouble();
            AccelGyroMsg->AccelerationRates.x = br.readDouble();
            AccelGyroMsg->AccelerationRates.y = br.readDouble();
            AccelGyroMsg->AccelerationRates.z = br.readDouble();
            AccelGyroMsg->GyroAngularRates.x = br.readDouble();
            AccelGyroMsg->GyroAngularRates.y = br.readDouble();
            AccelGyroMsg->GyroAngularRates.z = br.readDouble();
            error = false;
        }
        return error;
    }

    //Read the data record from the file.  It is assumed the file is set to the
    // start of the record and the record header has been read.
    //Returns true if error, false otherwise.
    bool IMU_AccelGyroRecord::readDataRecordFromFile(std::ifstream &logFile, int recordSize)
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