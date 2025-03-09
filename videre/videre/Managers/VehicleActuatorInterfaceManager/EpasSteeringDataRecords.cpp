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
#include "EpasSteeringDataRecords.h"
#include "ByteArrayReaderWriterVidere.h"

namespace videre
{

    EpasSteeringCommandDataRecord::EpasSteeringCommandDataRecord()
            : DataRecorderAbstractRecord()
    {
        _maxRecordSize = EPASSTEERINGCMDMAXRECORDSIZE;
        RecordType = DataRecordType_e::EPAS_Steering_Cmd;
        SteeringTorqueCtrlMsg = nullptr;
    }


    //Serialize data and put results into the local buffer;
    //DataPtr is a pointer to a structure or object containing the data
    //to be serialized.  If is is null, then the data to be serialized
    //is part of the concrete object.
    //Returns the number of bytes of the serialized data.
    char* EpasSteeringCommandDataRecord::serializeDataRecord(uint32_t *recordSizeOut)
    {
        _currentSerializedRecordSize = 0;
        if(SteeringTorqueCtrlMsg != nullptr)
        {
            VidereUtils::ByteArrayWriterVidere bw((byte *) _recordBuf, EPASSTEERINGCMDMAXRECORDSIZE,
                                                  Rabit::EndianOrder_e::Endian_Big);

            TimeStampSec = SteeringTorqueCtrlMsg->GetTimeStamp();
            bw.writeDouble(TimeStampSec);  //system timestamp
            bw.writeByte(SteeringTorqueCtrlMsg->SteeringControlEnabled);
            bw.writeByte(SteeringTorqueCtrlMsg->ManualExtControl);
            bw.writeByte((byte)SteeringTorqueCtrlMsg->getSteeringTorqueMap());
            bw.writeDouble(SteeringTorqueCtrlMsg->getSteeringTorquePercent());
            _currentSerializedRecordSize = bw.Idx;
            *recordSizeOut = _currentSerializedRecordSize;
        }
        return _recordBuf;
    }


    //Deserialized the data buffer into a data structure.
    //Returns true if error, false otherwise.
    bool EpasSteeringCommandDataRecord::deserialzedDataRecord(char* dataBufPtr)
    {
        bool error = true;
        if(SteeringTorqueCtrlMsg.get() == nullptr)
        {
            SteeringTorqueCtrlMsg = std::make_shared<SteeringTorqueCtrlMessage>();
        }
        if(SteeringTorqueCtrlMsg != nullptr)
        {
            VidereUtils::ByteArrayReaderVidere br((byte *) dataBufPtr, EPASSTEERINGCMDMAXRECORDSIZE,
                                            Rabit::EndianOrder_e::Endian_Big);
            TimeStampSec = br.readDouble();
            SteeringTorqueCtrlMsg->SetTimeStamp(TimeStampSec);
            SteeringTorqueCtrlMsg->SteeringControlEnabled = br.readByte();
            SteeringTorqueCtrlMsg->ManualExtControl = br.readByte();
            SteeringTorqueCtrlMsg->setSteeringTorqueMap(br.readByte());
            SteeringTorqueCtrlMsg->setSteeringTorquePercent(br.readDouble());
            error = false;
        }
        return error;
    }

    //Read the data record from the file.  It is assumed the file is set to the
    // start of the record and the record header has been read.
    //Returns true if error, false otherwise.
    bool EpasSteeringCommandDataRecord::readDataRecordFromFile(std::ifstream &logFile, int recordSize)
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


    EpasSteeringStatusDataRecord::EpasSteeringStatusDataRecord()
            : DataRecorderAbstractRecord()
    {
        _maxRecordSize = EPASSTEERINGSTATUSMAXRECORDSIZE;
        RecordType = DataRecordType_e::EPAS_Steering_Status;
        SteeringStatusMsg = nullptr;
    }


    //Serialize data and put results into the local buffer;
    //DataPtr is a pointer to a structure or object containing the data
    //to be serialized.  If is is null, then the data to be serialized
    //is part of the concrete object.
    //Returns the number of bytes of the serialized data.
    char* EpasSteeringStatusDataRecord::serializeDataRecord(uint32_t *recordSizeOut)
    {
        _currentSerializedRecordSize = 0;
        if(SteeringStatusMsg != nullptr)
        {
            VidereUtils::ByteArrayWriterVidere bw((byte *) _recordBuf, EPASSTEERINGSTATUSMAXRECORDSIZE,
                                                  Rabit::EndianOrder_e::Endian_Big);

            TimeStampSec = SteeringStatusMsg->GetTimeStamp();
            bw.writeDouble(TimeStampSec);  //system timestamp
            bw.writeDouble(SteeringStatusMsg->MotorCurrentAmps);
            bw.writeDouble(SteeringStatusMsg->PWMDutyCyclePercent);
            bw.writeDouble(SteeringStatusMsg->MotorTorquePercent);
            bw.writeDouble(SteeringStatusMsg->SupplyVoltage);
            bw.writeDouble(SteeringStatusMsg->TempDegC);
            bw.writeDouble(SteeringStatusMsg->SteeringAngleDeg);
            bw.writeByte((byte)SteeringStatusMsg->SteeringTorqueMapSetting);
            bw.writeByte((byte)SteeringStatusMsg->SwitchPosition);
            bw.writeByte((byte)SteeringStatusMsg->TorqueA);
            bw.writeByte((byte)SteeringStatusMsg->TorqueB);
            bw.writeByte((byte)SteeringStatusMsg->ErrorCode);
            bw.writeByte((byte)SteeringStatusMsg->StatusFlags);
            bw.writeByte((byte)SteeringStatusMsg->LimitFlags);

            _currentSerializedRecordSize = bw.Idx;
            *recordSizeOut = _currentSerializedRecordSize;
        }
        return _recordBuf;
    }


    //Deserialized the data buffer into a data structure.
    //Returns true if error, false otherwise.
    bool EpasSteeringStatusDataRecord::deserialzedDataRecord(char* dataBufPtr)
    {
        bool error = true;
        if(SteeringStatusMsg.get() == nullptr)
        {
            SteeringStatusMsg = std::make_shared<DceEPASteeringStatusMessage>();
        }
        if(SteeringStatusMsg != nullptr)
        {
            VidereUtils::ByteArrayReaderVidere br((byte *) dataBufPtr, EPASSTEERINGSTATUSMAXRECORDSIZE,
                                            Rabit::EndianOrder_e::Endian_Big);
            TimeStampSec = br.readDouble();
            SteeringStatusMsg->SetTimeStamp(TimeStampSec);
            SteeringStatusMsg->MotorCurrentAmps = br.readDouble();
            SteeringStatusMsg->PWMDutyCyclePercent = br.readDouble();
            SteeringStatusMsg->MotorTorquePercent = br.readDouble();
            SteeringStatusMsg->SupplyVoltage = br.readDouble();
            SteeringStatusMsg->TempDegC = br.readDouble();
            SteeringStatusMsg->SteeringAngleDeg = br.readDouble();

            SteeringStatusMsg->SteeringTorqueMapSetting = (SteeringTorqueMap_e)br.readByte();
            SteeringStatusMsg->SwitchPosition = br.readByte();
            SteeringStatusMsg->TorqueA = br.readByte();
            SteeringStatusMsg->TorqueB = br.readByte();
            SteeringStatusMsg->ErrorCode = br.readByte();
            SteeringStatusMsg->StatusFlags = br.readByte();
            SteeringStatusMsg->LimitFlags = br.readByte();
            error = false;
        }
        return error;
    }

    //Read the data record from the file.  It is assumed the file is set to the
    // start of the record and the record header has been read.
    //Returns true if error, false otherwise.
    bool EpasSteeringStatusDataRecord::readDataRecordFromFile(std::ifstream &logFile, int recordSize)
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