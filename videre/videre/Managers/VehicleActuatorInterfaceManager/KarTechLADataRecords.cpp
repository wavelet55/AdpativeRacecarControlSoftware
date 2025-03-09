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
#include "KarTechLADataRecords.h"
#include "ByteArrayReaderWriterVidere.h"

namespace videre
{

    KarTechLACommandDataRecord::KarTechLACommandDataRecord()
            : DataRecorderAbstractRecord()
    {
        _maxRecordSize = KARTECHCMDMAXRECORDSIZE;
        RecordType = DataRecordType_e::KTLA_Brake_Cmd;
        PositionControlMsg = nullptr;
    }

    KarTechLACommandDataRecord::KarTechLACommandDataRecord(DataRecordType_e recordType)
            : DataRecorderAbstractRecord()
    {
        _maxRecordSize = KARTECHCMDMAXRECORDSIZE;
        RecordType = recordType;
        PositionControlMsg = nullptr;
    }

    //Serialize data and put results into the local buffer;
    //DataPtr is a pointer to a structure or object containing the data
    //to be serialized.  If is is null, then the data to be serialized
    //is part of the concrete object.
    //Returns the number of bytes of the serialized data.
    char* KarTechLACommandDataRecord::serializeDataRecord(uint32_t *recordSizeOut)
    {
        _currentSerializedRecordSize = 0;
        if(PositionControlMsg != nullptr)
        {
            VidereUtils::ByteArrayWriterVidere bw((byte *) _recordBuf, KARTECHCMDMAXRECORDSIZE,
                                                  Rabit::EndianOrder_e::Endian_Big);

            TimeStampSec = PositionControlMsg->GetTimeStamp();
            bw.writeDouble(TimeStampSec);  //system timestamp
            bw.writeByte(PositionControlMsg->ClutchEnable);
            bw.writeByte(PositionControlMsg->MotorEnable);
            bw.writeByte(PositionControlMsg->ManualExtControl);
            bw.writeDouble(PositionControlMsg->getPositionPercent());
            bw.writeDouble(PositionSetInches);
            _currentSerializedRecordSize = bw.Idx;
            *recordSizeOut = _currentSerializedRecordSize;
        }
        return _recordBuf;
    }


    //Deserialized the data buffer into a data structure.
    //Returns true if error, false otherwise.
    bool KarTechLACommandDataRecord::deserialzedDataRecord(char* dataBufPtr)
    {
        bool error = true;
        if(PositionControlMsg.get() == nullptr)
        {
            PositionControlMsg = std::make_shared<LinearActuatorPositionCtrlMessage>();
            if(RecordType == DataRecordType_e::KTLA_Brake_Cmd)
                PositionControlMsg->FunctionType == LinearActuatorFunction_e::LA_Brake;
            else
                PositionControlMsg->FunctionType == LinearActuatorFunction_e::LA_Accelerator;
        }
        if(PositionControlMsg != nullptr)
        {
            VidereUtils::ByteArrayReaderVidere br((byte *) dataBufPtr, KARTECHCMDMAXRECORDSIZE,
                                            Rabit::EndianOrder_e::Endian_Big);
            TimeStampSec = br.readDouble();
            PositionControlMsg->SetTimeStamp(TimeStampSec);
            PositionControlMsg->ClutchEnable = br.readByte();
            PositionControlMsg->MotorEnable = br.readByte();
            PositionControlMsg->ManualExtControl = br.readByte();
            PositionControlMsg->setPositionPercent(br.readDouble());
            PositionSetInches = br.readDouble();
            error = false;
        }
        return error;
    }

    //Read the data record from the file.  It is assumed the file is set to the
    // start of the record and the record header has been read.
    //Returns true if error, false otherwise.
    bool KarTechLACommandDataRecord::readDataRecordFromFile(std::ifstream &logFile, int recordSize)
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


    KarTechLAStatusDataRecord::KarTechLAStatusDataRecord()
            : DataRecorderAbstractRecord()
    {
        _maxRecordSize = KARTECHSTATUSMAXRECORDSIZE;
        RecordType = DataRecordType_e::KTLA_Brake_Cmd;
        LinearActuatorStatusFeedbackMsg = nullptr;
    }

    KarTechLAStatusDataRecord::KarTechLAStatusDataRecord(DataRecordType_e recordType)
            : DataRecorderAbstractRecord()
    {
        _maxRecordSize = KARTECHSTATUSMAXRECORDSIZE;
        RecordType = recordType;
        LinearActuatorStatusFeedbackMsg = nullptr;
    }

    //Serialize data and put results into the local buffer;
    //DataPtr is a pointer to a structure or object containing the data
    //to be serialized.  If is is null, then the data to be serialized
    //is part of the concrete object.
    //Returns the number of bytes of the serialized data.
    char* KarTechLAStatusDataRecord::serializeDataRecord(uint32_t *recordSizeOut)
    {
        _currentSerializedRecordSize = 0;
        if(LinearActuatorStatusFeedbackMsg != nullptr)
        {
            VidereUtils::ByteArrayWriterVidere bw((byte *) _recordBuf, KARTECHSTATUSMAXRECORDSIZE,
                                                  Rabit::EndianOrder_e::Endian_Big);

            TimeStampSec = LinearActuatorStatusFeedbackMsg->GetTimeStamp();
            bw.writeDouble(TimeStampSec);  //system timestamp
            bw.writeDouble(LinearActuatorStatusFeedbackMsg->getPositionPercent());
            bw.writeDouble(ActuatorPostionInches);
            bw.writeDouble(LinearActuatorStatusFeedbackMsg->getMotorCurrentAmps());
            _currentSerializedRecordSize = bw.Idx;
            *recordSizeOut = _currentSerializedRecordSize;
        }
        return _recordBuf;
    }


    //Deserialized the data buffer into a data structure.
    //Returns true if error, false otherwise.
    bool KarTechLAStatusDataRecord::deserialzedDataRecord(char* dataBufPtr)
    {
        bool error = true;
        if(LinearActuatorStatusFeedbackMsg.get() == nullptr)
        {
            LinearActuatorStatusFeedbackMsg = std::make_shared<LinearActuatorPositionCtrlMessage>();
            if(RecordType == DataRecordType_e::KTLA_Brake_Status)
                LinearActuatorStatusFeedbackMsg->FunctionType == LinearActuatorFunction_e::LA_Brake;
            else
                LinearActuatorStatusFeedbackMsg->FunctionType == LinearActuatorFunction_e::LA_Accelerator;
        }
        if(LinearActuatorStatusFeedbackMsg != nullptr)
        {
            VidereUtils::ByteArrayReaderVidere br((byte *) dataBufPtr, KARTECHCMDMAXRECORDSIZE,
                                            Rabit::EndianOrder_e::Endian_Big);
            TimeStampSec = br.readDouble();
            LinearActuatorStatusFeedbackMsg->SetTimeStamp(TimeStampSec);
            LinearActuatorStatusFeedbackMsg->setPositionPercent(br.readDouble());
            ActuatorPostionInches = br.readDouble();
            LinearActuatorStatusFeedbackMsg->setMotorCurrentAmps(br.readDouble());
            error = false;
        }
        return error;
    }

    //Read the data record from the file.  It is assumed the file is set to the
    // start of the record and the record header has been read.
    //Returns true if error, false otherwise.
    bool KarTechLAStatusDataRecord::readDataRecordFromFile(std::ifstream &logFile, int recordSize)
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