/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: June 2018
 *
  *******************************************************************/


#include "VidereSystemControlDataRecord.h"
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include "ByteArrayReaderWriterVidere.h"

namespace videre
{

    VidereSystemControlDataRecord::VidereSystemControlDataRecord()
            : DataRecorderAbstractRecord()
    {
        _maxRecordSize = VIDERESYSTEMCONTROLMAXRECORDSIZE;
        RecordType = DataRecordType_e::DRT_SystemControl;
    }

    void VidereSystemControlDataRecord::Clear()
    {
        VidereTimestamp = 0;
        ControlTypeState = 0;
        DriverEnableSW = 0;
        DriverTorqueHit = 0;
        deltaTime = 0;
        SipnPuffValue = 0;
        ThrottleBrakeIntegralVal = 0;
        ThrottleControlVal = 0;
        BrakecontrolVal = 0;
        HeadRollAngleDegees = 0;
        HeadPitchAngleDegrees = 0;
        HeadYawAngleDegrees = 0;
        HeadLRAngleClamped = 0;
        HeadLRAngleLPF = 0;
        SteeringAngle = 0;
        SAError = 0;
        DtSAError = 0;
        IntgSAError = 0;
        SteeringTorqueCtrl = 0;
        VehiclePos_X = 0;
        VehiclePos_Y = 0;
        VehicleSpeed = 0;
        VehicleVel_X = 0;
        VehicleVel_Y = 0;

    }

    //Serialize data and put results into the local buffer;
    //DataPtr is a pointer to a structure or object containing the data
    //to be serialized.  If is is null, then the data to be serialized
    //is part of the concrete object.
    //Returns the number of bytes of the serialized data.
    char* VidereSystemControlDataRecord::serializeDataRecord(uint32_t *recordSizeOut)
    {
        _currentSerializedRecordSize = 0;
        VidereUtils::ByteArrayWriterVidere bw((byte *) _recordBuf, VIDERESYSTEMCONTROLMAXRECORDSIZE,
                                              Rabit::EndianOrder_e::Endian_Big);

        bw.writeDouble(VidereTimestamp);  //system timestamp
        bw.writeByte(ControlTypeState);
        bw.writeByte(DriverEnableSW);
        bw.writeByte(DriverTorqueHit);
        bw.writeDouble(deltaTime);
        bw.writeDouble(SipnPuffValue);
        bw.writeDouble(ThrottleBrakeIntegralVal);
        bw.writeDouble(ThrottleControlVal);
        bw.writeDouble(BrakecontrolVal);
        bw.writeDouble(HeadRollAngleDegees);
        bw.writeDouble(HeadPitchAngleDegrees);
        bw.writeDouble(HeadYawAngleDegrees);
        bw.writeDouble(HeadLRAngleClamped);
        bw.writeDouble(HeadLRAngleLPF);
        bw.writeDouble(SteeringAngle);
        bw.writeDouble(SAError);
        bw.writeDouble(DtSAError);
        bw.writeDouble(IntgSAError);
        bw.writeDouble(SteeringTorqueCtrl);
        bw.writeDouble(VehiclePos_X);
        bw.writeDouble(VehiclePos_Y);
        bw.writeDouble(VehicleSpeed);
        bw.writeDouble(VehicleVel_X);
        bw.writeDouble(VehicleVel_Y);

        _currentSerializedRecordSize = bw.Idx;
        *recordSizeOut = _currentSerializedRecordSize;
        return _recordBuf;
    }


    //Deserialized the data buffer into a data structure.
    //Returns true if error, false otherwise.
    bool VidereSystemControlDataRecord::deserialzedDataRecord(char* dataBufPtr)
    {
        bool error = true;
        VidereUtils::ByteArrayReaderVidere br((byte *) dataBufPtr, VIDERESYSTEMCONTROLMAXRECORDSIZE,
                                        Rabit::EndianOrder_e::Endian_Big);
        VidereTimestamp = br.readDouble();
        ControlTypeState = br.readByte();
        DriverEnableSW = br.readByte();
        DriverTorqueHit = br.readByte();
        deltaTime = br.readDouble();
        SipnPuffValue = br.readDouble();
        ThrottleBrakeIntegralVal = br.readDouble();
        ThrottleControlVal = br.readDouble();
        BrakecontrolVal = br.readDouble();
        HeadRollAngleDegees = br.readDouble();
        HeadPitchAngleDegrees = br.readDouble();
        HeadYawAngleDegrees = br.readDouble();
        HeadLRAngleClamped = br.readDouble();
        HeadLRAngleLPF = br.readDouble();
        SteeringAngle = br.readDouble();
        SAError = br.readDouble();
        DtSAError = br.readDouble();
        IntgSAError = br.readDouble();
        SteeringTorqueCtrl = br.readDouble();
        VehiclePos_X = br.readDouble();
        VehiclePos_Y = br.readDouble();
        VehicleSpeed = br.readDouble();
        VehicleVel_X = br.readDouble();
        VehicleVel_Y = br.readDouble();

        error = false;
        return error;
    }

    //Read the data record from the file.  It is assumed the file is set to the
    // start of the record and the record header has been read.
    //Returns true if error, false otherwise.
    bool VidereSystemControlDataRecord::readDataRecordFromFile(std::ifstream &logFile, int recordSize)
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