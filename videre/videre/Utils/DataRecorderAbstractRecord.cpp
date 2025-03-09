/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: June 2018
 *
  *******************************************************************/


#include "DataRecorderAbstractRecord.h"
#include "DataRecorderStdHeader.h"
#include "../Managers/IMUCommManager/IMU_DataRecord.h"
#include "../Managers/SipnPuffManager/SipnPuffDataRecord.h"
#include "../Managers/VehicleActuatorInterfaceManager/KarTechLADataRecords.h"
#include "../Managers/VehicleActuatorInterfaceManager/EpasSteeringDataRecords.h"
#include "../Managers/GPSManager/GPSDataRecordHeader.h"
#include "../Managers/GPSManager/GPSDataRecord.h"

using namespace videre;
using namespace IMU_SensorNS;

namespace VidereUtils
{

    char* DataRecorderAbstractRecord::getRecordTypeAndLenghtItems(int recordLength)
    {
        if( recordLength < 0 || recordLength > _maxRecordSize)
        {
            recordLength = _maxRecordSize;
        }
        _typeSizeTermBuf[0] = (char)RecordType;
        _typeSizeTermBuf[1] = (char)(recordLength >> 24);
        _typeSizeTermBuf[2] = (char)(recordLength >> 16);
        _typeSizeTermBuf[3] = (char)(recordLength >> 8);
        _typeSizeTermBuf[4] = (char)(recordLength & 0xFF);
        _typeSizeTermBuf[5] = '=';
        _typeSizeTermBuf[6] = 0;  //Null term c-str
        return _typeSizeTermBuf;
    }


    std::shared_ptr<DataRecorderAbstractRecord> generateDataRecord(DataRecordType_e recordType)
    {
        std::shared_ptr<DataRecorderAbstractRecord> recordSPtr;
        switch(recordType)
        {
            case DataRecordType_e::DRT_Header:
                recordSPtr = std::make_shared<DataRecorderStdHeader>("Unknown", 0);
                break;
            case DataRecordType_e::IMURT_AccelGyro:
                recordSPtr = std::make_shared<IMU_AccelGyroRecord>();
                break;
            case DataRecordType_e::IMURT_HeadOrientation:
                recordSPtr = std::make_shared<IMU_HeadOrientationRecord>();
                break;
            case DataRecordType_e::SPRT_SipnPuffVals:
                recordSPtr = std::make_shared<SipnPuffDataRecord>();
                break;
            case DataRecordType_e::GPSRD_GPS_Header:
                recordSPtr = std::make_shared<GPSDataRecordHeader>("Unknown", 0);
                break;
            case DataRecordType_e::GPSRT_GPS_Data:
                recordSPtr = std::make_shared<GPSDataRecord>();
                break;
            case DataRecordType_e::KTLA_Brake_Cmd:
                recordSPtr = std::make_shared<KarTechLACommandDataRecord>(DataRecordType_e::KTLA_Brake_Cmd);
                break;
            case DataRecordType_e::KTLA_Brake_Status:
                recordSPtr = std::make_shared<KarTechLAStatusDataRecord>(DataRecordType_e::KTLA_Brake_Status);
                break;
            case DataRecordType_e::KTLA_Throttle_Cmd:
                recordSPtr = std::make_shared<KarTechLACommandDataRecord>(DataRecordType_e::KTLA_Throttle_Cmd);
                break;
            case DataRecordType_e::KTLA_Throttle_Status:
                recordSPtr = std::make_shared<KarTechLAStatusDataRecord>(DataRecordType_e::KTLA_Throttle_Status);
                break;
            case DataRecordType_e::EPAS_Steering_Cmd:
                recordSPtr = std::make_shared<EpasSteeringCommandDataRecord>();
                break;
            case DataRecordType_e::EPAS_Steering_Status:
                recordSPtr = std::make_shared<EpasSteeringStatusDataRecord>();
                break;

        }
        return recordSPtr;
    }

}


