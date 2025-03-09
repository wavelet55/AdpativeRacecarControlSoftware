/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: June 2018
 *
  *******************************************************************/


#ifndef VIDERE_DEV_IMU_DATARECORD_H
#define VIDERE_DEV_IMU_DATARECORD_H

#include "DataRecorderAbstractRecord.h"
#include "IMU_DataTypeDefs.h"
#include "HeadOrientationMessage.h"
#include "AccelerometerGyroMessage.h"

using namespace videre;
using namespace VidereUtils;

namespace IMU_SensorNS
{
#define IMUHEADORIENTATIONMAXRECORDSIZE 64
#define IMUACCELGYROMAXRECORDSIZE 128


    class IMU_HeadOrientationRecord : public DataRecorderAbstractRecord
    {
    private:

        char _recordBuf[IMUHEADORIENTATIONMAXRECORDSIZE];

    public:
        std::shared_ptr<HeadOrientationMessage> HeadOrientationMsg;

    public:
        IMU_HeadOrientationRecord();

        IMU_HeadOrientationRecord(std::shared_ptr<HeadOrientationMessage> headOrientatinMsg);


        void setHeadOrientationMsg(std::shared_ptr<HeadOrientationMessage> headOrientationMsg)
        {
            HeadOrientationMsg = headOrientationMsg;
        }

        //Get a pointer to the Serialized Data Record
        virtual char* GetSerializedDataRecordPtr() { return _recordBuf; }

        //Serialize data and put results into the local buffer;
        //DataPtr is a pointer to a structure or object containing the data
        //to be serialized.  If is is null, then the data to be serialized
        //is part of the concrete object.
        //Returns the number of bytes of the serialized data.
        virtual char* serializeDataRecord(uint32_t *recordSizeOut) final;


        //Deserialized the data buffer into a data structure.
        //Returns true if error, false otherwise.
        virtual bool deserialzedDataRecord(char* dataBufPtr) final;

        //Read the data record from the file.  It is assumed the file is set to the
        // start of the record and the record header has been read.
        //Returns true if error, false otherwise.
        virtual bool readDataRecordFromFile(std::ifstream &logFile, int recordSize) final;

    };

    class IMU_AccelGyroRecord : public DataRecorderAbstractRecord
    {
    private:

        char _recordBuf[IMUHEADORIENTATIONMAXRECORDSIZE];

    public:
        std::shared_ptr<AccelerometerGyroMessage> AccelGyroMsg;

    public:
        IMU_AccelGyroRecord();

        IMU_AccelGyroRecord(std::shared_ptr<AccelerometerGyroMessage> accelGyroMsg);


        //Get a pointer to the Serialized Data Record
        virtual char* GetSerializedDataRecordPtr() { return _recordBuf; }

        //Serialize data and put results into the local buffer;
        //DataPtr is a pointer to a structure or object containing the data
        //to be serialized.  If is is null, then the data to be serialized
        //is part of the concrete object.
        //Returns the number of bytes of the serialized data.
        virtual char* serializeDataRecord(uint32_t *recordSizeOut) final;


        //Deserialized the data buffer into a data structure.
        //Returns true if error, false otherwise.
        virtual bool deserialzedDataRecord(char* dataBufPtr) final;

        //Read the data record from the file.  It is assumed the file is set to the
        // start of the record and the record header has been read.
        //Returns true if error, false otherwise.
        virtual bool readDataRecordFromFile(std::ifstream &logFile, int recordSize) final;

    };


}
#endif //VIDERE_DEV_IMU_DATARECORD_H
