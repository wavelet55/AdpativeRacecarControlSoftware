/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Aug 2018
 *
  *******************************************************************/


#ifndef VIDERE_DEV_VEHICLESTATEDATARECORD_H
#define VIDERE_DEV_VEHICLESTATEDATARECORD_H

#include "DataRecorderAbstractRecord.h"
#include "../IMUCommManager/IMU_DataTypeDefs.h"
#include "SipnPuffMessage.h"

using namespace videre;
using namespace VidereUtils;

namespace videre
{
#define VEHICLESTATEMAXRECORDSIZE 64

    class VehicleStateDataRecord : public DataRecorderAbstractRecord
    {
    private:

        char _recordBuf[VEHICLESTATEMAXRECORDSIZE];

    public:
        std::shared_ptr<SipnPuffMessage> SipnPuffValueMsg;

    public:
        VehicleStateDataRecord();


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
#endif //VIDERE_DEV_VEHICLESTATEDATARECORD_H
