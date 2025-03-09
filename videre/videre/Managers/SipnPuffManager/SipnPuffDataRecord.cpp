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
#include "SipnPuffDataRecord.h"
#include "ByteArrayReaderWriterVidere.h"

namespace videre
{

    SipnPuffDataRecord::SipnPuffDataRecord()
            : DataRecorderAbstractRecord()
    {
        _maxRecordSize = SIPNPUFFMAXRECORDSIZE;
        RecordType = DataRecordType_e::SPRT_SipnPuffVals;
        SipnPuffValueMsg = nullptr;
    }

    //Serialize data and put results into the local buffer;
    //DataPtr is a pointer to a structure or object containing the data
    //to be serialized.  If is is null, then the data to be serialized
    //is part of the concrete object.
    //Returns the number of bytes of the serialized data.
    char* SipnPuffDataRecord::serializeDataRecord(uint32_t *recordSizeOut)
    {
        _currentSerializedRecordSize = 0;
        if(SipnPuffValueMsg != nullptr)
        {
            VidereUtils::ByteArrayWriterVidere bw((byte *) _recordBuf, SIPNPUFFMAXRECORDSIZE,
                                                  Rabit::EndianOrder_e::Endian_Big);

            TimeStampSec = SipnPuffValueMsg->GetTimeStamp();
            bw.writeDouble(TimeStampSec);  //system timestamp
            bw.writeDouble(SipnPuffValueMsg->SipnPuffPecent);
            bw.writeDouble(SipnPuffValueMsg->SipnPuffIntegralPercent);
            _currentSerializedRecordSize = bw.Idx;
            *recordSizeOut = _currentSerializedRecordSize;
        }
        return _recordBuf;
    }


    //Deserialized the data buffer into a data structure.
    //Returns true if error, false otherwise.
    bool SipnPuffDataRecord::deserialzedDataRecord(char* dataBufPtr)
    {
        bool error = true;
        if(SipnPuffValueMsg.get() == nullptr)
        {
            SipnPuffValueMsg = std::make_shared<SipnPuffMessage>();
        }
        if(SipnPuffValueMsg != nullptr)
        {
            VidereUtils::ByteArrayReaderVidere br((byte *) dataBufPtr, SIPNPUFFMAXRECORDSIZE,
                                            Rabit::EndianOrder_e::Endian_Big);
            TimeStampSec = br.readDouble();
            SipnPuffValueMsg->SetTimeStamp(TimeStampSec);
            SipnPuffValueMsg->SipnPuffPecent = br.readDouble();
            SipnPuffValueMsg->SipnPuffIntegralPercent = br.readDouble();
            error = false;
        }
        return error;
    }

    //Read the data record from the file.  It is assumed the file is set to the
    // start of the record and the record header has been read.
    //Returns true if error, false otherwise.
    bool SipnPuffDataRecord::readDataRecordFromFile(std::ifstream &logFile, int recordSize)
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