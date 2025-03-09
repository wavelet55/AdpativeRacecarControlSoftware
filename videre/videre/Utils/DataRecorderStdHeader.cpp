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
#include "DataRecorderStdHeader.h"
#include <SystemTimeClock.h>
#include <memory.h>
#include "ByteArrayReaderWriterVidere.h"
#include <time.h>

using namespace std;

namespace VidereUtils
{

    DataRecorderStdHeader::DataRecorderStdHeader(const std::string &headerName, uint32_t versionNo)
        : DataRecorderAbstractRecord()
    {
        _maxRecordSize = DATARECORDERSTDHEADERSIZE;
        RecordType = DataRecordType_e::DRT_Header;
         VersionNumber = versionNo;
        if(headerName.size() > 64)
        {
            //Truncate header name.
            HeaderName = headerName.substr(0, 64);
        }
        else
        {
            HeaderName = headerName;
        }

        SetTimeNow();
    }

    //Serialize data and put results into the local buffer;
    //DataPtr is a pointer to a structure or object containing the data
    //to be serialized.  If is is null, then the data to be serialized
    //is part of the concrete object.
    //Returns the number of bytes of the serialized data.
    char* DataRecorderStdHeader::serializeDataRecord(uint32_t *recordSizeOut)
    {
        std::tm localTimeDate;
        uint32_t size = 0;
        memset(_recordBuf, 0, DATARECORDERSTDHEADERSIZE);
        VidereUtils::ByteArrayWriterVidere bw((byte*)_recordBuf, DATARECORDERSTDHEADERSIZE, Rabit::EndianOrder_e::Endian_Big);
        ComputerEndianess = GetSystemEndianness();
        SetTimeNow();
        localTimeDate = Rabit::SystemTimeClock::GetSystemTimeClock()->GetLocalTimeStructFromGpsTimeStamp(TimeStampSec);
        bw.writeString(HeaderName);
        bw.writeUInt32(VersionNumber);
        bw.writeByte((uint8_t)ComputerEndianess);
        bw.writeDouble(TimeStampSec);
        bw.writeInt16(localTimeDate.tm_year + 1900);
        bw.writeByte(localTimeDate.tm_mon);
        bw.writeByte(localTimeDate.tm_mday);
        bw.writeByte(localTimeDate.tm_hour);
        bw.writeByte(localTimeDate.tm_min);
        bw.writeByte(localTimeDate.tm_sec);
        _currentSerializedRecordSize = bw.Idx;
        *recordSizeOut = _currentSerializedRecordSize;
        return _recordBuf;
    }


    //Deserialized the data buffer into a data structure.
    //Returns true if error, false otherwise.
    bool DataRecorderStdHeader::deserialzedDataRecord(char* dataBufPtr)
    {
        bool error = false;
        string hdrName;
        VidereUtils::ByteArrayReaderVidere br((byte*)dataBufPtr, DATARECORDERSTDHEADERSIZE, Rabit::EndianOrder_e::Endian_Big);
        //check the header Name
        br.readString(&hdrName);
        VersionNumber = br.readUInt32();
        ComputerEndianess = (Rabit::EndianOrder_e)br.readByte();
        TimeStampSec = br.readDouble();

        //No point in reading the Time info... it is contained in the timestamp.
        return error;
    }

    //Read the data record from the file.  It is assumed the file is set to the
    // start of the record and the record header has been read.
    //Returns true if error, false otherwise.
    bool DataRecorderStdHeader::readDataRecordFromFile(std::ifstream &logFile, int recordSize)
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