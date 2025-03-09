/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: June 2018
 *
  *******************************************************************/


#ifndef VIDERE_DEV_DATARECORDERSTDHEADER_H
#define VIDERE_DEV_DATARECORDERSTDHEADER_H

#include "DataRecorderAbstractRecord.h"
#include "ByteArrayReaderWriterVidere.h"

namespace VidereUtils
{
#define DATARECORDERSTDHEADERSIZE 128

    class DataRecorderStdHeader : public DataRecorderAbstractRecord
    {
    private:
        char _recordBuf[DATARECORDERSTDHEADERSIZE];

    public:
        std::string HeaderName;
        uint32_t VersionNumber = 0;

        //The Endianess of the Computer generating the Image Plus Metadata
        //File.  This information is required because the Metadata information
        //is stored directly as a blob, and the Endianess of the data will
        //depend on the computer operating system.
        Rabit::EndianOrder_e ComputerEndianess;

        DataRecorderStdHeader(const std::string &headerName, uint32_t versionNo);

        //Get a pointer to the Serialized Data Record
        virtual char* GetSerializedDataRecordPtr() { return _recordBuf; }

        //Serialize data and put results into the local buffer;
        //DataPtr is a pointer to a structure or object containing the data
        //to be serialized.  If is is null, then the data to be serialized
        //is part of the concrete object.
        //Returns the number of bytes of the serialized data.
        virtual char* serializeDataRecord(uint32_t *recordSizeOut);


        //Deserialized the data buffer into a data structure.
        //Returns true if error, false otherwise.
        virtual bool deserialzedDataRecord(char* dataBufPtr);

        //Read the data record from the file.  It is assumed the file is set to the
        // start of the record and the record header has been read.
        //Returns true if error, false otherwise.
        virtual bool readDataRecordFromFile(std::ifstream &logFile, int recordSize) final;

    };

}
#endif //VIDERE_DEV_DATARECORDERSTDHEADER_H
