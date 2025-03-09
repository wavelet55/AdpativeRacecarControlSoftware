/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: June 2018
 *
  *******************************************************************/

#ifndef VIDERE_DEV_DATARECORDER_H
#define VIDERE_DEV_DATARECORDER_H

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <boost/filesystem.hpp>
#include "global_defines.h"
#include "logger.h"
#include "DataRecorderAbstractRecord.h"

namespace VidereUtils
{
#define DATARECORDSTSTRSIZE 2

    //A Data Recorder for recording/logging data to files.
    //Data is recorded one record at a time.
    //Each record will start with a short ID: "*D:"
    //followed by a 1-byte record type,
    //then a 2-byte record length, then an "=" sign and then the
    //serialized record.  The serialized record can be of any format.
    //Typically data is serialized into a binary format for efficiency,
    //but the user has complete control.
    //The file may contain upto 256 different record types as suites
    //the user.  The user defines the record types.
    //It is a good idea to start a log file with a header record type
    //that contains the type of info stored in the file and any versioning
    //info that will help in the decoding of the file later.
    //
    //The DataRecorder supports multiple indexed files of the same type
    //to help keep any one log file from becoming too large.
    class DataRecorder
    {
    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        const int MegaByte = 1024 * 1024;
        const std::string LOG_EXTENSION_STR = "dat";

        const std::string FILE_HDR_STR = "H:";  //The start of each File Header
        const std::string RECORD_HDR_STR = "R:";  //The start of each Record

        bool _dirExists = false;
        bool _imageSaveError = false;
        int _currentFileNumber = 0;

        std::string _dirName = "DataLogs";
        std::string _baseFilename = "DataLog"; /* Base filename*/
        boost::filesystem::path _currentFilePath; /* Full path to log file */
        std::ofstream _logFile;

        size_t _maxFileSize = 1000 * MegaByte;

        //The user can supply a header record that will be added to the start of
        //each new log file.  A header Record must contain the data and
        //it must contain the buffer to serialized the data into.
        DataRecorderAbstractRecord *_headerRecordPtr = nullptr;

    public:
        DataRecorder();

        DataRecorder(const std::string baseFilename, DataRecorderAbstractRecord *logFileHeaderRecord = nullptr);

        ~DataRecorder();

        //Seting the directory name.
        //IF createDir == false... it is assumed the directory has already been created.
        bool setDirectory(std::string &dirname, bool createDir = false, bool addTimestamp = false)
        {
            _dirName = dirname;
            if( createDir )
            {
                CreateDirectory(dirname, addTimestamp);
            }
            else
            {
                _dirExists = true;
            }
            return _dirExists;
        }

        void setBaseFilename(std::string fn)
        {
            _baseFilename = fn;
        }

        bool CreateDirectory(const std::string &directory, bool addTimestamp);

        void setMaxFileSizeMegaBytes(int maxFileSize)
        {
            maxFileSize = maxFileSize < 1 ? 1 : maxFileSize > 10000 ? 10000 : maxFileSize;
        }

        bool openNewLogFile();

        void closeLogFile();

        void setHeaderRecord(DataRecorderAbstractRecord *headerRecordPtr)
        {
            _headerRecordPtr = headerRecordPtr;
            uint32_t recordSize = 0;
            if(_headerRecordPtr != nullptr)
            {
                _headerRecordPtr->SetTimeNow();
                //The header does not change over an operational run so
                //it can be serialized once here.
                _headerRecordPtr->serializeDataRecord(&recordSize);
            }
        }

        bool writeDataRecord(DataRecorderAbstractRecord &dataRecordObj);

        //This is based upon the file pointer being at the
        //end of the file... works when writting to the file at the end.
        size_t GetCurrentFileSize()
        {
            size_t size = 0;
            if(_logFile.is_open())
            {
                size = _logFile.tellp();
            }
            return size;
        }

    };

}
#endif //VIDERE_DEV_DATARECORDER_H
