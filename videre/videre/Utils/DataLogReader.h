/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: July 2018
 *
  *******************************************************************/

#ifndef VIDERE_DEV_DATALOGREADER_H
#define VIDERE_DEV_DATALOGREADER_H

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

#define DATARECORDHDRSIZE   DATARECORDERTYPESIZETERMSIZE + 2

    struct RecordTypeAndSize_t
    {
        bool IsHeader;
        DataRecordType_e RecordType;
        int RecordSize;

        void Clear()
        {
            IsHeader = false;
            RecordType = DataRecordType_e::DRT_Header;
            RecordSize = 0;
        }
    };

    //DataLogReader is for reading and playing back data log
    //files created by the Data Recorder.
    class DataLogReader
    {
    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        const int MegaByte = 1024 * 1024;
        const std::string LOG_EXTENSION_STR = "dat";

        bool _dirExists = false;
        bool _imageSaveError = false;
        int _currentFileNumber = 0;

        std::string _dirName = "PlaybackDataLogs";
        std::string _baseFilename = "DataLog"; /* Base filename*/
        boost::filesystem::path _currentFilePath; /* Full path to log file */
        std::ifstream _logFile;
        size_t _logFileSize = 0;


        std::vector<boost::filesystem::path> _listOfLogFiles;
        int _logFileIndex = 0;

        std::map<DataRecordType_e, std::shared_ptr<DataRecorderAbstractRecord>> _recordMap;

        int _NumberOfDataRecordsReadIn = 0;

        bool _EndOfDataRecords = false;


    public:
        //If set to true... will loop back to the start of the data records
        //in the first file after reaching the end of all the records.
        bool LoopBackToStartOfDataRecords = false;

        //The user can supply a header record that will be added to the start of
        //each new log file.  A header Record must contain the data and
        //it must contain the buffer to serialized the data into.
        std::shared_ptr<DataRecorderAbstractRecord> HeaderRecordPtr;

    public:
        DataLogReader();

        DataLogReader(const std::string &baseFilename);

        ~DataLogReader();

        size_t GetNumberOfBytesRemainingInFile();

        int GetNumberOfDataRecordsReadIn()
        {
            return _NumberOfDataRecordsReadIn;
        }

        bool IsEndOfDataRecords()
        {
            return _EndOfDataRecords;
        }

        void closeLogFile();

        bool openNextLogFile();

        void setBaseFilename(std::string &fn)
        {
            _baseFilename = fn;
        }

        void setDirectoryName(std::string &fn)
        {
            _dirName = fn;
        }

        //Get a Sorted List of Data Log files located in the
        //given directory.
        //Returns the number of files found.
        int GetListOfFilesFromDirectory(const std::string &dir );

        int GetListOfFilesFromDirectory()
        {
            GetListOfFilesFromDirectory(_dirName);
        }

        //Add a record to the internal dictionary of records.
        //Use this method to add record types that will be read from the data log
        //file.  Since records often contain an internal message, this allows linking
        //in a mesage that will be populated when read from the log file.
        int AddRecordToRecords(std::shared_ptr<DataRecorderAbstractRecord> record)
        {
            if(record.get() != nullptr)
                _recordMap[record->RecordType] = record;

            return _recordMap.size();
        }

        int GetNumberOfRecordsInDictionary()
        {
            return _recordMap.size();
        }

        void ClearRecordDictionary()
        {
            _recordMap.clear();
        }

        void ResetReaderToStartOfLogFiles()
        {
            closeLogFile();
            _logFileIndex = 0;
        }

        //Read the record header... return the size of the record
        //and the record type.  If there is no more records
        //a zero is returned.  If there is an error a value < 0 is returned.
        //Assumes the file is open.
        RecordTypeAndSize_t readRecordHeader();

        bool checkRecordHeader(char buf[]);

        bool ReadFileHeader();

        //Read the next available record.
        //Returns a shared pointer to the record if available.
        //returns an empty shared pointer if no record is available
        //such as at the end of the record file.
        std::shared_ptr<DataRecorderAbstractRecord> ReadNextRecord();

        std::shared_ptr<DataRecorderAbstractRecord> getRecordOfType(DataRecordType_e recordType);
    };

}
#endif //VIDERE_DEV_DATALOGREADER_H
