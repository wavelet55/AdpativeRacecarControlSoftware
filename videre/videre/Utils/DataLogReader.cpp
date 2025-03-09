/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: July 2018
 *
  *******************************************************************/

#include "DataLogReader.h"
#include "FileUtils.h"

using namespace std;

namespace VidereUtils
{

    DataLogReader::DataLogReader()
        : _recordMap(), _listOfLogFiles()
    {
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        _logFileSize = 0;
        _dirExists = false;
        _logFileIndex = 0;
        _currentFileNumber = 0;
        _NumberOfDataRecordsReadIn = 0;
        _EndOfDataRecords = false;
    }


    DataLogReader::DataLogReader(const std::string &baseFilename)
        : _recordMap(), _listOfLogFiles()
    {
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        _baseFilename = baseFilename;

        _logFileSize = 0;
        _dirExists = false;
        _logFileIndex = 0;
        _currentFileNumber = 0;
        _NumberOfDataRecordsReadIn = 0;
        _EndOfDataRecords = false;
    }

    DataLogReader::~DataLogReader()
    {
        closeLogFile();
    }

    void DataLogReader::closeLogFile()
    {
        try
        {
            if(_logFile.is_open())
            {
                _logFile.close();
                _logFileSize = 0;
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("DataLogReader:closeLogFile: Exception: " << e.what());
        }
    }

    //Get a sorted list of data log files.
    int DataLogReader::GetListOfFilesFromDirectory(const std::string &dir)
    {
        int numOfFiles = 0;
        _EndOfDataRecords = false;
        std::string ipmExt = LOG_EXTENSION_STR;
        if(dir.length() > 0)
        {
            _dirName = dir;
        }
        numOfFiles = VidereFileUtils::GetListFilesInDirectory(&_listOfLogFiles, _dirName,
                                                              ipmExt, _baseFilename, true);
        if(numOfFiles <= 0 )
            _EndOfDataRecords = true;

        return numOfFiles;
    }

    size_t DataLogReader::GetNumberOfBytesRemainingInFile()
    {
        size_t count = 0;
        if(_logFile.is_open())
        {
            count = _logFileSize - _logFile.tellg();
            count = count < 0 ? 0 : count;
        }
        return count;
    }

    bool DataLogReader::openNextLogFile()
    {
        bool fileOpened = false;
        try
        {
            //First ensure last file is closed
            closeLogFile();

            if(LoopBackToStartOfDataRecords && _logFileIndex >= _listOfLogFiles.size())
            {
                _logFileIndex = 0;
            }

            //Check the file index and ensure it is less
            //than the number of available files
            while(!fileOpened && (_logFileIndex < _listOfLogFiles.size()))
            {
                _currentFilePath = _listOfLogFiles[_logFileIndex];
                ios_base::openmode fileMode = ios_base::in | ios_base::binary;
                _logFile.open(_currentFilePath.c_str(), fileMode);
                //Ensure we are at the beginning of the file
                _logFile.seekg (0, std::ios::end);
                _logFileSize = _logFile.tellg();
                _logFile.seekg (0, std::ios::beg);
                ++_logFileIndex;
                fileOpened = true;
                _EndOfDataRecords = false;

                if( ReadFileHeader() )
                {
                    //If the Header cannot be read... close the file
                    //and try the next one.
                    LOGWARN("DataLogReader:openNextLogFile: Could not read File Header in file: " << _currentFilePath );
                    fileOpened = false;
                    closeLogFile();
                    _EndOfDataRecords = true;
                }
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("DataLogReader:openNextLogFile: Exception: " << e.what());
            closeLogFile();
            fileOpened = false;
            _EndOfDataRecords = true;
        }
        return fileOpened;
    }


    bool DataLogReader::checkRecordHeader(char buf[])
    {
        bool hdrOk = true;
        hdrOk = buf[0] == 'H' || buf[0] == 'R';
        hdrOk &= buf[1] == ':';
        hdrOk &= buf[DATARECORDHDRSIZE - 1] == '=';
        return hdrOk;
    }

    RecordTypeAndSize_t DataLogReader::readRecordHeader()
    {
        RecordTypeAndSize_t recordTypeSize;
        recordTypeSize.Clear();
        size_t bytesLeftInFile = 0;
        bool hdrFound = false;
        int maxTries = 1024;
        char rcdHdr[DATARECORDHDRSIZE + 1];

        try
        {
            while(!hdrFound && --maxTries > 0)
            {
                size_t currentFileLoc = _logFile.tellg();
                bytesLeftInFile = GetNumberOfBytesRemainingInFile();
                if(bytesLeftInFile > DATARECORDHDRSIZE)
                {
                    _logFile.read(rcdHdr, DATARECORDHDRSIZE);
                    if(checkRecordHeader(rcdHdr))
                    {
                        hdrFound = true;
                        recordTypeSize.IsHeader = rcdHdr[0] == 'H';
                        recordTypeSize.RecordType = (DataRecordType_e)rcdHdr[2];
                        recordTypeSize.RecordSize = (int)rcdHdr[3] << 24;
                        recordTypeSize.RecordSize |= (int)rcdHdr[4] << 16;
                        recordTypeSize.RecordSize |= (int)rcdHdr[5] << 8;
                        recordTypeSize.RecordSize |= (int)rcdHdr[6];
                    }
                    else
                    {
                        //This should not normally occur... but step forward and try again.
                        _logFile.seekg (currentFileLoc + 1, std::ios::beg);
                    }
                }
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("DataLogReader:readRecordHeader: Exception: " << e.what());
        }
        return recordTypeSize;
    }

    bool DataLogReader::ReadFileHeader()
    {
        bool error = true;
        RecordTypeAndSize_t recordTypeSize;
        if(_logFile.is_open())
        {
            try
            {
                recordTypeSize = readRecordHeader();
                if(recordTypeSize.RecordSize > 0)
                {
                    if(recordTypeSize.IsHeader)
                    {
                        if(HeaderRecordPtr.get() == nullptr || HeaderRecordPtr->GetRecordType() != recordTypeSize.RecordType)
                        {
                            HeaderRecordPtr = generateDataRecord(recordTypeSize.RecordType);
                        }
                        if(HeaderRecordPtr.get() != nullptr)
                        {
                            error = HeaderRecordPtr->readDataRecordFromFile(_logFile, recordTypeSize.RecordSize);
                        }
                    }
                }
                else
                {
                    //Assume no header... which is ok... and move on to reading data records.
                    //Reset the file back to its begining.
                    _logFile.seekg (0, std::ios::beg);
                }
            }
            catch (std::exception &e)
            {
                LOGERROR("DataLogReader:ReadFileHeader: Exception: " << e.what());
                error = true;
            }
        }
        return error;
    }


    std::shared_ptr<DataRecorderAbstractRecord> DataLogReader::getRecordOfType(DataRecordType_e recordType)
    {
        std::shared_ptr<DataRecorderAbstractRecord> recordPtr;
        auto mapElement = _recordMap.find(recordType);
        if(mapElement == _recordMap.end())
        {
            //Create a new record and add it to the map.
            recordPtr = generateDataRecord(recordType);
            if( recordPtr.get() != nullptr)
            {
                _recordMap[recordType] = recordPtr;
            }
        }
        else
        {
            recordPtr = mapElement->second;
        }
        return recordPtr;
    }


    std::shared_ptr<DataRecorderAbstractRecord> DataLogReader::ReadNextRecord()
    {
        std::shared_ptr<DataRecorderAbstractRecord> recordPtr;
        RecordTypeAndSize_t recordTypeSize;
        _EndOfDataRecords = false;
        bool logFileOpen = true;
        bool recordRead = false;
        int numberOfReadRecordTries = 0;

        while(!_EndOfDataRecords && !recordRead && numberOfReadRecordTries < 3)
        {
            ++numberOfReadRecordTries;
            if(!_logFile.is_open())
            {
                logFileOpen = openNextLogFile();
            }
            if(logFileOpen)
            {
                recordTypeSize = readRecordHeader();
                if(recordTypeSize.RecordSize > 0)
                {
                    if(GetNumberOfBytesRemainingInFile() >= recordTypeSize.RecordSize)
                    {
                        //Check to see if a record exists
                        recordPtr = getRecordOfType(recordTypeSize.RecordType);
                        if(recordPtr.get() != nullptr)
                        {
                            if(recordPtr->readDataRecordFromFile(_logFile, recordTypeSize.RecordSize))
                            {
                                LOGERROR("DataLogReader::ReadNextRecord: error reading record of type: "
                                                 << recordTypeSize.RecordType);
                            }
                            else
                            {
                                recordRead = true;
                            }
                        }
                        else
                        {
                            LOGERROR("DataLogReader::ReadNextRecord: can't get record of type: "
                                             << recordTypeSize.RecordType);
                        }
                    }
                    else
                    {
                        //We are at the end of the file
                        closeLogFile();
                    }
                }
                else
                {
                    //We are at the end of the file
                    closeLogFile();
                }
            }
            else
            {
                _EndOfDataRecords = true;
            }
        }
        return recordPtr;
    }

}
