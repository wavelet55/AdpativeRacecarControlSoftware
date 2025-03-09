/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: June 2018
 *
  *******************************************************************/

#include "DataRecorder.h"
#include "FileUtils.h"

using namespace std;
using namespace VidereFileUtils;

namespace VidereUtils
{
    DataRecorder::DataRecorder()
    {
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);
        _dirExists = false;
        _currentFileNumber = 0;
    }

    DataRecorder::DataRecorder(const std::string baseFilename, DataRecorderAbstractRecord *logFileHeaderRecord)
    {
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        _baseFilename = baseFilename;
        _headerRecordPtr = logFileHeaderRecord;

        if( _headerRecordPtr != nullptr )
        {
            uint32_t recordSize;
            _headerRecordPtr->SetTimeNow();
            //The header does not change over an operational run so
            //it can be serialized once here.
            _headerRecordPtr->serializeDataRecord(&recordSize);
        }

        _dirExists = false;
        _currentFileNumber = 0;
    }

    DataRecorder::~DataRecorder()
    {
        closeLogFile();
    }

    bool DataRecorder::CreateDirectory(const std::string &directory, bool addTimestamp)
    {
        _dirName = directory;
        _dirExists = false;
        try
        {
            if (addTimestamp)
            {
                _dirName = VidereFileUtils::AddCurrentTimeDateStampToString(_dirName);
            }
            if (VidereFileUtils::CreateDirectory(_dirName))
            {
                _dirExists = true;
            }
            else
            {
                LOGERROR("DataRecorder:CreateDirectory: Could not create direcory: " << _dirName);
                _dirExists = false;
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("DataRecorder:CreateDirectory: Exception: " << e.what());
        }
        return _dirExists;
    }

    bool DataRecorder::openNewLogFile()
    {
        bool error = true;
        try
        {
            //First ensure last file is closed
            if(_logFile.is_open())
            {
                _logFile.close();
            }

            if(!_dirExists)
            {
                if(!CreateDirectory(_dirName, true))
                {
                    return true;
                }
            }

            ++_currentFileNumber;
            string filename = VidereFileUtils::AddIndexToFilename(_baseFilename,
                                                                  _currentFileNumber,
                                                                  4, LOG_EXTENSION_STR);
            _currentFilePath = _dirName + "/" + filename;
            ios_base::openmode fileMode = ios_base::out | ios_base::trunc | ios_base::binary;
            _logFile.open(_currentFilePath.c_str(), fileMode);

            if(_logFile.is_open())
            {
            //Write File Header to the file.
                if(_headerRecordPtr != nullptr)
                {
                    uint32_t recordSize;
                    _headerRecordPtr->SetTimeNow();
                    char* recordPtr = _headerRecordPtr->serializeDataRecord(&recordSize);
                    if(recordPtr != nullptr && recordSize > 0)
                    {
                        _logFile.write(FILE_HDR_STR.c_str(), FILE_HDR_STR.size());
                        _logFile.write(_headerRecordPtr->getRecordTypeAndLenghtItems(recordSize),
                                DATARECORDERTYPESIZETERMSIZE);
                        _logFile.write(recordPtr, recordSize);
                    }
                }
                error = false;
            }
            else
            {
                error = true;
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("ImagePlusMetadataRecorder:openNewImageFile: Exception: " << e.what());
            error = true;
        }
        return error;
    }

    void DataRecorder::closeLogFile()
    {
        try
        {
            if(_logFile.is_open())
            {
                _logFile.close();
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("DataRecorder:closeImageFile: Exception: " << e.what());
        }
    }

    bool DataRecorder::writeDataRecord(DataRecorderAbstractRecord &dataRecordObj)
    {
        bool error = true;
        char* dataRecordPtr = nullptr;
        uint32_t dataRecordSize = 0;
        if(!_logFile.is_open())
        {
            if( openNewLogFile() )
            {
                LOGERROR("DataRecorder:writeDataRecord: Can't open Data Log File.");
                return true;
            }
        }
        try
        {
            dataRecordPtr = dataRecordObj.serializeDataRecord(&dataRecordSize);
            if(dataRecordPtr != nullptr &&  dataRecordSize > 0)
            {
                _logFile.write(RECORD_HDR_STR.c_str(), RECORD_HDR_STR.size());
                _logFile.write(dataRecordObj.getRecordTypeAndLenghtItems(dataRecordSize),
                               DATARECORDERTYPESIZETERMSIZE);
                _logFile.write(dataRecordPtr, dataRecordSize);
            }
            if(GetCurrentFileSize() > _maxFileSize)
            {
                //A new log file will be opened on the next write process
                closeLogFile();
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("DataRecorder:writeDataRecord: Exception: " << e.what());
        }
        return error;
    }


}