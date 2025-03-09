/* ****************************************************************
 * Athr(s): Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: Sept. 2021
 *
 * DTX IMU Message Parser
 *******************************************************************/

#include "DTX_IMU_RxMessageParser.h"
#include "DTX_IMU_InterfaceManagerWSRMgr.h"
#include "global_defines.h"
#include <RabitMessage.h>
#include <fastcdr/Cdr.h>
#include "host_conn_rep.h"
#include "host_conn_req.h"

using namespace std;
//using namespace DTX_DataRecords;

namespace videre
{

    DTX_IMU_RxMessageParser::DTX_IMU_RxMessageParser(Rabit::RabitManager* mgrPtr,
                                            DTX_IMU_InterfaceManagerWSRMgrWorkSpace *mgrWSPtr,
                                             std::shared_ptr<ConfigData> config)
            //: _dataRecorder(),
            //  _dataRecorderStdHeader("DTX IMU Interface Data Log", 0),
            //  _exDataWrapperRecord(),
            //  _exerciseProgressRecord(mgrWSPtr->GF360ExerciseProgressMsg),
            //  _lateralMotorPosCtrlMonitorRecord(mgrWSPtr->GF360LateralMotorPosCtrlMonitorMsg)
    {
        _mgrPtr = mgrPtr;
        _wsPtr = mgrWSPtr;
        _config_sptr = config;
        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("ctli");
        log4cpp_->setAdditivity(false);

        //_dataRecorder.setDirectory(DataLogDirectory);

        //_dataRecorder.setBaseFilename("GF360CIDataLog");
        //_dataRecorder.setBaseFilename(GF360_RecordFileNames[(int)GF360_RecordFileType_e::GF360RFT_GF360_DTXCTRL]);
        //_dataRecorder.setHeaderRecord(&_dataRecorderStdHeader);
        //_dataRecorder.setFileIndexNumber(0);
        //_recordExerciseData = false;
        //_currentRecordCtrlState = GF360_RecordControlCmd_e::GF360RCC_None;
        //_exerciseProgressRecord.ValuesRecord = _wsPtr->GF360ExerciseProgressMsg;
        //_lateralMotorPosCtrlMonitorRecord.ValuesRecord = _wsPtr->GF360LateralMotorPosCtrlMonitorMsg;

    }

    bool DTX_IMU_RxMessageParser::Initialize()
    {
        bool error = false;
        _totalNumberMsgs = 0;
        _totalNumberMsgCrcErrors = 0;
        //The data recorder file should be closed, but be sure.
        //_dataRecorder.closeLogFile();
        //_dataRecorder.setFileIndexNumber(0);
        //_recordExerciseData = false;
        //_currentRecordCtrlState = GF360_RecordControlCmd_e::GF360RCC_None;
        //_wsPtr->DTX_IMU_CtrlInterfaceMgrRecordStatusMsg->Clear();
        //_wsPtr->DTX_IMU_CtrlInterfaceMgrRecordStatusMsg->record_file_type = (uint8_t)GF360_RecordFileType_e::GF360RFT_GF360_DTXCTRL;
        //_wsPtr->DTX_IMU_CtrlInterfaceMgrRecordStatusMsg->record_file_status = (uint8_t)GF360_RecordStatus_e::GF360RST_FileClosed;
        //_wsPtr->DTX_IMU_CtrlInterfaceMgrRecordStatusMsg->record_status = (uint8_t)GF360_DataRecordStatus_e::GF360DRST_Idle;
        //_wsPtr->DTX_IMU_CtrlInterfaceMgrRecordStatusMsg->PostMessage();
        return error;
    }


    void DTX_IMU_RxMessageParser::Shutdown()
    {
        //_dataRecorder.closeLogFile();
    }

    bool DTX_IMU_RxMessageParser::parseOnOff(char* val)
    {
        bool On = false;
        if(val[1] == 'N')
            On = true;
        return On;
    }

    /**************************************************************
    int DTX_IMU_RxMessageParser::startNewDataRecordLogfile()
    {
        int error = 0;
        //First check to see if the log file is open and has the right index.
        bool open = _dataRecorder.IsLogFileOpen();
        bool isRightIndex = _dataRecorder.getFileIndexNumber() == _wsPtr->RecordControlMsg->file_index;

        if(!open || !isRightIndex)
        {
            _dataRecorder.closeLogFile();
            _wsPtr->DTX_IMU_CtrlInterfaceMgrRecordStatusMsg->Clear();
            _wsPtr->DTX_IMU_CtrlInterfaceMgrRecordStatusMsg->record_file_type = (uint8_t)GF360_RecordFileType_e::GF360RFT_GF360_DTXCTRL;
            _wsPtr->DTX_IMU_CtrlInterfaceMgrRecordStatusMsg->record_file_status = (uint8_t)GF360_RecordStatus_e::GF360RST_FileClosed;
            _wsPtr->DTX_IMU_CtrlInterfaceMgrRecordStatusMsg->record_status = (uint8_t)GF360_DataRecordStatus_e::GF360DRST_Idle;
            _wsPtr->DTX_IMU_CtrlInterfaceMgrRecordStatusMsg->file_index = _wsPtr->RecordControlMsg->file_index;

            _dataRecorder.setFileIndexNumber(_wsPtr->RecordControlMsg->file_index);
            _dataRecorderStdHeader.setFileGuid(_wsPtr->RecordControlMsg->file_guid);
            _dataRecorderStdHeader.VersionNumber = _wsPtr->RecordControlMsg->record_version;
            if(_dataRecorder.openNewLogFile())
            {
                _wsPtr->DTX_IMU_CtrlInterfaceMgrRecordStatusMsg->record_file_status = (uint8_t)GF360_RecordStatus_e::GF360RST_Error;
                error = 1;
            }
            else
            {
                _wsPtr->DTX_IMU_CtrlInterfaceMgrRecordStatusMsg->record_file_status = (uint8_t)GF360_RecordStatus_e::GF360RST_FileOpenAndFlushed;
            }
            _wsPtr->DTX_IMU_CtrlInterfaceMgrRecordStatusMsg->PostMessage();
        }
        return error;
    }

    int DTX_IMU_RxMessageParser::checkDataRecorderProcess()
    {
        int error = 0;
        _wsPtr->RecordControlMsg->FetchMessage();
        if(_wsPtr->RecordControlMsg->command != (uint8_t)_currentRecordCtrlState)
        {
            switch((GF360_RecordControlCmd_e)_wsPtr->RecordControlMsg->command)
            {
                case GF360_RecordControlCmd_e::GF360RCC_StartRecording:
                    //ensure there is a log file with the right index open.
                    startNewDataRecordLogfile();
                    if(_dataRecorder.IsLogFileOpen())
                    {
                        _wsPtr->DTX_IMU_CtrlInterfaceMgrRecordStatusMsg->record_guid = _wsPtr->RecordControlMsg->exercise_guid;
                        _wsPtr->DTX_IMU_CtrlInterfaceMgrRecordStatusMsg->record_status = (uint8_t)GF360_DataRecordStatus_e::GF360DRST_Recording;
                        _wsPtr->DTX_IMU_CtrlInterfaceMgrRecordStatusMsg->data_start_index = _dataRecorder.GetCurrentFileOffset();
                        _wsPtr->DTX_IMU_CtrlInterfaceMgrRecordStatusMsg->data_stop_index = _dataRecorder.GetCurrentFileOffset();
                        _wsPtr->DTX_IMU_CtrlInterfaceMgrRecordStatusMsg->record_file_status = (uint8_t)GF360_RecordStatus_e::GF360RST_FileOpenNotFlushed;

                        _exDataWrapperRecord.setExerciseGuid(_wsPtr->RecordControlMsg->exercise_guid);
                        _exDataWrapperRecord.ExerciseStatus = (GF360_ExerciseDataWrapperStatus_e)_wsPtr->RecordControlMsg->exercise_status;
                        _dataRecorder.writeDataRecord(_exDataWrapperRecord);
                        _recordExerciseData = true;
                        _wsPtr->DTX_IMU_CtrlInterfaceMgrRecordStatusMsg->PostMessage();
                    }
                break;
                case GF360_RecordControlCmd_e::GF360RCC_StopRecording:
                    if(_dataRecorder.IsLogFileOpen())
                    {
                        _exDataWrapperRecord.setExerciseGuid(_wsPtr->RecordControlMsg->exercise_guid);
                        _exDataWrapperRecord.ExerciseStatus = (GF360_ExerciseDataWrapperStatus_e)_wsPtr->RecordControlMsg->exercise_status;
                        _dataRecorder.writeDataRecord(_exDataWrapperRecord);
                        _dataRecorder.flushLogFile();

                        _wsPtr->DTX_IMU_CtrlInterfaceMgrRecordStatusMsg->record_status = (uint8_t)GF360_DataRecordStatus_e::GF360DRST_Idle;
                        _wsPtr->DTX_IMU_CtrlInterfaceMgrRecordStatusMsg->data_stop_index = _dataRecorder.GetCurrentFileOffset();
                        _wsPtr->DTX_IMU_CtrlInterfaceMgrRecordStatusMsg->record_file_status = (uint8_t)GF360_RecordStatus_e::GF360RST_FileOpenAndFlushed;
                        _wsPtr->DTX_IMU_CtrlInterfaceMgrRecordStatusMsg->PostMessage();
                    }
                    _recordExerciseData = false;
                    break;
                case GF360_RecordControlCmd_e::GF360RCC_CloseRecordFile:
                    _dataRecorder.closeLogFile();
                    _wsPtr->DTX_IMU_CtrlInterfaceMgrRecordStatusMsg->record_status = (uint8_t)GF360_DataRecordStatus_e::GF360DRST_Idle;
                    _wsPtr->DTX_IMU_CtrlInterfaceMgrRecordStatusMsg->record_file_status = (uint8_t)GF360_RecordStatus_e::GF360RST_FileClosed;
                    _wsPtr->DTX_IMU_CtrlInterfaceMgrRecordStatusMsg->PostMessage();
                    _recordExerciseData = false;
                    break;
                case GF360_RecordControlCmd_e::GF360RCC_OpenNewRecordFile:
                    startNewDataRecordLogfile();
                    _dataRecorder.flushLogFile();
                    _recordExerciseData = false;
                break;
                default:
                    break;
            }
            _currentRecordCtrlState = (GF360_RecordControlCmd_e)_wsPtr->RecordControlMsg->command;

        }

        if(_wsPtr->DTX_IMU_CtrlInterfaceMgrRecordStatusMsg->record_file_status != (uint8_t) _dataRecorder.getRecorderStatus()
                || _wsPtr->DTX_IMU_CtrlInterfaceMgrRecordStatusMsg->file_index != _dataRecorder.getFileIndexNumber())
        {
            _wsPtr->DTX_IMU_CtrlInterfaceMgrRecordStatusMsg->record_file_status = (uint8_t) _dataRecorder.getRecorderStatus();
            _wsPtr->DTX_IMU_CtrlInterfaceMgrRecordStatusMsg->file_index = _dataRecorder.getFileIndexNumber();
            _wsPtr->DTX_IMU_CtrlInterfaceMgrRecordStatusMsg->PostMessage();
        }
        return error;
    }
    ***********************************/

    void DTX_IMU_RxMessageParser::rxMsgHandler(dtiUtils::SerialCommMessage_t &msg)
    {
        if(MgrIsRunning)
        {
            //checkDataRecorderProcess();

            ++_totalNumberMsgs;
            //The TopicID or MsgId is defined in the PX4/DTX code for each message.
            try {
                switch (msg.MsgId)
                {
                    case 46:
                    {
                        //Text Message is not currently being handled.
                        break;
                    }
                    case 81:
                    {
                        //_wsPtr->GF360DTXSystemStatusMsg->DeSerialize((uint8_t *)msg.msgPtr, msg.MsgSize, 2);
                        //_wsPtr->GF360DTXSystemStatusMsg->SetTimeNow();
                        //_wsPtr->GF360DTXSystemStatusMsg->PostMessage();
                        break;
                    }
                    case 82:
                    {
                        // Managing connection to gf360 Ctrl board. This message tells the board that it
                        // is connected. If not, it will bounce the microRTPS client.
                        host_conn_req tmpMsg;
                        tmpMsg.DeSerialize((uint8_t *)msg.msgPtr, msg.MsgSize, 2);
                        tmpMsg.SetTimeNow();
                        auto repPtr = make_shared<host_conn_rep>();
                        repPtr->sys_id = 1;
                        repPtr->seq = tmpMsg.seq + 1;
                        repPtr->tc1 = tmpMsg.tc1;
                        auto rmsgPtr = dynamic_pointer_cast<Rabit::RabitMessage, host_conn_rep>(repPtr);
                        _mgrPtr->AddMessageToQueue("DTX_IMU_InterfaceManager", rmsgPtr);
                        break;
                    }
                    case 117:
                    {
                        //_wsPtr->GF360ExerciseCompleteMsg->DeSerialize((uint8_t *)msg.msgPtr, msg.MsgSize, 2);
                        //_wsPtr->GF360ExerciseCompleteMsg->SetTimeNow();
                        //_wsPtr->GF360ExerciseCompleteMsg->PostMessage();
                        break;
                    }
                    default:
                        LOGWARN("DTX IMU  Interface Msg Parser, Unknown MsgID=" << msg.MsgId);
                        break;
                }
            }
            catch(std::exception &e)
            {
                LOGERROR("DTX IMU Msg Parser: Error parsing MsgID=" << msg.MsgId
                                 << "Exception: " << e.what());
            }
        }
    }

    //A 16-bit CRC is the last three characters in B64 format on the message
    //string.
    bool DTX_IMU_RxMessageParser::checkMsgCRCOk(char* pCmd, int msgSize)
    {
        bool ok = false;
        msgSize -= 3;
        if(msgSize > 0)
        {
            //uint16_t msgCRC = base64ToUInt16(pCmd + msgSize);
            //uint16_t cmpCRC = Compute_CRC16(pCmd, msgSize);
            //ok = msgCRC == cmpCRC;
        }
        return ok;
    }


}
