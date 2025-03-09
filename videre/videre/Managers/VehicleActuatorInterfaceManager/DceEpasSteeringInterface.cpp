/* ****************************************************************
* Athr(s): Harry Direen PhD, Randy Direen Phd.
* DireenTech Inc.  (www.DireenTech.com)
* Date: June 2018
*
*******************************************************************/

#include "DceEpasSteeringInterface.h"


namespace  videre
{


    DceEpasSteeringInterface::DceEpasSteeringInterface(Rabit::RabitManager *mgrPtr,
                                                std::shared_ptr<ConfigData> config)
        : SteeringStatusDataRecord()
    {
        _mgrPtr = mgrPtr;
        _config = config;
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        SteeringTorqueCtrlMsg = std::make_shared<SteeringTorqueCtrlMessage>();
        SteeringStatusMsg = std::make_shared<DceEPASteeringStatusMessage>();
        _mgrPtr->AddPublishSubscribeMessage("SteeringTorqueCtrlMsg", SteeringTorqueCtrlMsg);
        _mgrPtr->AddPublishSubscribeMessage("SteeringStatusMsg", SteeringStatusMsg);

        _loggingControlMsg = std::make_shared<ImageLoggingControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage("ImageLoggingControlMessage", _loggingControlMsg);

        SteeringStatusDataRecord.SteeringStatusMsg = SteeringStatusMsg;

        //EPAS Torque Limit Trip level [10, 100]
        //If the TorqueA or TorqueB values which are nominally at 127 of 255
        //drop or jump in value greater than Abs(Torque - 127) > value... then an
        //over torque will be declared... which gives control back to the safety driver.
        EPASMaxSafetyDriverTorque = _config->GetConfigIntValue("VehicleActuatorInterface.EPASMaxSafetyDriverTorque", 50);
        EPASMaxSafetyDriverTorque = EPASMaxSafetyDriverTorque < 10 ? 10 : EPASMaxSafetyDriverTorque > 200 ? 200 : EPASMaxSafetyDriverTorque;

        EPASTorqueMapNo = _config->GetConfigIntValue("VehicleActuatorInterface.EPASTorqueMapNo", 3);
        EPASTorqueMapNo = EPASTorqueMapNo < 1 ? 1 : EPASTorqueMapNo > 5 ? 5 : EPASTorqueMapNo;

        EPASSteeringCenterVal = _config->GetConfigDoubleValue("VehicleActuatorInterface.EPASSteeringCenterVal", 127.5);
        EPASAngleToDegeesSF = _config->GetConfigDoubleValue("VehicleActuatorInterface.EPASAngleToDegeesSF", 0.532);
    }


    //Generate the CAN Message to send to the DCE EPAS Steering System
    //to control the steering torque.  The control is steeringTorquePecent
    //in the range of -100.0 to 100.0.  Negative numbers are steer to the left,
    //positive numbers are steer to the righ.
    void  DceEpasSteeringInterface::generateSteeringTorqueControlMsg(struct can_frame &canMsg,
                                        double steeringTorquePecent )
    {
        canMsg.can_id = SteeringControlID;
        int dval = (int) round((64.0/100.0) * steeringTorquePecent);
        dval = dval < -MaxSteeringTorqueVal ? -MaxSteeringTorqueVal : dval > MaxSteeringTorqueVal ? MaxSteeringTorqueVal : dval;
        uint8_t torqueA = (uint8_t)(0x80 + dval);
        uint8_t torqueB = (uint8_t)(0xFF - torqueA);
        canMsg.can_dlc = 8;
        canMsg.data[0] = (uint8_t)SteeringTorqueMap;
        canMsg.data[1] = torqueA;
        canMsg.data[2] = torqueB;
        canMsg.data[3] = 0;
        canMsg.data[4] = 0;
        canMsg.data[5] = 0;
        canMsg.data[6] = 0;
        canMsg.data[7] = 0;
    }

    void DceEpasSteeringInterface::generatoeDisableAutoSteeringControlMsg(struct can_frame &canMsg)
    {
        canMsg.can_id = SteeringControlID;
        uint8_t torqueA = (uint8_t)(0x80 + 0);
        uint8_t torqueB = (uint8_t)(0xFF - torqueA);
        canMsg.can_dlc = 3;
        canMsg.data[0] = (uint8_t)SteeringTorqueMap_e::STM_Disable;
        canMsg.data[1] = torqueA;
        canMsg.data[2] = torqueB;
    }

    void DceEpasSteeringInterface::processSteeringTorqueMsg(struct can_frame &canMsg)
    {
        SteeringStatusMsg->MotorTorquePercent = (double)canMsg.data[0];
        SteeringStatusMsg->PWMDutyCyclePercent = (double)canMsg.data[1];
        SteeringStatusMsg->MotorCurrentAmps = (double)canMsg.data[2];
        SteeringStatusMsg->SupplyVoltage = (double)canMsg.data[3];
        SteeringStatusMsg->SwitchPosition = (uint32_t)canMsg.data[4];
        SteeringStatusMsg->TempDegC = (double)canMsg.data[5];
        SteeringStatusMsg->TorqueA = (int)canMsg.data[6];
        SteeringStatusMsg->TorqueB = (int)canMsg.data[7];

        SteeringStatusMsg->DriverTorqueHit = false;
        int maxTorqueChange = abs(SteeringStatusMsg->TorqueA - 127);
        if( maxTorqueChange > EPASMaxSafetyDriverTorque)
        {
            SteeringStatusMsg->DriverTorqueHit = true;
        }
        maxTorqueChange = abs(SteeringStatusMsg->TorqueB - 127);
        if( maxTorqueChange > EPASMaxSafetyDriverTorque)
        {
            SteeringStatusMsg->DriverTorqueHit = true;
        }

        SteeringStatusMsg->PostMessage();
        logSteeringStatusData();
    }

    void DceEpasSteeringInterface::processSteeringAngleMsg(struct can_frame &canMsg)
    {
        double saDeg = (double)canMsg.data[0] - EPASSteeringCenterVal;
        saDeg = EPASAngleToDegeesSF * saDeg;
        SteeringStatusMsg->SteeringAngleDeg = saDeg;
        SteeringStatusMsg->SteeringTorqueMapSetting = (SteeringTorqueMap_e)canMsg.data[3];
        SteeringStatusMsg->ErrorCode = (uint32_t)canMsg.data[4];
        SteeringStatusMsg->StatusFlags = (uint32_t)canMsg.data[6];
        SteeringStatusMsg->LimitFlags = (uint32_t)canMsg.data[7];
        SteeringStatusMsg->PostMessage();
        logSteeringStatusData();
    }

    void DceEpasSteeringInterface::logSteeringStatusData()
    {
        if( CanRxDataRecorderPtr != nullptr && EnableLoggingSteeringStatus)
        {
            bool logMsgChanged = _loggingControlMsg->FetchMessage();
            if(_loggingControlMsg->EnableLogging)
            {
                CanRxDataRecorderPtr->writeDataRecord(SteeringStatusDataRecord);
            }
            else if(logMsgChanged && !_loggingControlMsg->EnableLogging)
            {
                CanRxDataRecorderPtr->closeLogFile();
            }
        }
    }

}
