/* ****************************************************************
 * Athr(s): Harry Direen PhD, Randy Direen Phd.
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: June 2018
 *
  *******************************************************************/

#ifndef VIDERE_DEV_DCEEPASSTEERINGINTERFACE_H
#define VIDERE_DEV_DCEEPASSTEERINGINTERFACE_H

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <linux/can.h>
#include <RabitManager.h>
#include "global_defines.h"
#include "../../Utils/logger.h"
#include "../../Utils/timing.h"
#include "../../Utils/config_data.h"
#include "SteeringTorqueCtrlMessage.h"
#include "DceEPASteeringStatusMessage.h"
#include "DataRecorder.h"
#include "EpasSteeringDataRecords.h"
#include "ImageLoggingControlMessage.h"

namespace videre
{

    class DceEpasSteeringInterface
    {
    public:
        const uint32_t SteeringTorqueRptID = 0x0290;
        const uint32_t SteeringAngleRptID = 0x0292;
        const uint32_t SteeringControlID = 0x0296;

        std::shared_ptr<SteeringTorqueCtrlMessage> SteeringTorqueCtrlMsg;
        std::shared_ptr<DceEPASteeringStatusMessage> SteeringStatusMsg;
        std::shared_ptr<ImageLoggingControlMessage> _loggingControlMsg;

        SteeringTorqueMap_e SteeringTorqueMap = SteeringTorqueMap_e::STM_L3;

        DataRecorder *CanRxDataRecorderPtr = nullptr;
        EpasSteeringStatusDataRecord SteeringStatusDataRecord;

        bool EnableLoggingSteeringStatus = true;

        void setSteeringTorqueMap(uint32_t mapNo)
        {
            mapNo = mapNo > 5 ? 5 : mapNo;
            SteeringTorqueMap = (SteeringTorqueMap_e)mapNo;
        }

        //EPAS Torque Limit Trip level [10, 100]
        //If the TorqueA or TorqueB values which are nominally at 127 of 255
        //drop or jump in value greater than Abs(Torque - 127) > value... then an
        //over torque will be declared... which gives control back to the safety driver.
        int EPASMaxSafetyDriverTorque = 50;

        int EPASTorqueMapNo = 3;

        //The Center steering angle POT reading
        //The EPAS unit has a 1 byte value for steering angle
        //where center is about 127 if the pot is properly centered.
        //The EPASAngleToDegeesSF is a scale factor to convert
        //the EPAS steering angle value to degrees.
        double EPASSteeringCenterVal = 127.5;
        double EPASAngleToDegeesSF = 0.532;


    private:

        const int32_t MaxSteeringTorqueVal = 64;

        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        //A reference to the CommManger... primarily used during setup of this
        //class object.
        Rabit::RabitManager *_mgrPtr;
        std::shared_ptr<ConfigData> _config;

    public:


        DceEpasSteeringInterface(Rabit::RabitManager *mgrPtr,
                                 std::shared_ptr<ConfigData> confi);


        //Generate the CAN Message to send to the DCE EPAS Steering System
        //to control the steering torque.  The control is steeringTorquePecent
        //in the range of -100.0 to 100.0.  Negative numbers are steer to the left,
        //positive numbers are steer to the righ.
        void  generateSteeringTorqueControlMsg(struct can_frame &canMsg,
                                                 double steeringTorquePecent);

        void generatoeDisableAutoSteeringControlMsg(struct can_frame &canMsg);

        void processSteeringTorqueMsg(struct can_frame &canMsg);

        void processSteeringAngleMsg(struct can_frame &canMsg);

        void logSteeringStatusData();
    };

}
#endif //VIDERE_DEV_DCEEPASSTEERINGINTERFACE_H
