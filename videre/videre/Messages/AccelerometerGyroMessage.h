/* ****************************************************************
 * Athr(s): Harry Direen PhD, Randy Direen Phd.
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: June 2018
 *
  *******************************************************************/


#ifndef VIDERE_DEV_ACCELEROMETERGYROMESSAGE_H
#define VIDERE_DEV_ACCELEROMETERGYROMESSAGE_H

#include "global_defines.h"
#include "RollPitchYaw_t.h"
#include "../Managers/IMUCommManager/IMU_DataTypeDefs.h"
#include "XYZCoord_t.h"

using namespace MathLibsNS;

namespace videre
{

    //Camera Calibration Command Message
    class AccelerometerGyroMessage : public Rabit::RabitMessage
    {
    public:
        IMU_SensorNS::Imu_SensorId_e IMU_SensorID;

        XYZCoord_t AccelerationRates;

        XYZCoord_t GyroAngularRates;

        double IMUTimeStampSec;

    public:

        AccelerometerGyroMessage() : RabitMessage()
        {
            Clear();
        }

        AccelerometerGyroMessage(const AccelerometerGyroMessage &msg)
        {
            *this = msg;
        }

        void SetAccelerationRates(double x, double y, double z)
        {
            AccelerationRates.x = x;
            AccelerationRates.y = y;
            AccelerationRates.z = z;
        }


        void SetGyroAngularRatesRadPerSec(double xAxisRad, double yAxisRad, double zAxisRad)
        {
            GyroAngularRates.x = xAxisRad;
            GyroAngularRates.y = yAxisRad;
            GyroAngularRates.z = zAxisRad;
        }

        void Clear()
        {
            IMU_SensorID = IMU_SensorNS::Imu_SensorId_e::IMUSensor_NA;
            AccelerationRates.Clear();
            GyroAngularRates.Clear();
            IMUTimeStampSec = 0;
        }

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<AccelerometerGyroMessage>(
                    new AccelerometerGyroMessage(*this));
            return std::move(clone);
        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            bool msgCopied = false;
            if(msg->GetTypeIndex() == std::type_index(typeid(AccelerometerGyroMessage)))
            {
                //Ensure the Copy process does not loose the mesages'
                //publish subscribe reference.
                std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

                Rabit::RabitMessage::CopyMessage(msg); // call baseclass
                AccelerometerGyroMessage *visMsg = static_cast<AccelerometerGyroMessage *>(msg);
                *this = *visMsg;

                this->SetGlobalPublishSubscribeMessageRef(psmr);
                msgCopied = true;
            }
            return msgCopied;
        }

    };

}

#endif //VIDERE_DEV_ACCELEROMETERGYROMESSAGE_H
