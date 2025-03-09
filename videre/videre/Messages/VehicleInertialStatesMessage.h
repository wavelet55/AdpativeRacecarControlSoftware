/* ****************************************************************
 * Athr(s): Harry Direen PhD, Randy Direen Phd.
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: March. 2016
 *
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
 *******************************************************************/

#ifndef VIDERE_DEV_VEHICLEINERTIALSTATESMESSAGE_H
#define VIDERE_DEV_VEHICLEINERTIALSTATESMESSAGE_H

#include <iostream>
#include <string>
#include <RabitMessage.h>
#include "XYZCoord_t.h"
#include "LatLonAltStruct.h"
#include "RollPitchYaw_t.h"
#include "ByteArrayReaderWriterVidere.h"

using namespace GeoCoordinateSystemNS;

namespace videre
{
    //The Inertial States of the Vehicle typically sent
    //over from the vehicle.
    class VehicleInertialStatesMessage : public Rabit::RabitMessage
    {

    public:
        //The earth-frame Lat/Lon coordinates of the vehicle.
        //Altitude is in meters above the mean sea level.
        //If the XYZ coordinates are updated... .the LatLonAlt
        //must be manually updated so that the two items coorispond.
        LatLonAltCoord_t LatLonAlt;

        //XYZ earth-frame coordinates of the vehicle. The values
        //are in meters, and:
        //X is the East-West axis
        //Y is the North-South axis
        //Z coorisponds to the Altitude in Meters above
        //mean sea level.
        //The XYZ coordiantes must be manually updated
        //after changing the
        XYZCoord_t XYZCoordinates;

        //The XYZ earth-frame velocities of the vehicle
        //in meters per second where:
        //  X is the East-West velocity axis
        //  Y is the North-South axis
        //  Z is the down-up axis with positive velocities in the down direction.
        XYZCoord_t XYZVelocities;

        //The vehicle's roll, pitch and yaw angles.
        RollPitchYaw_t RollPitchYaw;

        //The rate of change of the roll, pitch and yaw angles.
        RollPitchYaw_t RollPitchYawRates;

        //Height Above Ground Level In Meters
        double HeightAGL = 0;

        //The GPS Time stamp when the states were captured.
        //(or updated)
        double GpsTimeStampSec = 0;

        VehicleInertialStatesMessage() : RabitMessage()
        {
            Clear();
        }

        VehicleInertialStatesMessage(const VehicleInertialStatesMessage& msg)
        {
            *this = msg;
        }

        void Clear();

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final;

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final;


        //Set the Lat/Lon and Altitude States.  Convert the Lat/Lon
        //to X-Y coordinates and store.
        void SetLatLonAltConvertToXYZ(const LatLonAltCoord_t &llaPos);

        //Set the X-Y-Z coordinates and convert to the Lat/Lon coordinates.
        void SetXYZCoordinatesConverToLatLonAlt(const XYZCoord_t &xyzPos);

        //Move the vehicle position and attitude states forware in time
        //based upon the velocity vectors and the given delta time in seconds.
        //This method assumes the XYZ Coordinates are set.  The method updates
        //the Lat/Lon coordinates accordingly.
        void MoveStatesForwardInTime(double deltaTsec);

        void SerializeToByteArray(VidereUtils::ByteArrayWriterVidere &bw);

        void ReadFromByteArray(VidereUtils::ByteArrayReaderVidere &br);

    };

}


#endif //VIDERE_DEV_VEHICLEINERTIALSTATESMESSAGE_H
