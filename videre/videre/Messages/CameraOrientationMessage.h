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

#ifndef VIDERE_DEV_CAMERAORIENTATIONMESSAGE_H
#define VIDERE_DEV_CAMERAORIENTATIONMESSAGE_H

#include <iostream>
#include <string>
#include <RabitMessage.h>
#include "XYZCoord_t.h"
#include "LatLonAltStruct.h"
#include "RollPitchYaw_t.h"
#include "AzimuthElevation_t.h"
#include "ByteArrayReaderWriterVidere.h"

using namespace GeoCoordinateSystemNS;

namespace videre
{
    //The Inertial States of the Vehicle typically sent
    //over from the vehicle.
    class CameraOrientationMessage : public Rabit::RabitMessage
    {
    public:
        /// <summary>
        /// SPOI:  Sensor Point of Interest
        /// If CameraSteeringModeSPOI is true, the Camera Sensor is being
        /// actively pointed to a specific target location.  The Lat/Lon or
        /// X-Y position inforation is the location (typically on the ground) the
        /// camera sensor is being actively pointed at.  In this case, the vehicle
        /// attitude information (pitch/roll/yaw) and camera Azmuth/Elevation angles,
        /// along with velocity information, should be ignored.  The Lat/Lon or X-Y position
        /// information provides the location the camera is pointed at.
        ///
        /// If CameraSteeringModeSPOI is false, the camera sensor is pointed (azmuth/elevation)
        /// relative to the vehicle position and attitude.
        /// </summary>
        bool CameraSteeringModeSPOI = false;

        //The Lat/Lon Point on the ground the camera is pointing at.
        //The Alt value may be used as distance from the UAV to the
        //ground SPOI location.
        LatLonAltCoord_t SPOI_LatLonDist;

        //The X-Y Point on the ground the camera is pointing at.
        //The Z value may be used as distance from the UAV to the
        //ground SPOI location.
        XYZCoord_t SPOI_XYZCoord;

        AzimuthElevation_t CameraAzimuthElevationAngles;

    private:
        /// <summary>
        /// Camera Zoom angle in percent [0 to 100]
        /// 100% is maximum zoom, 0 percent is minimum
        /// or no zoom.
        /// </summary>
        double _cameraZoomPercent = 0;

    public:

        CameraOrientationMessage() : RabitMessage()
        {
            Clear();
        }

        CameraOrientationMessage(const CameraOrientationMessage& msg)
        {
            *this = msg;
        }

        void Clear();

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final;

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final;


        //Set the SPOI Lat/Lon positions.  Convert the Lat/Lon
        //to X-Y coordinates and store.
        void SetSPOILatLonConvertToXY(const LatLonAltCoord_t &spoiLL);

        //Set the SPOI X-Y coordinates and convert to the Lat/Lon coordinates.
        void SetSPOIXYCoordinatesConverToLatLon(const XYZCoord_t &spoiXY);


        double CameraZoomPercent()
        {
            return _cameraZoomPercent;
        }


        void SetCameraZoomPercent(double value)
        {
            _cameraZoomPercent = value < 0 ? 0 : value > 100.0 ? 100.0 : value;
        }

        void SerializeToByteArray(VidereUtils::ByteArrayWriterVidere &bw);

        void ReadFromByteArray(VidereUtils::ByteArrayReaderVidere &br);


    };

}


#endif //VIDERE_DEV_CameraOrientationMessage_H

