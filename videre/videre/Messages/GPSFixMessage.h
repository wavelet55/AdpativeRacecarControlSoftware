/* ****************************************************************
 * Athr(s): Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: May, 2018
 *
 * NeuroGroove GPS Interface
 *******************************************************************/


#ifndef VIDERE_DEV_GPSFIXMESSAGE_H
#define VIDERE_DEV_GPSFIXMESSAGE_H

#include <iostream>
#include <string>
#include <RabitMessage.h>
#include "XYZCoord_t.h"
#include "LatLonAltStruct.h"
#include "RollPitchYaw_t.h"
#include "ByteArrayReaderWriterVidere.h"
#include "../NemaTodeGpsParser/GPSFix.h"
#include "../GeoCoordinateSystem/GeoCoordinateSystem.h"


using namespace GeoCoordinateSystemNS;

namespace videre
{
    //GPS Fix data from GPS Module
    class GPSFixMessage : public Rabit::RabitMessage
    {

    public:
        nmea::GPSFix GPSFixData;

        //X-Axis is East-West with East being postive
        //Y-Axis is North-South with North being positive
        //Z-Axis is positive up... relative to the Mean-Sea-Level.
        //X-Y Axis origen is set in the GeoCoordinateSystem.
        XYZCoord_t XYZPositionMeters;

        //The GPS Pod does not measure the velocity in the Z direction
        //so this coord. will be zero.
        XYZCoord_t  XYZVelocityMetersPerSec;

        GPSFixMessage() : RabitMessage()
        {
            Clear();
        }

        GPSFixMessage(const GPSFixMessage& msg)
        {
            *this = msg;
        }

        void Clear();

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final;

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final;

        //SetGPSFix
        //This method copies the gpxFix into the message and computes
        //the X-Y Coordinates from the GPS Lat/Lon data.
        void SetGPSFix(nmea::GPSFix &gpsFix);

    };

}
#endif //VIDERE_DEV_GPSFIXMESSAGE_H
