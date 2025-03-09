/* ****************************************************************
 * Athr(s): Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: May, 2018
 *
 * NeuroGroove GPS Interface
 *******************************************************************/


#include "GPSFixMessage.h"

using namespace nmea;
using namespace std;

namespace videre
{

    void GPSFixMessage::Clear()
    {
        XYZPositionMeters.Clear();
        XYZVelocityMetersPerSec.Clear();
    }

    //SetGPSFix
    //This method copies the gpxFix into the message and computes
    //the X-Y Coordinates from the GPS Lat/Lon data.
    void GPSFixMessage::SetGPSFix(nmea::GPSFix &gpsFix)
    {
        LatLonAltCoord_t lla;
        GPSFixData = gpsFix;
        GeoCoordinateSystem * gcsPtr = GeoCoordinateSystem::GetGeoCoordinateSystemReference();
        lla.SetLatitudeDegrees(gpsFix.latitude);
        lla.SetLongitudeDegrees(gpsFix.longitude);
        lla.SetAltitude(gpsFix.altitude);
        XYZPositionMeters = gcsPtr->LatLonAltToXYZ(lla);
        //change the Z-position to be relative to the reference Altitude
        XYZPositionMeters.z = gpsFix.altitude - gcsPtr->ReferenceLatLonAltLocation().Altitude();

        double speedMetersPerSec = 0.27777777777 * gpsFix.speed;   //(0.277777 = 1000.0/3600.0)
        XYZVelocityMetersPerSec.SetHeadingDegMagnitudeXY(gpsFix.travelAngle, speedMetersPerSec);
        XYZVelocityMetersPerSec.z = 0;
    }


    std::unique_ptr<Rabit::RabitMessage> GPSFixMessage::Clone() const
    {
        auto clone = std::unique_ptr<GPSFixMessage>(new GPSFixMessage(*this));
        return std::move(clone);
    }

    bool GPSFixMessage::CopyMessage(Rabit::RabitMessage *msg)
    {
        bool msgCopied = false;
        if (msg->GetTypeIndex() == std::type_index(typeid(GPSFixMessage)))
        {
            //Ensure the Copy process does not loose the mesages'
            //publish subscribe reference.
            std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

            Rabit::RabitMessage::CopyMessage(msg); // call baseclass
            GPSFixMessage *visMsg = static_cast<GPSFixMessage *>(msg);
            *this = *visMsg;

            this->SetGlobalPublishSubscribeMessageRef(psmr);
            msgCopied = true;
        }
        return msgCopied;
    }



}