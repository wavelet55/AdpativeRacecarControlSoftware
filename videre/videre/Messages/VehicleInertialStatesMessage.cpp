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


#include "VehicleInertialStatesMessage.h"
#include "GeoCoordinateSystem.h"

using namespace GeoCoordinateSystemNS;

namespace videre
{



    void VehicleInertialStatesMessage::Clear()
    {
        LatLonAlt.Clear();
        XYZCoordinates.Clear();
        XYZVelocities.Clear();
        RollPitchYaw.Clear();
        RollPitchYawRates.Clear();
        HeightAGL = 0;
        GpsTimeStampSec = 0;
    }

    std::unique_ptr<Rabit::RabitMessage> VehicleInertialStatesMessage::Clone() const
    {
        auto clone = std::unique_ptr<VehicleInertialStatesMessage>(new VehicleInertialStatesMessage(*this));
        return std::move(clone);
    }

    bool VehicleInertialStatesMessage::CopyMessage(Rabit::RabitMessage *msg)
    {
        bool msgCopied = false;
        if (msg->GetTypeIndex() == std::type_index(typeid(VehicleInertialStatesMessage)))
        {
            //Ensure the Copy process does not loose the mesages'
            //publish subscribe reference.
            std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

            Rabit::RabitMessage::CopyMessage(msg); // call baseclass
            VehicleInertialStatesMessage *visMsg = static_cast<VehicleInertialStatesMessage *>(msg);
            *this = *visMsg;

            this->SetGlobalPublishSubscribeMessageRef(psmr);
            msgCopied = true;
        }
        return msgCopied;
    }

    //Set the Lat/Lon and Altitude States.  Convert the Lat/Lon
    //to X-Y coordinates and store.
    void VehicleInertialStatesMessage::SetLatLonAltConvertToXYZ(const LatLonAltCoord_t &llaPos)
    {
        LatLonAlt = llaPos;
        GeoCoordinateSystem *gcs = GeoCoordinateSystem::GetGeoCoordinateSystemReference();
        XYZCoordinates = gcs->LatLonAltToXYZ(llaPos);
    }

    //Set the X-Y-Z coordinates and convert to the Lat/Lon coordinates.
    void VehicleInertialStatesMessage::SetXYZCoordinatesConverToLatLonAlt(const XYZCoord_t &xyzPos)
    {
        XYZCoordinates = xyzPos;
        GeoCoordinateSystem *gcs = GeoCoordinateSystem::GetGeoCoordinateSystemReference();
        LatLonAlt = gcs->XYZToLatLonAlt(xyzPos);
    }

    //Move the vehicle position and attitude states forware in time
    //based upon the velocity vectors and the given delta time in seconds.
    //This method assumes the XYZ Coordinates are set.  The method updates
    //the Lat/Lon coordinates accordingly.
    void VehicleInertialStatesMessage::MoveStatesForwardInTime(double deltaTsec)
    {
        XYZCoordinates = XYZCoordinates + (XYZVelocities * deltaTsec);
        RollPitchYaw =  RollPitchYaw + RollPitchYawRates.RollPitchYawRatesXTimeSecs(deltaTsec);
        GeoCoordinateSystem *gcs = GeoCoordinateSystem::GetGeoCoordinateSystemReference();
        LatLonAlt = gcs->XYZToLatLonAlt(XYZCoordinates);
    }

    //Writes 14*8 = 112 bytes
    void VehicleInertialStatesMessage::SerializeToByteArray(VidereUtils::ByteArrayWriterVidere &bw)
    {
        bw.writeLatLonAlt(LatLonAlt);
        //Note:  the XYZ Coordinates are redundant so don't write
        bw.writeXYZ(XYZVelocities);
        bw.writeRollPitchYaw(RollPitchYaw);
        bw.writeRollPitchYaw(RollPitchYawRates);
        bw.writeDouble(HeightAGL);
        bw.writeDouble(GpsTimeStampSec);
    }

    void VehicleInertialStatesMessage::ReadFromByteArray(VidereUtils::ByteArrayReaderVidere &br)
    {
        LatLonAltCoord_t llaPos;
        llaPos = br.readLatLonAlt();
        SetLatLonAltConvertToXYZ(llaPos);

        XYZVelocities = br.readXYZ();
        RollPitchYaw = br.readRollPitchYaw();
        RollPitchYawRates = br.readRollPitchYaw();
        HeightAGL = br.readDouble();
        GpsTimeStampSec = br.readDouble();
    }



}