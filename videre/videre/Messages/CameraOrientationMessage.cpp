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

#include "CameraOrientationMessage.h"
#include "GeoCoordinateSystem.h"
#include <math.h>
#include <vector>
#include <boost/math/constants/constants.hpp>

using namespace GeoCoordinateSystemNS;


namespace videre
{

    void CameraOrientationMessage::Clear()
    {
        CameraSteeringModeSPOI = false;
        SPOI_LatLonDist.Clear();
        SPOI_XYZCoord.Clear();
        CameraAzimuthElevationAngles.Clear();
        //Default camera orientation is straight down.
        CameraAzimuthElevationAngles.SetElevationAngleDegrees(-90.0);
        _cameraZoomPercent = 0;
    }

    std::unique_ptr<Rabit::RabitMessage> CameraOrientationMessage::Clone() const
    {
        auto clone = std::unique_ptr<CameraOrientationMessage>(new CameraOrientationMessage(*this));
        return std::move(clone);
    }

    bool CameraOrientationMessage::CopyMessage(Rabit::RabitMessage *msg)
    {
        Rabit::RabitMessage::CopyMessage(msg); // call baseclass
        if (msg->GetTypeIndex() == std::type_index(typeid(CameraOrientationMessage)))
        {
            //Ensure the Copy process does not loose the mesages'
            //publish subscribe reference.
            std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

            CameraOrientationMessage *coMsg = static_cast<CameraOrientationMessage *>(msg);
            *this = *coMsg;

            this->SetGlobalPublishSubscribeMessageRef(psmr);
            return true;
        }
        return false;
    }

    //Set the Lat/Lon and Altitude States.  Convert the Lat/Lon
    //to X-Y coordinates and store.
    void CameraOrientationMessage::SetSPOILatLonConvertToXY(const LatLonAltCoord_t &spoiLL)
    {
        SPOI_LatLonDist = spoiLL;
        GeoCoordinateSystem *gcs = GeoCoordinateSystem::GetGeoCoordinateSystemReference();
        SPOI_XYZCoord = gcs->LatLonAltToXYZ(spoiLL);
    }

    //Set the X-Y-Z coordinates and convert to the Lat/Lon coordinates.
    void CameraOrientationMessage::SetSPOIXYCoordinatesConverToLatLon(const XYZCoord_t &spoiXY)
    {
        SPOI_XYZCoord = spoiXY;
        GeoCoordinateSystem *gcs = GeoCoordinateSystem::GetGeoCoordinateSystemReference();
        SPOI_LatLonDist = gcs->XYZToLatLonAlt(spoiXY);
    }

    //Writes 6*8 + 1 = 49 bytes
    void CameraOrientationMessage::SerializeToByteArray(VidereUtils::ByteArrayWriterVidere &bw)
    {
        bw.writeByte(CameraSteeringModeSPOI == true ? 1 : 0);
        bw.writeLatLonAlt(SPOI_LatLonDist);
        //Note:  the XYZ Coordinates are redundant so don't write
        bw.writeAzimuthElevation(CameraAzimuthElevationAngles);
        bw.writeDouble(_cameraZoomPercent);
    }

    void CameraOrientationMessage::ReadFromByteArray(VidereUtils::ByteArrayReaderVidere &br)
    {
        CameraSteeringModeSPOI = br.readByte() != 0 ? true : false;

        LatLonAltCoord_t llaPos;
        llaPos = br.readLatLonAlt();
        SetSPOILatLonConvertToXY(llaPos);

        CameraAzimuthElevationAngles = br.readAzimuthElevation();
        _cameraZoomPercent = br.readDouble();
    }


}