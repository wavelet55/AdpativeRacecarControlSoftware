/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Oct 2016
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
  *******************************************************************/

#include "ByteArrayReaderWriterVidere.h"
#include <vector>

using namespace std;
using namespace GeoCoordinateSystemNS;

namespace VidereUtils
{


    bool ByteArrayWriterVidere::writeXYZ(XYZCoord_t &value)
    {
        bool error = false;
        error |= addBytesToArray((byte*)&value.x, sizeof(double));
        error |= addBytesToArray((byte*)&value.y, sizeof(double));
        error |= addBytesToArray((byte*)&value.z, sizeof(double));
        return error;
    }

    bool ByteArrayWriterVidere::writeLatLonAlt(LatLonAltCoord_t &value)
    {
        bool error = false;
        double val = value.LatitudeRadians();
        error |= addBytesToArray((byte*)&val, sizeof(double));
        val = value.LongitudeRadians();
        error |= addBytesToArray((byte*)&val, sizeof(double));
        val = value.Altitude();
        error |= addBytesToArray((byte*)&val, sizeof(double));
        return error;
    }

    bool ByteArrayWriterVidere::writeRollPitchYaw(RollPitchYaw_t &value)
    {
        bool error = false;
        double val = value.RollRadians();
        error |= addBytesToArray((byte*)&val, sizeof(double));
        val = value.PitchRadians();
        error |= addBytesToArray((byte*)&val, sizeof(double));
        val = value.YawRadians();
        error |= addBytesToArray((byte*)&val, sizeof(double));
        return error;
    }

    bool ByteArrayWriterVidere::writeAzimuthElevation(AzimuthElevation_t &value)
    {
        bool error = false;
        double val = value.AzimuthAngleRad();
        error |= addBytesToArray((byte*)&val, sizeof(double));
        val = value.ElevationAngleRad();
        error |= addBytesToArray((byte*)&val, sizeof(double));
        return error;
    }


 /************* Byte Array Reader *****************/


    XYZCoord_t ByteArrayReaderVidere::readXYZ()
    {
        XYZCoord_t xyz;
        double value = 0;
        getBytes((byte*)&value, sizeof(double));
        xyz.x = value;
        getBytes((byte*)&value, sizeof(double));
        xyz.y = value;
        getBytes((byte*)&value, sizeof(double));
        xyz.z = value;
        return xyz;
    }

    LatLonAltCoord_t ByteArrayReaderVidere::readLatLonAlt()
    {
        LatLonAltCoord_t latLon;
        double value = 0;
        getBytes((byte*)&value, sizeof(double));
        latLon.SetLatitudeRadians(value);
        getBytes((byte*)&value, sizeof(double));
        latLon.SetLongitudeRadians(value);
        getBytes((byte*)&value, sizeof(double));
        latLon.SetAltitude(value);
        return latLon;
    }

    RollPitchYaw_t  ByteArrayReaderVidere::readRollPitchYaw()
    {
        RollPitchYaw_t rpy;
        double value = 0;
        getBytes((byte*)&value, sizeof(double));
        rpy.SetRollRadians(value);
        getBytes((byte*)&value, sizeof(double));
        rpy.SetPitchRadians(value);
        getBytes((byte*)&value, sizeof(double));
        rpy.SetYawRadians(value);
        return rpy;
    }

    AzimuthElevation_t ByteArrayReaderVidere::readAzimuthElevation()
    {
        AzimuthElevation_t ae;
        double value = 0;
        getBytes((byte*)&value, sizeof(double));
        ae.SetAzimuthAngleRad(value);
        getBytes((byte*)&value, sizeof(double));
        ae.SetElevationAngleRad(value);
        return ae;
    }


}