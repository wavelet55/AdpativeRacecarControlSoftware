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

#include "GeoCoordinateSystem.h"
#include "LatLonAltStruct.h"
#include "XYZCoord_t.h"
#include "LatLonUtmTransformations.h"
#include <memory>


namespace GeoCoordinateSystemNS
{

    //Declare the Static _GeoCoordinateSystemPtr
    GeoCoordinateSystem *GeoCoordinateSystem::_GeoCoordinateSystemPtr;

    /// <summary>
    /// Setup a GeoCoordinateSystem
    /// The reference Latitude and Longitude is the location where
    /// the X-Y coordinate will be zero (0,0).
    /// The Alititude is optional.  If provided it establishs the
    /// nominal ground level at the reference point.
    /// </summary>
    /// <param name="latLonAlt"></param>
    /// <param name="conversionType"></param>
    bool GeoCoordinateSystem::SetupGeoCoordinateSystem(LatLonAltCoord_t &latLonAlt,
                                  GeoCoordinateSystemConversionType_e conversionType)
    {
        bool error = false;
        UTMCoord_t utmXY;

        _referenceLocation = latLonAlt;
        _GeoCoordinateSystemConversionType = conversionType;
        _UtmWgs84XYCenter.Clear();
        switch (conversionType)
        {
            case GeoCoordinateSystemConversionType_e::Linear:
            {
                std::vector<LatLonAltCoord_t> latLonList(1);
                latLonList[0] = _referenceLocation;
                SetupGeoCoordinateSystem(latLonList, conversionType);
                break;
            }

            case GeoCoordinateSystemConversionType_e::WGS84_Relative:
            DefaultCoordSystemSetup:
                utmXY = LatLonUtmWgs84Conv::LLtoUTM_Degrees(_referenceLocation.LatitudeDegrees(),
                                                            _referenceLocation.LongitudeDegrees(), 0, true,
                                                            _referenceLocation.LongitudeDegrees());
                _UtmWgs84XYCenter.x = utmXY.UTMEasting;
                _UtmWgs84XYCenter.y = utmXY.UTMNorthing;
                _UTMZoneNumberAtRefLatLon = utmXY.UTMZoneNumber;
                _UTMZoneLatDesAtRefLatLon = utmXY.UTMZoneLatDes;
                _IsCoordinateSystemValid = true;
                break;

            case GeoCoordinateSystemConversionType_e::WGS84_Map:
                utmXY = LatLonUtmWgs84Conv::LLtoUTM_Degrees(_referenceLocation.LatitudeDegrees(),
                                                            _referenceLocation.LongitudeDegrees(), 0, false,
                                                            _referenceLocation.LongitudeDegrees());
                _UtmWgs84XYCenter.x = utmXY.UTMEasting;
                _UtmWgs84XYCenter.y = utmXY.UTMNorthing;
                _UTMZoneNumberAtRefLatLon = utmXY.UTMZoneNumber;
                _UTMZoneLatDesAtRefLatLon = utmXY.UTMZoneLatDes;
                _IsCoordinateSystemValid = true;
                break;

            default:
            {
                std::vector<LatLonAltCoord_t> latLonList(1);
                latLonList[0] = _referenceLocation;
                SetupGeoCoordinateSystem(latLonList, conversionType);
                break;
            }
;
        }
        return error;
    }

    /// <summary>
    /// Setup the GeoCoordinateSystem
    /// Pass in a list of Lat/Lon locations that cover the extent of the region
    /// the vehicle is expected to operate over.  The center of this region will be used
    /// to set the center of the of the X-Y coordinate system (0,0).
    /// The extent of the region will be used to esblishes the linearization of the
    /// Lat/Lon to X-Y conversion if useLinearConversion = true;
    /// </summary>
    /// <param name="latLonList"></param>
    /// <param name="useLinearConversion"></param>
    bool GeoCoordinateSystem::SetupGeoCoordinateSystem(std::vector<LatLonAltCoord_t> &latLonAltList,
                                  GeoCoordinateSystemConversionType_e conversionType)
    {
        bool error = false;
        double refAlt = _referenceLocation.Altitude();
        if (latLonAltList.size() < 1)
        {
            return true;     //No valid data
        }
        if (latLonAltList.size() > 1)
        {
            _referenceLocation = LatLonAltCoord_t::FindCenterOfSetOfLatLonPoints(latLonAltList);
        }
        else
        {
            _referenceLocation = latLonAltList[0];
        }
        //Restore Altitude info lost in the above steps.
        _referenceLocation.SetAltitude(refAlt);

        //Set the Center UTM Location
        _UtmWgs84XYCenter.Clear();
        bool relLatLon = conversionType == GeoCoordinateSystemConversionType_e::WGS84_Map ? false : true;
        UTMCoord_t utmXY = LatLonUtmWgs84Conv::LLtoUTM_Degrees(_referenceLocation.LatitudeDegrees(),
                                                              _referenceLocation.LongitudeDegrees(), 0,
                                                               relLatLon, _referenceLocation.LongitudeDegrees());
        _UtmWgs84XYCenter.x = utmXY.UTMEasting;
        _UtmWgs84XYCenter.y = utmXY.UTMNorthing;
        _UTMZoneNumberAtRefLatLon = utmXY.UTMZoneNumber;
        _UTMZoneLatDesAtRefLatLon = utmXY.UTMZoneLatDes;

        if (conversionType == GeoCoordinateSystemConversionType_e::Linear)
        {
            //We temporarily need the convertion type to be WGS84_Relative
            //in order to get various xy and lat/lon points to setup the
            //linear scale factors... we will set it back to linear after the setup process.
            _GeoCoordinateSystemConversionType = GeoCoordinateSystemConversionType_e::WGS84_Relative;
            LatLonAltCoord_t maxLatLon;
            LatLonAltCoord_t minLatLon;
            LatLonAltCoord_t delLatLon;
            XYZCoord_t xyMax;
            XYZCoord_t xyMin;
            XYZCoord_t delXY;
            if (latLonAltList.size() < 2)
            {
                //Generate corner locations... assume +/- 1 km area
                xyMax = XYZCoord_t(1000.0, 1000.0);
                maxLatLon = XYZToLatLonAlt(xyMax);
                xyMin = XYZCoord_t(-1000.0, -1000.0);
                minLatLon = XYZToLatLonAlt(xyMin);
            }
            else
            {
                maxLatLon = LatLonAltCoord_t::FindMaxNortEastCornerOfSetOfLatLonPoints(latLonAltList);
                minLatLon = LatLonAltCoord_t::FindMinSouthWestCornerOfSetOfLatLonPoints(latLonAltList);
                //don't use the full extent... use a partial exent... so that on average
                //the linear approximation will be good.
                delLatLon = (maxLatLon - minLatLon) * 0.5;
                maxLatLon = _referenceLocation + (delLatLon * 0.75);
                minLatLon = _referenceLocation - (delLatLon * 0.75);

                xyMax = LatLonAltToXYZ(maxLatLon);
                xyMin = LatLonAltToXYZ(minLatLon);
                delXY = xyMax - xyMin;
                if (delXY.x < 100.0 || delXY.y < 100.0)
                {
                    //Too small of an area..generate a more reasonable size
                    if (delXY.x < 100.0)
                    {
                        xyMax.x = 50.0;
                        xyMin.x = -50.0;
                    }
                    if (delXY.y < 100.0)
                    {
                        xyMax.y = 50.0;
                        xyMin.y = -50.0;
                    }
                    maxLatLon = XYZToLatLonAlt(xyMax);
                    minLatLon = XYZToLatLonAlt(xyMin);
                }
            }
            delLatLon = maxLatLon - minLatLon;
            delXY = xyMax - xyMin;
            _xToLonCF = (delLatLon.LongitudeRadians()) / delXY.x;
            _lonToXCF = delXY.x / (delLatLon.LongitudeRadians());
            _yToLatCF = (delLatLon.LatitudeRadians()) / delXY.y;
            _latToYCF = delXY.y / (delLatLon.LatitudeRadians());
            _GeoCoordinateSystemConversionType = GeoCoordinateSystemConversionType_e::Linear;
        }
        _IsCoordinateSystemValid = true;
        return error;
    }

    /// <summary>
    /// Setup the GeoCoordinateSystem
    /// This is used when the Linear Conversion Factors are known.
    /// </summary>
    /// <param name="latLonList"></param>
    /// <param name="LatitudeRadToY"></param>
    /// <param name="LongitudeRadToX"></param>
    bool GeoCoordinateSystem::SetupLinearGeoCoordinateSystemFromConvFactors(LatLonAltCoord_t &refLatLonAlt,
                                                       double LatitudeRadToY, double LongitudeRadToX)
    {
        bool error = false;
        _GeoCoordinateSystemConversionType = GeoCoordinateSystemConversionType_e::Linear;
        _referenceLocation = refLatLonAlt;
        _lonToXCF = LatitudeRadToY;
        _latToYCF = LongitudeRadToX;
        if( LatitudeRadToY > 1.0 && LongitudeRadToX > 1.0 )
        {
            _xToLonCF = 1.0 / LongitudeRadToX;
            _yToLatCF = 1.0 / LatitudeRadToY;
            _IsCoordinateSystemValid = true;
        }
        else
        {
            _xToLonCF = 2.007e-7;
            _yToLatCF = 1.573e-7;
            _lonToXCF = 4.982e6;
            _latToYCF = 6.358e6;
            _IsCoordinateSystemValid = false;
            error = true;
        }
        return error;
    }


    /// <summary>
    /// Convert Latitude/Longitude to the X-Y Coordinate system.
    /// The Center of the X-Y Coordinate system is the Reference (Lat/Lon) Location
    /// which is were X-Y is (0,0)
    /// </summary>
    /// <param name="latLon"></param>
    /// <param name="forceUTMTransformation">Optional... normally false.</param>
    /// <returns></returns>
    XYZCoord_t GeoCoordinateSystem::LatLonAltToXYZ(const LatLonAltCoord_t &latLonAlt)
    {
        XYZCoord_t xyzPos;
        LatLonAltCoord_t delLatLon;
        //xyPos.GeoCoordinateSys = this;
        UTMCoord_t utmXY;
        switch (_GeoCoordinateSystemConversionType)
        {
            case GeoCoordinateSystemConversionType_e::Linear:
                //Linear Transormation... used for speed of operation.
                delLatLon = latLonAlt - _referenceLocation;
                xyzPos.x = _lonToXCF * delLatLon.LongitudeRadians();
                xyzPos.y = _latToYCF * delLatLon.LatitudeRadians();
                xyzPos.z = latLonAlt.Altitude();
                break;

            case GeoCoordinateSystemConversionType_e::WGS84_Relative:
                utmXY = LatLonUtmWgs84Conv::LLtoUTM_Degrees(latLonAlt.LatitudeDegrees(),
                                                           latLonAlt.LongitudeDegrees(), 0, true,
                                                           _referenceLocation.LongitudeDegrees());
                xyzPos.x = utmXY.UTMEasting - _UtmWgs84XYCenter.x;
                xyzPos.y = utmXY.UTMNorthing - _UtmWgs84XYCenter.y;
                xyzPos.z = latLonAlt.Altitude();
                break;

            case GeoCoordinateSystemConversionType_e::WGS84_Map:
                utmXY = LatLonUtmWgs84Conv::LLtoUTM_Degrees(latLonAlt.LatitudeDegrees(),
                                                           latLonAlt.LongitudeDegrees(), 0, false,
                                                           _referenceLocation.LongitudeDegrees());
                xyzPos.x = utmXY.UTMEasting;
                xyzPos.y = utmXY.UTMNorthing;
                xyzPos.z = latLonAlt.Altitude();
                break;

            default:  //Assume Linear
                delLatLon = latLonAlt - _referenceLocation;
                xyzPos.x = _lonToXCF * delLatLon.LongitudeRadians();
                xyzPos.y = _latToYCF * delLatLon.LatitudeRadians();
                xyzPos.z = latLonAlt.Altitude();
        }
        return xyzPos;
    }


    /// <summary>
    /// Conver an X-Y coordinate to the Latitude/Longitude Location.
    /// The Center of the X-Y Coordinate system is the Reference (Lat/Lon) Location
    /// which is were X-Y is (0,0)
    /// </summary>
    /// <param name="xyCoord"></param>
    /// <param name="forceUTMTransformation"></param>
    /// <returns></returns>
    LatLonAltCoord_t GeoCoordinateSystem::XYZToLatLonAlt(const XYZCoord_t &xyzCoord)
    {
        LatLonAltCoord_t latLonAltPt;
        //latLonAltPt.GeoCoordinateSys = this;
        UTMCoord_t utmXY;
        switch (_GeoCoordinateSystemConversionType)
        {
            case GeoCoordinateSystemConversionType_e::Linear:
                //Linear Transformation... used for speed of operation.
                latLonAltPt.SetLongitudeRadians(_referenceLocation.LongitudeRadians() + _xToLonCF * xyzCoord.x);
                latLonAltPt.SetLatitudeRadians(_referenceLocation.LatitudeRadians() + _yToLatCF * xyzCoord.y);
                latLonAltPt.SetAltitude(xyzCoord.z);
                break;

            case GeoCoordinateSystemConversionType_e::WGS84_Relative:
            xyToLatLonDefaultConversion:
                utmXY.UTMEasting = _UtmWgs84XYCenter.x + xyzCoord.x;
                utmXY.UTMNorthing = _UtmWgs84XYCenter.y + xyzCoord.y;
                utmXY.UTMZoneNumber = UTMZoneNumberAtRefLatLon();
                utmXY.UTMZoneLatDes = UTMZoneLatDesAtRefLatLon();
                utmXY.CenterLongitudeDegrees = _referenceLocation.LongitudeDegrees();
                utmXY.UTMRelativeToCenterLongitude = true;
                latLonAltPt = LatLonUtmWgs84Conv::UTMtoLL_Degrees(utmXY);
                latLonAltPt.SetAltitude(xyzCoord.z);
                break;

            case GeoCoordinateSystemConversionType_e::WGS84_Map:
                utmXY.UTMEasting = xyzCoord.x;
                utmXY.UTMNorthing = xyzCoord.y;
                utmXY.UTMZoneNumber = UTMZoneNumberAtRefLatLon();
                utmXY.UTMZoneLatDes = UTMZoneLatDesAtRefLatLon();
                utmXY.CenterLongitudeDegrees = _referenceLocation.LongitudeDegrees();
                utmXY.UTMRelativeToCenterLongitude = false;
                latLonAltPt = LatLonUtmWgs84Conv::UTMtoLL_Degrees(utmXY);
                latLonAltPt.SetAltitude(xyzCoord.z);
                break;

            default:
                goto xyToLatLonDefaultConversion;
        }
        return latLonAltPt;
    }

    //Haversine
    //formula:a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
    //c = 2 ⋅ atan2( √a, √(1−a) )
    //d = R ⋅ c
    //φ is latitude, λ is longitude, R is earth’s radius (mean radius = 6,371km);
    double GeoCoordinateSystem::HaversineDistanceBetweenLatLonPointsRad(LatLonAltCoord_t &pt1, LatLonAltCoord_t &pt2)
    {
        double dist = 0;
        LatLonAltCoord_t deltaLatLon = pt2 - pt1;
        double sinSqLat = sin(0.5 * deltaLatLon.LatitudeRadians());
        sinSqLat = sinSqLat * sinSqLat;
        double sinSqLon = sin(0.5 * deltaLatLon.LongitudeRadians());
        sinSqLon = sinSqLon * sinSqLon;
        double a = sinSqLat + cos(pt1.LatitudeRadians()) * cos(pt2.LatitudeRadians()) * sinSqLon;
        double c = 2.0 * atan2( sqrt(a), sqrt(1 - a) );
        dist = EarthRadiusAveMeters * c;
        //dist = EarthRadiusWGS84Meters * c;
        return dist;
    }


}