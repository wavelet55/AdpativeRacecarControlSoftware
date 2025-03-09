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
 *
 * Desc: This class establishes the phyical/geo coordinate system
 * used by the UAV during a mission.  The coordinate system will be
 * centered on the mid point of the search area.  The mid point
 * between the Min and Max latitudes and longitudes will be choosen
 * as (0,0) on an x-y plane.  The x axis will be the east-west axis
 * with east being positive, and west being negative.  The y axis
 * will be the north-south axis with north being positive and south
 * being negative.
 *
 * Distances are measured in meters.
 * Latitude longitude may be provided in degrees or radians.
 * Positive longitude represent east of the prime meridian,
 * negative longitude represent west of the prime meridian.
 * Positve latitude is north of the equator
 * and negative latitude is south of the equator.
 *
 * The GeoCoordinate System will support a linear aproximation to
 * Lat/Lon to X-Y conversions and supports WGS-84 UTM conversions.
 * If the mission area is relatively small (less that a few kilometers)
 * the linear conversion process should be sufficient and substantially
 * reduces the amount of calculation require for conversions.
 * For larger areas, use the full conversion process... a conversion on
 * a fast Intel Core i7 is on the order of 1 - 2 microseconds.
 *
 * STANAG 4586 Notes:
 * All earth-fixed position references shall be expressed in the latitude-longitude
 * system with respect to the WGS-84 ellipsoid in units of radians using double
 * precision floating-point numbers. Representations in other systems, such as
 * Universal Transverse Mercator (UTM), shall be converted at the point of use.
 *
 * HOPS Time will follow the STANAG 4586 convention:
 * All times shall be represented in Universal Time Coordinated (UTC) in seconds
 * since Jan 1, 1970 using IEEE double precision floating point numbers.
 * To ensure consistant time, the UAVTime object must be use.  This
 * system will be synchronized with the GPS system clock.
 *
 *
 * The methods in this class are thread safe.  The coordinate system
 * must not be used until it has been set up with misison parameters.
 * Once setup, the coordinate system cannot change during the course
 * of a mission.
 *******************************************************************/

#ifndef VIDERE_DEV_GEOCOORDINATESYSTEM_H
#define VIDERE_DEV_GEOCOORDINATESYSTEM_H

#include "LatLonUtmTransformations.h"
#include "LatLonAltStruct.h"
#include "XYZCoord_t.h"
#include <vector>
#include <memory>
using namespace MathLibsNS;

namespace GeoCoordinateSystemNS
{

    enum GeoCoordinateSystemConversionType_e
    {
        /// <summary>
        /// For small areas < +/2 2/5 km
        /// </summary>
                Linear,

        /// <summary>
        /// Preferred for Larger Areas.
        /// Based upon WGS84 Conversions relative to a Reference location.
        /// X-Y zero (0,0) is at the provide Lat/Lon Reference location
        /// Does not have issues with Map Boundarys or crossing the equator
        /// </summary>
                WGS84_Relative,

        /// <summary>
        /// Provides X-Y Coordinates that are established by the
        /// WGS-84 Mapping standards.
        /// Warning!!! There are hugh step changes at map boundaries
        /// and at the equator.  Do not used this conversion if there
        /// are any chances of crossing from one WGS-84 map boundary to another.
        /// For this reason... I highly recommend using the WGS84_Relative option.
        /// </summary>
                WGS84_Map,
    };


    //This is a Singleton Class for handling the Conversion between Lat/Lon and X-Y
    //Coordinates.  The system should be setup at the start of a mission.   It is
    //not valid to change the GeoCoordinateSystem in the middle of a mission
    class GeoCoordinateSystem
    {

    private:

        static GeoCoordinateSystem *_GeoCoordinateSystemPtr;

        LatLonAltCoord_t _referenceLocation;
        int _UTMZoneNumberAtRefLatLon = 0;
        char _UTMZoneLatDesAtRefLatLon = 'N';
        GeoCoordinateSystemConversionType_e _GeoCoordinateSystemConversionType = GeoCoordinateSystemConversionType_e::Linear;

        //Linear Scale Factors for Conversion between Lat/Lon
        //and X-Y coordinates or X-Y to Lat/Lon
        //Note: inverses of the scale factors are created
        //so that division can be avoided during normal conversions.
        //Division is always more time consuming that multiplication.
        //The defaults below work for Colorado.
        double _xToLonCF = 2.007e-7;
        double _yToLatCF = 1.573e-7;
        double _lonToXCF = 4.982e6;
        double _latToYCF = 6.358e6;

        XYZCoord_t _UtmWgs84XYCenter;

        bool _IsCoordinateSystemValid = false;

    public:

        const double EarthRadiusAveMeters = 6371000.0;   //Average
        const double EarthRadiusWGS84Meters = 6378137.0;   //WGS84

        GeoCoordinateSystemConversionType_e GetConversionType()
        {
            return _GeoCoordinateSystemConversionType;
        }

        double GetLongitudeRadToXConversionFactor()
        {
            return _lonToXCF;
        }

        double GetLatitudeRadToYConversionFactor()
        {
            return _latToYCF;
        }


        /// <summary>
        /// The reference location is the Lat/Lon location
        /// where the X-Y coordintate will be (0, 0)
        /// The Altitude is typically set to the nominial ground
        /// altitude in meters above/below mean sea level.
        /// </summary>
        LatLonAltCoord_t ReferenceLatLonAltLocation()
        {
            return _referenceLocation;
        }

        /// <summary>
        /// The is the UTM Zone Number at the Reference
        /// Lat/Lon Location
        /// </summary>
        int UTMZoneNumberAtRefLatLon()
        {
                return _UTMZoneNumberAtRefLatLon;
        }

        /// <summary>
        /// This is the UTM Zone Latitude Reference Designator
        /// at the Reference Lat/Lon Location
        /// </summary>
        char UTMZoneLatDesAtRefLatLon()
        {
            return _UTMZoneLatDesAtRefLatLon;
        }

        /// <summary>
        ///The UTM X-Y Location of the ReferenceLatLonLocation
        ///For this system the X (or East-West) value will normally be zero.
        //The Y value will be the number of meter from the equator.
        /// </summary>
        XYZCoord_t UtmWgs84XYCenter()
        {
            return _UtmWgs84XYCenter;
        }

        bool IsCoordinateSystemValid()
        {
            return _IsCoordinateSystemValid;
        }


    private:
        //A private Constructor;
        GeoCoordinateSystem()
        {
            _IsCoordinateSystemValid = false;
            _GeoCoordinateSystemConversionType = GeoCoordinateSystemConversionType_e::Linear;
        }

    public:

        //Get a Pointer to the GeoCoordinateSystem
        static  GeoCoordinateSystem *GetGeoCoordinateSystemReference()
        {
            if (!_GeoCoordinateSystemPtr)
            {
                _GeoCoordinateSystemPtr = new GeoCoordinateSystem();
            }
            return _GeoCoordinateSystemPtr;
        }

        /// <summary>
        /// Setup a GeoCoordinateSystem
        /// The reference Latitude and Longitude is the location where
        /// the X-Y coordinate will be zero (0,0).
        /// The Alititude is optional.  If provided it establishs the
        /// nominal ground level at the reference point.
        /// </summary>
        /// <param name="refLatitude">The Reference Latitude</param>
        /// <param name="refLongitude">The Reference Longitude</param>
        /// <param name="nominalGroundAltitudeMSL">The Nominal Ground Level Altitude in
        /// meters above mean sea level, or height above the standard elipsoid</param>
        /// <param name="inDegrees"></param>
        /// <param name="conversionType">if true linear conversion will be setup
        /// and used by default for lat/lon - x/y conversions</param>
        bool SetupGeoCoordinateSystem(double refLatitude, double refLongitude,
                               double nominalGroundAltitudeMSL = 0.0,
                               bool inDegrees = false,
                               GeoCoordinateSystemConversionType_e conversionType = GeoCoordinateSystemConversionType_e::Linear)
        {
            LatLonAltCoord_t latLonAlt(refLatitude, refLongitude, nominalGroundAltitudeMSL, inDegrees);
            return SetupGeoCoordinateSystem(latLonAlt, conversionType);
        }

        /// <summary>
        /// Setup a GeoCoordinateSystem
        /// The reference Latitude and Longitude is the location where
        /// the X-Y coordinate will be zero (0,0).
        /// The Alititude is optional.  If provided it establishs the
        /// nominal ground level at the reference point.
        /// </summary>
        /// <param name="latLonAlt"></param>
        /// <param name="conversionType"></param>
        bool SetupGeoCoordinateSystem(LatLonAltCoord_t &latLonAlt,
                                      GeoCoordinateSystemConversionType_e conversionType = GeoCoordinateSystemConversionType_e::Linear);

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
        bool SetupGeoCoordinateSystem(std::vector<LatLonAltCoord_t> &latLonAltList,
                                      GeoCoordinateSystemConversionType_e conversionType = GeoCoordinateSystemConversionType_e::Linear);


        /// <summary>
        /// Setup the GeoCoordinateSystem
        /// This is used when the Linear Conversion Factors are known.
        /// </summary>
        /// <param name="latLonList"></param>
        /// <param name="LatitudeRadToY"></param>
        /// <param name="LongitudeRadToX"></param>
        bool SetupLinearGeoCoordinateSystemFromConvFactors(LatLonAltCoord_t &refLatLonAltList,
                                      double LatitudeRadToY, double LongitudeRadToX);

        /// <summary>
        /// Convert Latitude/Longitude to the X-Y Coordinate system.
        /// The Center of the X-Y Coordinate system is the Reference (Lat/Lon) Location
        /// which is were X-Y is (0,0)
        /// </summary>
        /// <param name="latLon"></param>
        /// <param name="forceUTMTransformation">Optional... normally false.</param>
        /// <returns></returns>
        XYZCoord_t LatLonAltToXYZ(const LatLonAltCoord_t &latLonAlt);

        /// <summary>
        /// Conver an X-Y coordinate to the Latitude/Longitude Location.
        /// The Center of the X-Y Coordinate system is the Reference (Lat/Lon) Location
        /// which is were X-Y is (0,0)
        /// </summary>
        /// <param name="xyCoord"></param>
        /// <param name="forceUTMTransformation"></param>
        /// <returns></returns>
        LatLonAltCoord_t XYZToLatLonAlt(const XYZCoord_t &xyzCoord);

        /// <summary>
        /// Get the Distance in meters between two Lat/Lon Altitude points.  The formula converts
        /// to the xyz cooridiante equivalents and then uses a Euclidian distance between them.
        /// </summary>
        /// <param name="pt1"></param>
        /// <param name="pt2"></param>
        /// <returns></returns>
        double DistanceBetweenLatLonAltPointsEuclidian(LatLonAltCoord_t &pt1, LatLonAltCoord_t &pt2)
        {
            XYZCoord_t xyPt1;
            XYZCoord_t xyPt2;
            xyPt1 = LatLonAltToXYZ(pt1);
            xyPt2 = LatLonAltToXYZ(pt2);
            return xyPt1.Distance(xyPt2);
        }

        /// <summary>
        /// Get the Distance in meters between two Lat/Lon points.  The distance is based
        /// upon the great circles acrosss the survace of the earth using the Haversine formula
        /// </summary>
        /// <param name="pt1"></param>
        /// <param name="pt2"></param>
        /// <returns></returns>
        double DistanceBetweenLatLonPointsGreatArc(LatLonAltCoord_t &pt1, LatLonAltCoord_t &pt2)
        {
            return HaversineDistanceBetweenLatLonPointsRad(pt1, pt2);
        }

        //Haversine
        //formula:a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
        //c = 2 ⋅ atan2( √a, √(1−a) )
        //d = R ⋅ c
        //φ is latitude, λ is longitude, R is earth’s radius (mean radius = 6,371km);
        double HaversineDistanceBetweenLatLonPointsRad(LatLonAltCoord_t &pt1, LatLonAltCoord_t &pt2);


    };

}

#endif //VIDERE_DEV_GEOCOORDINATESYSTEM_H
