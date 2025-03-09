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

#ifndef LAT_LON_UMT_TRANSFORMATIONS
#define LAT_LON_UMT_TRANSFORMATIONS

#include "LatLonAltStruct.h"

//using namespace MathLibsNS;

namespace GeoCoordinateSystemNS
{

    /** This is the UTM X-Y Coordinate
     * The Standard UTM Coordinates are based upon center Longitude Zones
     * that are 6 degrees appart going around the world and positive Northing
     * starting at zero going north from the equator... 
     * If UTMRelativeToCenterLongitude is true, a non-standard Center 
     * Longitude reference is used.
     */
    struct UTMCoord_t
    {
        /** If true, the umt_easting is relative to the center_longitude_degrees and 
         *umt_northing is positive in the northern hemisphere and negative in the southern
         *hemisphere 
         */
        bool UTMRelativeToCenterLongitude;

        double UTMNorthing;
        double UTMEasting;
        int UTMZoneNumber;
        char UTMZoneLatDes;

        /** Height Above Earth or the reference of ellipsoid this falue is normally zero and
         * is not currently used
         */
        double AltitudeHEA;

        /** umt_relative_to_center_longitude = true... this is set by the user to the desired
         * center longitude. If umt_relative_to_center_longitude = false, this is the center
         * longitude of the utm_zone_number
         */
        double CenterLongitudeDegrees;

        void Clear()
        {
            UTMRelativeToCenterLongitude = false;
            UTMNorthing = 0.0;
            UTMEasting = 0.0;
            UTMZoneNumber = 0;
            UTMZoneLatDes = 'N';
            AltitudeHEA = 0.0;
        }
    };

    /** This is a static class that converts Lat/Lon to UTM - WGS84
     * coordinates and UTM coordinates to Lat/Lon.  
     * The WGS84 system is assumed.  Other systems can be used,
     * simply change the EquatorialRadius and the eccentricitySquared 
     * constants.
     * The Standard UTM Coordinates are based upon center Longitude Zones
     * that are 6 degrees apart going around the world and positive.
     * Because of this, there are large discontinuities at the boundary
     * of each UTMZoneNumber.
     * The Northing coordinate starts at zero at the equator going North,
     * but in the southern hemisphere the UTMNorthing is 0 at the south-pole
     * and 10,000,000.00 at the equator.
     * ... which gives a large discontinuity at the equator.
     * 
     * This class will work with the Standard UTM system, but is also 
     * designed to work relative to a provided Center Longitude.  For the
     * relative Center Longitude, the center longitude is provided by the user
     * and Easting is positive for latitudes positive relative to the center long
     * and negative if (long - center longitude is negative.  Also, the Northing
     * will be positive going north from the equator and negative going south from
     * the equator.  This avoids discontinuitys at boundaries.  
     * Note:  for AbsValues of (Longitude - CenterLongitude) > 3 degrees, the error
     * may be larger than the standard UTM.  This should not be an issue for any 
     * reasonable UAV mission.
     */
    class LatLonUtmWgs84Conv
    {
    public:

        /**This routine determines the correct UTM letter designator for the given latitude
         * returns 'Z' if latitude is outside the UTM limits of 84N to 80S
         * Written by Chuck Gantz- chuck.gantz@globalstar.com
         * @param Lat 
         */
        static char UtmLetterDesignator(double Lat);

        /**Compute the UTM Zone number base on the Longitude.
         * @param latDeg is the latitude in Degrees
         * @param longDeg is the longitude in Degrees
         * @return the zone number.
         */
        static int UtmZoneNumber(double latDeg, double longDeg);

        /** Converts lat/long to UTM coords.  Equations from USGS Bulletin 1532 
         * East Longitudes are positive, West longitudes are negative. 
         * North latitudes are positive, South latitudes are negative
         * Lat and Long are in radians
         * Written by Chuck Gantz- chuck.gantz@globalstar.com
         * @param latRadians is the latitude in radians.
         * @param longRadians is the longitude in radians.
         * @return the UTMCoord_t
         */
        static UTMCoord_t LLtoUTM_Radians(double latRadians, double longRadians, double altHAE = 0.0,
                                          bool setUTMRelativeToCenterLongitude = false, double centerLongDegrees = 0.0)
        {
            return LLtoUTM_Degrees(latRadians * RTOD, longRadians * RTOD, altHAE, setUTMRelativeToCenterLongitude,
                                   centerLongDegrees);
        }


        /** Converts lat/long to UTM coords.  Equations from USGS Bulletin 1532 
         * East Longitudes are positive, West longitudes are negative. 
         * North latitudes are positive, South latitudes are negative
         * Lat and Long are in decimal degrees
         * Written by Chuck Gantz- chuck.gantz@globalstar.com
         * @param latDegrees is the latitude in degrees.
         * @param longDegrees is the longitude in degrees.
         * @return UTMCoord.
         */
        static UTMCoord_t LLtoUTM_Degrees(double latDegrees, double longDegrees, double altHAE = 0.0,
                                          bool setUTMRelativeToCenterLongitude = false, double centerLongDegrees = 0.0);


        /**converts UTM coords to lat/long.  Equations from USGS Bulletin 1532 
         * East Longitudes are positive, West longitudes are negative. 
         * North latitudes are positive, South latitudes are negative
         * Lat and Long are in decimal degrees. 
         * Written by Chuck Gantz- chuck.gantz@globalstar.com
         * @param utm is the position in UTM coordinates
         * @return lat lon alt coordinate.
         * */
        static LatLonAltCoord_t UTMtoLL_Degrees(const UTMCoord_t &utm);

    private:
        static const double PI;
        static const double HALFPI; /*   pi / 2   */
        static const double RTOD; /*  180 / pi  */
        static const double DTOR; /*   pi / 180 */
        static const double EQUATORIALRADIUS; /* WGS84 */
        static const double ECCENTRICITYSQUARED; /* WGS84 */
        static const double MAXWGS84VAL; /* 10 million meters. */
    };
}

#endif 