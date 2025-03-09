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

#ifndef LAT_LON_ALTITUDE_STRUCTURE
#define LAT_LON_ALTITUDE_STRUCTURE

#include <string>
#include <sstream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <boost/math/constants/constants.hpp>

namespace GeoCoordinateSystemNS
{

    /**
     * A Coordinate System for Latitude, Longitude and Altitude.
     * Latitude, Longitude are well defined.
     * The Altitude is typically in meters above Mean-Sea-Level
     * or above the standard elipsoid model... but it could have
     * different reference points based upon use.
     */
    struct LatLonAltCoord_t
    {

        LatLonAltCoord_t()
        {
            lat_rad_ = 0.0;
            lon_rad_ = 0.0;
            altitude_meter_ = 0.0;
        }

        LatLonAltCoord_t(double lat, double lon, double altitude_meters = 0, bool inDegrees = false)
        {
            if(inDegrees)
            {
                SetLatitudeDegrees(lat);
                SetLongitudeDegrees(lon);
            }
            else
            {
                SetLatitudeRadians(lat);
                SetLongitudeRadians(lon);
            }
            SetAltitude(altitude_meters);
        }


        void Clear()
        {
            lat_rad_ = 0.0;
            lon_rad_ = 0.0;
            altitude_meter_ = 0.0;
        }


        /**
         * Element-wise addition.
         */
        LatLonAltCoord_t operator+(const LatLonAltCoord_t &lla) const
        {
            LatLonAltCoord_t r;
            r.SetLatitudeRadians(LatitudeRadians() + lla.LatitudeRadians());
            r.SetLongitudeRadians(LongitudeRadians() + lla.LongitudeRadians());
            r.SetAltitude(Altitude() + lla.Altitude());
            return r;
        }

        /**
         * Element-wise subtraction.
         */
        LatLonAltCoord_t operator-(const LatLonAltCoord_t &lla) const;

        /**
         * Scale Lat and Lon by the scalar factor "c".
         * The altitude is not scaled as it does not normally make sense to
         * scale the altitude by the same factor as lat and lon.
         * @param c is the constant to scale Lat and Lon by.
         */
        LatLonAltCoord_t operator*(const double c) const
        {
            LatLonAltCoord_t r;
            r.SetLatitudeRadians(LatitudeRadians() * c);
            r.SetLongitudeRadians(LongitudeRadians() * c);
            r.SetAltitude(Altitude());
            return r;
        }

        /**
         * Divide Lat and Lon by the scalar factor "c".
         * The altitude is not scaled as it does not normally make sense to
         * scale the altitude by the same factor as lat and lon.
         * @param c is the constant to scale Lat and Lon by.
         */
        LatLonAltCoord_t operator/(const double c) const;


        /// <summary>
        /// Find the Maximum North-East Corner of a set of Latitude-Longitude values.
        /// It is assumed  Latitude values are in the range: [-Pi/2, Pi/2] or [-90.0, 90.0]
        /// and Longitude values are in the range: [-Pi, Pi) or [-180.0, 180.0)
        /// </summary>
        /// <param name="latLonList"></param>
        /// <returns></returns>
        static LatLonAltCoord_t FindMaxNortEastCornerOfSetOfLatLonPoints(std::vector<LatLonAltCoord_t> &latLonList);

        /// <summary>
        /// Find the Minimum South-West Corner of a set of Latitude-Longitude values.
        /// It is assumed  Latitude values are in the range: [-Pi/2, Pi/2] or [-90.0, 90.0]
        /// and Longitude values are in the range: [-Pi, Pi) or [-180.0, 180.0)
        /// </summary>
        /// <param name="latLonList"></param>
        /// <returns></returns>
        static LatLonAltCoord_t FindMinSouthWestCornerOfSetOfLatLonPoints(std::vector<LatLonAltCoord_t> &latLonList);


        /// <summary>
        /// Find the Center of a set of Latitude-Longitude values.
        /// It is assumed  Latitude values are in the range: [-Pi/2, Pi/2] or [-90.0, 90.0]
        /// and Longitude values are in the range: [-Pi, Pi) or [-180.0, 180.0)
        /// The method handles cases where there is a cross over at -180 degress Longitude
        /// </summary>
        /// <param name="latLonList"></param>
        /// <returns></returns>
        static LatLonAltCoord_t FindCenterOfSetOfLatLonPoints(std::vector<LatLonAltCoord_t> &latLonList);


        std::string ToString() const;

        std::string ToCvsString() const;



        /* Accessors */

        /**
         * Latitude in Radians [-PI/2, PI/2]
         */
        inline double LatitudeRadians() const
        {
            return lat_rad_;
        }

        /**
         * Set Latitude in Radians [-PI/2, PI/2]
         */
        inline void SetLatitudeRadians(double value)
        {
            lat_rad_ = value < -HALFPI ? -HALFPI : value > HALFPI ? HALFPI : value;
        }

        /**
         * Latitude in Degrees [-90.0, 90.0]
         */
        double LatitudeDegrees() const
        {
            return lat_rad_ * RTOD;
        }

        /**
         * Set_Latitude in Degrees [-90.0, 90.0]
         */
        void SetLatitudeDegrees(double value)
        {
            SetLatitudeRadians(value * DTOR);
        }

        /**
         * Longitude in Radians [-PI, PI]
         */
        inline double LongitudeRadians() const
        {
            return lon_rad_;
        }

        /**
         * Set Longitude in Radians [-PI, PI]
         */
        void SetLongitudeRadians(double value)
        {
            if (value <= -PI || value > PI)
            {
                value = fmod(value, 2 * PI);
                if (value > PI)
                    value -= 2 * PI;
                else if (value <= -PI)
                    value += 2 * PI;
            }
            lon_rad_ = value;
        }

        /**
         * Longitude in Degrees [-180, 180]
         */
        inline double LongitudeDegrees() const
        {
            return lon_rad_ * RTOD;
        }

        /**
         * Set Longitude in Degrees [-180, 180]
         */
        void SetLongitudeDegrees(double value)
        {
            if (value <= -180.0 || value > 180.0)
            {
                value = fmod(value, 360.0);
                if (value > 180.0)
                    value -= 360.0;
                else if (value <= -180.0)
                    value += 360.0;
            }
            lon_rad_ = value * DTOR;
        }

        /**
         * Altitude in Meters.
         * The Altitude is typically in meters above Mean-Sea-Level 
         * or above the standard elipsoid model...
         * but it coud have different reference points based upon use.
         */
        inline double Altitude() const
        {
            return altitude_meter_;
        }

        inline void SetAltitude(double value)
        {
            altitude_meter_ = value;
        }



        static const double STRPRECISION ;
        static const double STRPRECISIONCSV;
        //const double PI = 3.1415926535897932;
        static const double PI;
        static const double HALFPI;
        static const double RTOD;
        static const double DTOR;

        //Used to test equality between Lat/Lon
        static const double EqualEpslon;

    private:
        double lat_rad_; /* Latitude in Radians */
        double lon_rad_; /* Longitude in Radians */
        double altitude_meter_; /* Altitude in Meters, typically above Mean-Sea-Level */

    };


}


#endif // LAT_LON_ALTITUDE_STRUCTURE