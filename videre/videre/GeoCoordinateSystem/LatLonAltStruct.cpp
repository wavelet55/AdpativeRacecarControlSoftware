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

#include "LatLonAltStruct.h"
#include <limits>

namespace GeoCoordinateSystemNS
{

    const double LatLonAltCoord_t::STRPRECISION = 6;
    const double LatLonAltCoord_t::STRPRECISIONCSV = 12;
    const double LatLonAltCoord_t::PI = boost::math::constants::pi<double>();
    const double LatLonAltCoord_t::HALFPI = 0.5 * boost::math::constants::pi<double>();
    const double LatLonAltCoord_t::RTOD = 180.0 / boost::math::constants::pi<double>();
    const double LatLonAltCoord_t::DTOR = boost::math::constants::pi<double>() / 180.0;

    //Used to test equality between Lat/Lon
    const double LatLonAltCoord_t::EqualEpslon = 1.0e-15;


    /**
    * Element-wise subtraction.
    */
    LatLonAltCoord_t LatLonAltCoord_t::operator-(const LatLonAltCoord_t &lla) const
    {
        LatLonAltCoord_t r;
        //Near the -180 or +180 degree Longitude point we must be careful so
        //that the difference gives us the distance in degrees between the locations
        //Note:  there are no wrap-around the +90 or -90 degree locations for LatitudeRadians.
        double delLon = lon_rad_  - lla.lon_rad_;
        if (delLon < -PI)
            delLon += 2.0 * PI;
        else if(delLon >= PI)
            delLon -= 2.0 * PI;

        r.SetLongitudeRadians(delLon);
        r.SetLatitudeRadians(LatitudeRadians() - lla.LatitudeRadians());
        r.SetAltitude(Altitude() - lla.Altitude());
        return r;
    }

    /**
     * Divide Lat and Lon by the scalar factor "c".
     * The altitude is not scaled as it does not normally make sense to
     * scale the altitude by the same factor as lat and lon.
     * @param c is the constant to scale Lat and Lon by.
     */
    LatLonAltCoord_t LatLonAltCoord_t::operator/(const double c) const
    {
        LatLonAltCoord_t r;
        //Handle a divide by zero is some reasonable fashion that
        //does not throw exceptions.
        double a = 1.0;
        if( fabs(c) > 0 )
            a = 1.0 / c;
        else if( c >= 0)
            a = 1.0 / EqualEpslon;
        else
            a = -1.0 / EqualEpslon;

        r.SetLatitudeRadians(LatitudeRadians() * a);
        r.SetLongitudeRadians(LongitudeRadians() * a);
        r.SetAltitude(Altitude());
        return r;
    }


    /// <summary>
    /// Find the Maximum North-East Corner of a set of Latitude-Longitude values.
    /// It is assumed  Latitude values are in the range: [-Pi/2, Pi/2] or [-90.0, 90.0]
    /// and Longitude values are in the range: [-Pi, Pi) or [-180.0, 180.0)
    /// Sets the Altitude to the maximum of all the altitudes.
    /// </summary>
    /// <param name="latLonList"></param>
    /// <returns></returns>
    LatLonAltCoord_t LatLonAltCoord_t::FindMaxNortEastCornerOfSetOfLatLonPoints(std::vector<LatLonAltCoord_t> &latLonList)
    {
        LatLonAltCoord_t maxLatLon;
        double maxLatRad = -std::numeric_limits<double>::max();
        double maxLonRad = -std::numeric_limits<double>::max();
        double maxAlt = -std::numeric_limits<double>::max();
        bool posLon = false;
        bool negLon = false;
        double pi = boost::math::constants::pi<double>();

        int N = latLonList.size();
        //GeoCoordinateSystem gcs = null;
        for(int i = 0; i < N; i++)
        {
            //if (latLon.GeoCoordinateSys != null) gcs = latLon.GeoCoordinateSys;
            maxLatRad = fmax(latLonList[i].LatitudeRadians(), maxLatRad);
            maxLonRad = fmax(latLonList[i].LongitudeRadians(), maxLonRad);
            maxAlt = fmax(latLonList[i].Altitude(), maxAlt);
            if (latLonList[i].LongitudeRadians() < 0)
                negLon = true;
            else
                posLon = true;
        }
        if (posLon && negLon && maxLonRad > 0.5 * pi)
        {
            //We have points on both sides of the 180 degree Longitude...
            //We must go back and choose the smaller largest negative Longitude.
            maxLonRad = -std::numeric_limits<double>::max();
            for(int i = 0; i < N; i++)
            {
                if (latLonList[i].LongitudeRadians() < 0)
                    maxLonRad = fmax(latLonList[i].LongitudeRadians(), maxLonRad);
            }
        }

        maxLatLon.SetLatitudeRadians(maxLatRad);
        maxLonRad = maxLonRad >= pi ? pi - 1.0e-12 : maxLonRad;
        maxLatLon.SetLongitudeRadians(maxLonRad);
        maxLatLon.SetAltitude(maxAlt);
        //maxLatLon.GeoCoordinateSys = gcs;
        return maxLatLon;
    }

    /// <summary>
    /// Find the Minimum South-West Corner of a set of Latitude-Longitude values.
    /// It is assumed  Latitude values are in the range: [-Pi/2, Pi/2] or [-90.0, 90.0]
    /// and Longitude values are in the range: [-Pi, Pi) or [-180.0, 180.0)
    /// Sets the Altitude to the Minimum of all the Altitudes.
    /// </summary>
    /// <param name="latLonList"></param>
    /// <returns></returns>
    LatLonAltCoord_t LatLonAltCoord_t::FindMinSouthWestCornerOfSetOfLatLonPoints(std::vector<LatLonAltCoord_t> &latLonList)
    {
        LatLonAltCoord_t minLatLon;
        double minLatRad = std::numeric_limits<double>::max();
        double minLonRad = std::numeric_limits<double>::max();
        double minAlt = std::numeric_limits<double>::max();
        bool posLon = false;
        bool negLon = false;
        double pi = boost::math::constants::pi<double>();
        //GeoCoordinateSystem gcs = null;
        int N = latLonList.size();
        for(int i = 0; i < N; i++)
        {
            //if (latLon.GeoCoordinateSys != null) gcs = latLon.GeoCoordinateSys;
            minLatRad = fmin(latLonList[i].LatitudeRadians(), minLatRad);
            minLonRad = fmin(latLonList[i].LongitudeRadians(), minLonRad);
            minAlt = fmin(latLonList[i].Altitude(), minAlt);
            if (latLonList[i].LongitudeRadians() < 0)
                negLon = true;
            else
                posLon = true;
        }
        if (posLon && negLon && minLonRad < -0.5 * pi)
        {
            //We have points on both sides of the 180 degree Longitude...
            //We must go back and choose the smalled positive Longitude.
            minLonRad = std::numeric_limits<double>::max();
            for(int i = 0; i < N; i++)
            {
                if (latLonList[i].LongitudeRadians() > 0)
                    minLonRad = fmin(latLonList[i].LongitudeRadians(), minLonRad);
            }
        }

        minLatLon.SetLatitudeRadians( minLatRad);
        minLonRad =  minLonRad < -pi ? -pi : minLonRad;
        minLatLon.SetLongitudeRadians(minLonRad);
        minLatLon.SetAltitude(minAlt);
        //minLatLon.GeoCoordinateSys = gcs;
        return minLatLon;
    }


    /// <summary>
    /// Find the Center of a set of Latitude-Longitude values.
    /// It is assumed  Latitude values are in the range: [-Pi/2, Pi/2] or [-90.0, 90.0]
    /// and Longitude values are in the range: [-Pi, Pi) or [-180.0, 180.0)
    /// The method handles cases where there is a cross over at -180 degress Longitude
    /// </summary>
    /// <param name="latLonList"></param>
    /// <returns></returns>
    LatLonAltCoord_t LatLonAltCoord_t::FindCenterOfSetOfLatLonPoints(std::vector<LatLonAltCoord_t> &latLonList)
    {
        LatLonAltCoord_t maxLatLon( FindMaxNortEastCornerOfSetOfLatLonPoints(latLonList) );
        LatLonAltCoord_t minLatLon( FindMinSouthWestCornerOfSetOfLatLonPoints(latLonList) );
        LatLonAltCoord_t delLatLon = (maxLatLon - minLatLon) * 0.5;
        LatLonAltCoord_t centerLatLon = minLatLon + delLatLon;
        return centerLatLon;
    }



    std::string LatLonAltCoord_t::ToString() const
    {
        std::ostringstream buff;
        buff << "Latitude(Deg)=";
        buff << std::setprecision(STRPRECISION) << lat_rad_ * RTOD;
        buff << ", Longitude(Deg)=";
        buff << std::setprecision(STRPRECISION) << lon_rad_ * RTOD;
        buff << ", Altitude(Meters)=";
        buff << std::scientific << std::setprecision(STRPRECISION) << altitude_meter_;

        return buff.str();
    }

    std::string LatLonAltCoord_t::ToCvsString() const
    {
        std::ostringstream buff;
        buff << std::scientific << std::setprecision(STRPRECISIONCSV) << lat_rad_ * RTOD;
        buff << ", ";
        buff << std::scientific << std::setprecision(STRPRECISIONCSV) << lon_rad_ * RTOD;
        buff << ", ";
        buff << std::scientific << std::setprecision(STRPRECISIONCSV) << altitude_meter_;

        return buff.str();
    }

}

