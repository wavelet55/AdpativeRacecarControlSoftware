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


#include "LatLonUtmTransformations.h"
#include <cmath>
#include <boost/math/constants/constants.hpp>

using namespace std;

namespace GeoCoordinateSystemNS
{
    const double LatLonUtmWgs84Conv::PI = boost::math::constants::pi<double>();
    const double LatLonUtmWgs84Conv::HALFPI = 0.5 * boost::math::constants::pi<double>();
    const double LatLonUtmWgs84Conv::RTOD = 180.0 / boost::math::constants::pi<double>();
    const double LatLonUtmWgs84Conv::DTOR = boost::math::constants::pi<double>() / 180.0;

    const double LatLonUtmWgs84Conv::EQUATORIALRADIUS = 6378137.0;
    const double LatLonUtmWgs84Conv::ECCENTRICITYSQUARED = 0.00669438;
    const double LatLonUtmWgs84Conv::MAXWGS84VAL = 10000000;

    char LatLonUtmWgs84Conv::UtmLetterDesignator(double Lat)
    {

        char LetterDesignator;

        if ((84 >= Lat) && (Lat >= 72)) LetterDesignator = 'X';
        else if ((72 > Lat) && (Lat >= 64)) LetterDesignator = 'W';
        else if ((64 > Lat) && (Lat >= 56)) LetterDesignator = 'V';
        else if ((56 > Lat) && (Lat >= 48)) LetterDesignator = 'U';
        else if ((48 > Lat) && (Lat >= 40)) LetterDesignator = 'T';
        else if ((40 > Lat) && (Lat >= 32)) LetterDesignator = 'S';
        else if ((32 > Lat) && (Lat >= 24)) LetterDesignator = 'R';
        else if ((24 > Lat) && (Lat >= 16)) LetterDesignator = 'Q';
        else if ((16 > Lat) && (Lat >= 8)) LetterDesignator = 'P';
        else if ((8 > Lat) && (Lat >= 0)) LetterDesignator = 'N';
        else if ((0 > Lat) && (Lat >= -8)) LetterDesignator = 'M';
        else if ((-8 > Lat) && (Lat >= -16)) LetterDesignator = 'L';
        else if ((-16 > Lat) && (Lat >= -24)) LetterDesignator = 'K';
        else if ((-24 > Lat) && (Lat >= -32)) LetterDesignator = 'J';
        else if ((-32 > Lat) && (Lat >= -40)) LetterDesignator = 'H';
        else if ((-40 > Lat) && (Lat >= -48)) LetterDesignator = 'G';
        else if ((-48 > Lat) && (Lat >= -56)) LetterDesignator = 'F';
        else if ((-56 > Lat) && (Lat >= -64)) LetterDesignator = 'E';
        else if ((-64 > Lat) && (Lat >= -72)) LetterDesignator = 'D';
        else if ((-72 > Lat) && (Lat >= -80)) LetterDesignator = 'C';
        else LetterDesignator = 'Z'; //This is here as an error flag to show that the Latitude is outside the UTM limits

        return LetterDesignator;
    }

    int LatLonUtmWgs84Conv::UtmZoneNumber(double latDeg, double longDeg)
    {
        int ZoneNumber;
        //Make sure the longitude is between -180.00 .. 179.9
        double LongTemp = (longDeg + 180) - (int) ((longDeg + 180) / 360) * 360 - 180; // -180.00 .. 179.9;
        ZoneNumber = (int) ((LongTemp + 180) / 6) + 1;

        if (latDeg >= 56.0 && latDeg < 64.0 && LongTemp >= 3.0 && LongTemp < 12.0)
            ZoneNumber = 32;

        // Special zones for Svalbard
        if (latDeg >= 72.0 && latDeg < 84.0)
        {
            if (LongTemp >= 0.0 && LongTemp < 9.0) ZoneNumber = 31;
            else if (LongTemp >= 9.0 && LongTemp < 21.0) ZoneNumber = 33;
            else if (LongTemp >= 21.0 && LongTemp < 33.0) ZoneNumber = 35;
            else if (LongTemp >= 33.0 && LongTemp < 42.0) ZoneNumber = 37;
        }

        return ZoneNumber;
    }

    UTMCoord_t LatLonUtmWgs84Conv::LLtoUTM_Degrees(double latDegrees,
                                                   double longDegrees, double altHAE,
                                                   bool setUTMRelativeToCenterLongitude,
                                                   double centerLongDegrees)
    {

        UTMCoord_t utm = UTMCoord_t();
        utm.Clear();
        utm.AltitudeHEA = altHAE;
        double a = EQUATORIALRADIUS;
        double eccSquared = ECCENTRICITYSQUARED;
        double k0 = 0.9996;

        double LongOrigin;
        double eccPrimeSquared;
        double N, T, C, A, M;
        double eastingOffset = setUTMRelativeToCenterLongitude ? 0.0 : 500000.0;
        double northingOffset = setUTMRelativeToCenterLongitude ? 0.0 : 10000000.0;

        utm.UTMRelativeToCenterLongitude = setUTMRelativeToCenterLongitude;

        //Make sure the longitude is between -180.00 .. 179.9
        double LongTemp = (longDegrees + 180) - (int) ((longDegrees + 180) / 360) * 360 - 180; // -180.00 .. 179.9;

        double LatRad = latDegrees * DTOR;
        double LongRad = LongTemp * DTOR;
        double LongOriginRad;
        int ZoneNumber = UtmZoneNumber(latDegrees, longDegrees);

        if (setUTMRelativeToCenterLongitude)
        {
            LongOrigin = (centerLongDegrees + 180) - (int) ((centerLongDegrees + 180) / 360) * 360 - 180;
        } else
        {
            LongOrigin = (ZoneNumber - 1) * 6 - 180 + 3; //+3 puts origin in middle of zone
        }
        utm.CenterLongitudeDegrees = LongOrigin;
        LongOriginRad = LongOrigin * DTOR;

        //compute the UTM Zone from the latitude and longitude
        utm.UTMZoneNumber = ZoneNumber;
        utm.UTMZoneLatDes = UtmLetterDesignator(latDegrees);

        eccPrimeSquared = (eccSquared) / (1 - eccSquared);

        N = a / sqrt(1 - eccSquared * sqrt(LatRad) * sin(LatRad));
        T = tan(LatRad) * tan(LatRad);
        C = eccPrimeSquared * cos(LatRad) * cos(LatRad);
        A = cos(LatRad) * (LongRad - LongOriginRad);

        M = a *
            ((1 - eccSquared / 4 - 3 * eccSquared * eccSquared / 64 - 5 * eccSquared * eccSquared * eccSquared / 256) *
             LatRad
             - (3 * eccSquared / 8 + 3 * eccSquared * eccSquared / 32 +
                45 * eccSquared * eccSquared * eccSquared / 1024) * sin(2 * LatRad)
             + (15 * eccSquared * eccSquared / 256 + 45 * eccSquared * eccSquared * eccSquared / 1024) * sin(4 * LatRad)
             - (35 * eccSquared * eccSquared * eccSquared / 3072) * sin(6 * LatRad));

        utm.UTMEasting = (double) (k0 * N * (A + (1 - T + C) * A * A * A / 6
                                              +
                                              (5 - 18 * T + T * T + 72 * C - 58 * eccPrimeSquared) * A * A * A * A * A /
                                              120)
                                    + eastingOffset);

        utm.UTMNorthing = (double) (k0 * (M + N * tan(LatRad) *
                                               (A * A / 2 + (5 - T + 9 * C + 4 * C * C) * A * A * A * A / 24
                                                + (61 - 58 * T + T * T + 600 * C - 330 * eccPrimeSquared) * A * A * A *
                                                  A * A * A / 720)));

        if (latDegrees < 0 && !setUTMRelativeToCenterLongitude)
            utm.UTMNorthing += northingOffset; //10000000 meter offset for southern hemisphere

        return utm;
    }

    LatLonAltCoord_t LatLonUtmWgs84Conv::UTMtoLL_Degrees(const UTMCoord_t &utm)
    {

        LatLonAltCoord_t latLon = LatLonAltCoord_t();
        latLon.SetAltitude(utm.AltitudeHEA);
        double k0 = 0.9996;
        double a = EQUATORIALRADIUS;
        double eccSquared = ECCENTRICITYSQUARED;
        double eccPrimeSquared;
        double e1 = (1 - sqrt(1 - eccSquared)) / (1 + sqrt(1 - eccSquared));
        double N1, T1, C1, R1, D, M;
        double LongOrigin;
        double mu, phi1, phi1Rad;
        double x, y;
        int ZoneNumber;

        x = utm.UTMEasting;
        y = utm.UTMNorthing;
        LongOrigin = utm.CenterLongitudeDegrees;
        ZoneNumber = utm.UTMZoneNumber;
        if (!utm.UTMRelativeToCenterLongitude)
        {
            x = utm.UTMEasting - 500000.0; //remove 500,000 meter offset for longitude}
            if (utm.UTMZoneLatDes < 'N')
            {
                y -= 10000000.0; //remove 10,000,000 meter offset used for southern hemisphere
            }

            LongOrigin = (ZoneNumber - 1) * 6 - 180 + 3; //+3 puts origin in middle of zone
        }

        eccPrimeSquared = (eccSquared) / (1 - eccSquared);

        M = y / k0;
        mu = M / (a * (1 - eccSquared / 4 - 3 * eccSquared * eccSquared / 64 -
                       5 * eccSquared * eccSquared * eccSquared / 256));

        phi1Rad = mu + (3 * e1 / 2 - 27 * e1 * e1 * e1 / 32) * sin(2 * mu)
                  + (21 * e1 * e1 / 16 - 55 * e1 * e1 * e1 * e1 / 32) * sin(4 * mu)
                  + (151 * e1 * e1 * e1 / 96) * sin(6 * mu);
        phi1 = phi1Rad * RTOD;

        N1 = a / sqrt(1 - eccSquared * sin(phi1Rad) * sin(phi1Rad));
        T1 = tan(phi1Rad) * tan(phi1Rad);
        C1 = eccPrimeSquared * cos(phi1Rad) * cos(phi1Rad);
        R1 = a * (1 - eccSquared) / pow(1 - eccSquared * sin(phi1Rad) * sin(phi1Rad), 1.5);
        D = x / (N1 * k0);

        latLon.SetLatitudeRadians(phi1Rad - (N1 * tan(phi1Rad) / R1) * (D * D / 2 -
                                                                        (5 + 3 * T1 + 10 * C1 - 4 * C1 * C1 -
                                                                         9 * eccPrimeSquared) * D * D * D * D / 24
                                                                        + (61 + 90 * T1 + 298 * C1 + 45 * T1 * T1 -
                                                                           252 * eccPrimeSquared - 3 * C1 * C1) * D *
                                                                          D * D * D * D * D / 720));

        double longRad = (D - (1 + 2 * T1 + C1) * D * D * D / 6 +
                                    (5 - 2 * C1 + 28 * T1 - 3 * C1 * C1 + 8 * eccPrimeSquared + 24 * T1 * T1)
                                    * D * D * D * D * D / 120) / cos(phi1Rad);

        longRad += (PI / 180.0) * LongOrigin;
        latLon.SetLongitudeRadians(longRad);
        return latLon;
    }
}