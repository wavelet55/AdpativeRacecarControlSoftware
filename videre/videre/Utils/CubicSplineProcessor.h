/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: July 2018
 *
  *******************************************************************/

#ifndef VIDERE_DEV_CUBICSPLINEPROCESSOR_H
#define VIDERE_DEV_CUBICSPLINEPROCESSOR_H

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <boost/filesystem.hpp>
#include "global_defines.h"
#include "logger.h"

namespace VidereUtils
{

    struct CubicSplineCoefs_t
    {
        double Xoffset;
        double C0;
        double C1;
        double C2;
        double C3;

        void Clear()
        {
            Xoffset;
            C0 = 0;
            C1 = 0;
            C2 = 0;
            C3 = 0;
        }
    };

    class CubicSplineProcessor
    {
    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        std::vector<CubicSplineCoefs_t> csCoefs;

    public:
        CubicSplineProcessor();

        double computeFx(double x);

        double computeFx1stDir(double x);

        double computeFx2ndDir(double x);

        double computeFx3rdDir(double x);


        //Read the Cubic Spline Coeficients from the given file.
        //Return false if ok, true if error
        bool readCoefsFromFile(std::string filename);

    private:
        bool parseLineToCSCoefVals(std::string line, CubicSplineCoefs_t &csVals);
    };


}
#endif //VIDERE_DEV_CUBICSPLINEPROCESSOR_H
