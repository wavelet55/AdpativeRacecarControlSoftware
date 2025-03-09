/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: June 2018
 *
  *******************************************************************/


#include "CubicSplineProcessor.h"
#include "FileUtils.h"
#include <iostream>
#include <fstream>

using namespace std;

namespace VidereUtils
{

    CubicSplineProcessor::CubicSplineProcessor()
    {
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);
    }


    bool CubicSplineProcessor::parseLineToCSCoefVals(string line, CubicSplineCoefs_t &csVals)
    {
        bool error = false;
        string valStr;
        double csdVals[5];
        try
        {
            stringstream ss( line );
            for(int i = 0; i < 5; i++)
            {
                if(getline(ss, valStr, ','))
                {
                    if(valStr.length() > 0)
                    {
                        stringstream fs( valStr );
                        fs >> csdVals[i];
                    }
                    else
                    {
                        error = true;
                        break;
                    }
                }
            }
            if( !error)
            {
                csVals.Xoffset = csdVals[0];
                csVals.C0 = csdVals[1];
                csVals.C1 = csdVals[2];
                csVals.C2 = csdVals[3];
                csVals.C3 = csdVals[4];
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("CubicSplineProcessor:readCoefsFromFile: Exception: " << e.what());
            error = true;
        }
        return error;
    }



    //Read the Cubic Spline Coeficients from the given file.
    //Return false if ok, true if error
    bool CubicSplineProcessor::readCoefsFromFile(std::string filename)
    {
        bool error = false;
        std::ifstream csFile;
        std::string line;
        csCoefs.clear();
        string field;
        CubicSplineCoefs_t coefSet;
        try
        {
            ios_base::openmode fileMode = ios_base::in ;
            csFile.open(filename.c_str(), fileMode);
            if(csFile.is_open())
            {
                std::getline(csFile, line);
                LOGINFO("Read Cubic Spline File: " << filename << " Hdr: " << line);
                std::getline(csFile, line);
                LOGINFO("CS File: " << line);  //Number of Records
                while(getline(csFile, line))
                {
                    if(!parseLineToCSCoefVals(line, coefSet))
                    {
                        csCoefs.push_back(coefSet);
                    }
                    else
                    {
                        LOGERROR("CubicSplineProcessor:readCoefsFromFile: error reading file: " << filename);
                        error = true;
                        break;
                    }
                }
                csFile.close();
            }
            else
            {
                LOGERROR("CubicSplineProcessor:readCoefsFromFile: Could not open file: " << filename);
                error = true;
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("CubicSplineProcessor:readCoefsFromFile: Exception: " << e.what());
            error = true;
        }
        return error;
    }


    double CubicSplineProcessor::computeFx(double x)
    {
        double fxVal = 0;
        int N = csCoefs.size();
        if( N > 2)
        {
            double delx = csCoefs[1].Xoffset - csCoefs[0].Xoffset;
            int n = (int)(x / delx);
            n = n < 0 ? 0 : n > N - 1 ? N - 1 : n;
            double xp = x - csCoefs[n].Xoffset;
            fxVal = csCoefs[n].C0;
            fxVal += csCoefs[n].C1 * xp;
            xp = xp * xp;
            fxVal += csCoefs[n].C2 * xp;
            xp = xp * xp;
            fxVal += csCoefs[n].C3 * xp;
        }
        return fxVal;
    }

    double CubicSplineProcessor::computeFx1stDir(double x)
    {
        double fxVal = 0;
        int N = csCoefs.size();
        if( N > 2)
        {
            double delx = csCoefs[1].Xoffset - csCoefs[0].Xoffset;
            int n = (int)(x / delx);
            n = n < 0 ? 0 : n > N - 1 ? N - 1 : n;
            double xp = x - csCoefs[n].Xoffset;
            fxVal = csCoefs[n].C1;
            fxVal += 2.0 * csCoefs[n].C2 * xp;
            xp = xp * xp;
            fxVal += 3.0 * csCoefs[n].C3 * xp;
        }
        return fxVal;
    }

    double CubicSplineProcessor::computeFx2ndDir(double x)
    {
        double fxVal = 0;
        int N = csCoefs.size();
        if( N > 2)
        {
            double delx = csCoefs[1].Xoffset - csCoefs[0].Xoffset;
            int n = (int)(x / delx);
            n = n < 0 ? 0 : n > N - 1 ? N - 1 : n;
            double xp = x - csCoefs[n].Xoffset;
            fxVal = 2.0 * csCoefs[n].C2;
            fxVal += 6.0 * csCoefs[n].C3 * xp;
        }
        return fxVal;
    }

    double CubicSplineProcessor::computeFx3rdDir(double x)
    {
        double fxVal = 0;
        int N = csCoefs.size();
        if( N > 2)
        {
            double delx = csCoefs[1].Xoffset - csCoefs[0].Xoffset;
            int n = (int)(x / delx);
            n = n < 0 ? 0 : n > N - 1 ? N - 1 : n;
            fxVal = 6.0 * csCoefs[n].C3;
        }
        return fxVal;
    }


}