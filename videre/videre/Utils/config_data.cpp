/* ****************************************************************
 * Athr(s): Harry Direen PhD, Randy Direen Phd.
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: Aug. 2016
 *
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
 *******************************************************************/

#include "config_data.h"
#include <boost/algorithm/string.hpp>

using namespace std;

namespace videre
{


    std::string ConfigData::GetConfigStringValue(const std::string &configName, const std::string &defaultValue)
    {
        std::string parmStrVal;
        try
        {
            parmStrVal = GetByPath<string>(configName);
        }
        catch (std::exception &e)
        {
            parmStrVal = defaultValue;
        }
        return parmStrVal;
    }

    std::string ConfigData::GetConfigStringLowerCaseValue(const std::string &configName, const std::string &defaultValue)
    {
        std::string parmStrVal;
        try
        {
            parmStrVal = GetByPath<string>(configName);
        }
        catch (std::exception &e)
        {
            parmStrVal = defaultValue;
        }
        boost::algorithm::to_lower(parmStrVal);
        return parmStrVal;
    }


    int ConfigData::GetConfigIntValue(std::string configName, int defaultValue)
    {
        int parmVal;
        try
        {
            parmVal = GetByPath<int>(configName);
        }
        catch (std::exception &e)
        {
            parmVal = defaultValue;
        }
        return parmVal;
    }

    double ConfigData::GetConfigDoubleValue(std::string configName, double defaultValue)
    {
        double parmVal;
        try
        {
            parmVal = GetByPath<double>(configName);
        }
        catch (std::exception &e)
        {
            parmVal = defaultValue;
        }
        return parmVal;
    }

    bool ConfigData::GetConfigBoolValue(std::string configName, bool defaultValue)
    {
        bool parmVal;
        try
        {
            parmVal = GetByPath<bool>(configName);
        }
        catch (std::exception &e)
        {
            parmVal = defaultValue;
        }
        return parmVal;
    }

}