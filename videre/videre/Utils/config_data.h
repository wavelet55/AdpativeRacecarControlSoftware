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

#ifndef CONFIG_DATA
#define CONFIG_DATA

#include <iostream>
#include <string>
#include <memory>
#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include "config_parse_exception.h"

namespace videre
{

    class ConfigData
    {

    private:
        std::string filename_;
        boost::property_tree::ptree _pt;

    public:
        ConfigData()
        {

        }

        void ParseConfigFile(std::string filename)
        {
            filename_ = filename;
            std::string path;

            boost::property_tree::ini_parser::read_ini(filename, _pt);
        }

        template<typename T>
        T GetByPath(std::string path_to_value)
        {
            try
            {
                return _pt.get<T>(path_to_value);

            } catch (boost::property_tree::ptree_bad_data &e1)
            {
                auto msg = "Data at " + path_to_value + " within " + filename_ + " cannot be parsed.";
                throw ConfigParseException(msg);
            } catch (boost::property_tree::ptree_bad_path &e2)
            {
                auto msg = "The path " + path_to_value + " within " + filename_ + " cannot be found.";
                throw ConfigParseException(msg);
            }
        }

        std::string GetConfigStringValue(const std::string &configName, const std::string &defaultValue = nullptr);

        std::string GetConfigStringLowerCaseValue(const std::string &configName, const std::string &defaultValue );

        int GetConfigIntValue(std::string configName, int defaultValue = 0);

        double GetConfigDoubleValue(std::string configName, double defaultValue = 0.0);

        bool GetConfigBoolValue(std::string configName, bool defaultValue = false);

    };

}

#endif //CONFIG_DATA
