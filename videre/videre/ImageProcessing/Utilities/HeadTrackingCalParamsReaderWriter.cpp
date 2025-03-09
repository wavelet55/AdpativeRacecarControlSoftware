/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Aug. 2018
 *
  *******************************************************************/

#include "HeadTrackingCalParamsReaderWriter.h"
#include "Quaternion.h"

using namespace std;

namespace videre
{


    void ReadHeadTrackingParametersFromIniFile(string filename, ImageProcLibsNS::HeadTrackingParameters_t &htParams)
    {
        std::shared_ptr<ConfigData> calDataCfg = make_shared<ConfigData>();
        try
        {
            calDataCfg->ParseConfigFile(filename);
            ReadHeadTrackingParametersFromConfig(calDataCfg, htParams);
        }
        catch (exception &e)
        {
            auto msg = "Error reading Camera Calibration File: " + filename
                       + " Exception: " + e.what();
            throw ConfigParseException(msg);
        }
    }


    void ReadHeadTrackingParametersFromConfig(std::shared_ptr<ConfigData> cfg, ImageProcLibsNS::HeadTrackingParameters_t &htParams)
    {
        int ival;
        double dval;
        ival = (int)cfg->GetConfigDoubleValue("HeadTrackingParameters.Canny_low", 50);
        htParams.Canny_low = ival < 1 ? 1 : ival > 255 ? 255 : ival;
        ival = (int)cfg->GetConfigDoubleValue("HeadTrackingParameters.Canny_high", 150);
        htParams.Canny_high = ival < 10 ? 10 : ival > 255 ? 255 : ival;
        ival = cfg->GetConfigIntValue("HeadTrackingParameters.GlyphAreaPixels_min", 1000);
        htParams.GlyphAreaPixels_min = ival < 10 ? 10 : ival > 10000 ? 10000 : ival;
        ival = cfg->GetConfigIntValue("HeadTrackingParameters.GlyphAreaPixels_max", 8000);
        htParams.GlyphAreaPixels_max = ival < 10 ? 10 : ival > 100000 ? 100000 : ival;

        ival = cfg->GetConfigIntValue("HeadTrackingParameters.NumberOfIterations", 10);
        htParams.NumberOfIterations = ival < 1 ? 1 : ival > 100 ? 100 : ival;

        dval = cfg->GetConfigDoubleValue("HeadTrackingParameters.ReprojectionErrorDistance", 5.0);
        htParams.ReprojectionErrorDistance = dval < 1.0 ? 1.0 : dval > 100.0 ? 100.0 : dval;
        dval = cfg->GetConfigDoubleValue("HeadTrackingParameters.ConfidencePercent", 95.0);
        htParams.ConfidencePercent = dval < 1.0 ? 1.0 : dval > 100.0 ? 100.0 : dval;
    }


    void WriteHeadTrackingParametersToIniFile(string filename, ImageProcLibsNS::HeadTrackingParameters_t &htParams)
    {
        std::ofstream cfgFile;
        ios_base::openmode fileMode = ios_base::out | ios_base::trunc;
        try
        {
            cfgFile.open(filename, fileMode);

            try
            {
                WriteHeadTrackingParametersToFile(cfgFile, htParams);
            }
            catch (exception &e)
            {
                cfgFile.close();
                throw ConfigParseException(e.what());
            }

            cfgFile.flush();
            cfgFile.close();
        }
        catch (exception &e)
        {
            std::ostringstream msgBuf;
            msgBuf << "Error Creating Camera Calibration File: " << filename
                   << " Exception: " << e.what() << endl;
            throw ConfigParseException(msgBuf.str());
        }
    }

    void WriteHeadTrackingParametersToFile(std::ofstream &cfgFile, ImageProcLibsNS::HeadTrackingParameters_t &htParams)
    {
        try
        {
            cfgFile << "#Head Tracking Parameters Data" << endl << endl;
            cfgFile << "[HeadTrackingParameters]" << endl;

            cfgFile << "Canny_low = " << htParams.Canny_low << endl;
            cfgFile << "Canny_high = " << htParams.Canny_high << endl;
            cfgFile << "GlyphAreaPixels_min = " << htParams.GlyphAreaPixels_min << endl;
            cfgFile << "GlyphAreaPixels_max = " << htParams.GlyphAreaPixels_max << endl;
            cfgFile << "NumberOfIterations = " << htParams.NumberOfIterations << endl;
            cfgFile << "ReprojectionErrorDistance = " << htParams.ReprojectionErrorDistance << endl;
            cfgFile << "ConfidencePercent = " << htParams.ConfidencePercent << endl;
        }
        catch(exception &e)
        {
            std::ostringstream msgBuf;
            msgBuf << "Error writing to ini Camera Calibration File, Exception: " << e.what() << endl;
            throw ConfigParseException(msgBuf.str());
        }
    }

    void ReadHeadOrientationCalDataFromIniFile(std::string filename, ImageProcLibsNS::HeadOrientationCalData_t &calData)
    {
        std::shared_ptr<ConfigData> calDataCfg = make_shared<ConfigData>();
        try
        {
            calDataCfg->ParseConfigFile(filename);
            ReadHeadOrientationCalDataFromConfig(calDataCfg, calData);
        }
        catch (exception &e)
        {
            auto msg = "Error reading Camera Calibration File: " + filename
                       + " Exception: " + e.what();
            throw ConfigParseException(msg);
        }
    }

    void ReadHeadOrientationCalDataFromConfig(std::shared_ptr<ConfigData> cfg, ImageProcLibsNS::HeadOrientationCalData_t &calData)
    {
        double dval;
        dval = cfg->GetConfigDoubleValue("HeadToModelQuaternion.Q_w", 1.0);
        dval = dval < -1.0 ? -1.0 : dval > 1.0 ? 1.0 : dval;
        calData.HeadToModelQ.qScale = dval;
        dval = cfg->GetConfigDoubleValue("HeadToModelQuaternion.Q_x", 0.0);
        dval = dval < -1.0 ? -1.0 : dval > 1.0 ? 1.0 : dval;
        calData.HeadToModelQ.qVec.x = dval;
        dval = cfg->GetConfigDoubleValue("HeadToModelQuaternion.Q_y", 0.0);
        dval = dval < -1.0 ? -1.0 : dval > 1.0 ? 1.0 : dval;
        calData.HeadToModelQ.qVec.y = dval;
        dval = cfg->GetConfigDoubleValue("HeadToModelQuaternion.Q_z", 0.0);
        dval = dval < -1.0 ? -1.0 : dval > 1.0 ? 1.0 : dval;
        calData.HeadToModelQ.qVec.z = dval;

        dval = cfg->GetConfigDoubleValue("CameraToCarQuaternion.Q_w", 1.0);
        dval = dval < -1.0 ? -1.0 : dval > 1.0 ? 1.0 : dval;
        calData.CameraToCarQ.qScale = dval;
        dval = cfg->GetConfigDoubleValue("CameraToCarQuaternion.Q_x", 0.0);
        dval = dval < -1.0 ? -1.0 : dval > 1.0 ? 1.0 : dval;
        calData.CameraToCarQ.qVec.x = dval;
        dval = cfg->GetConfigDoubleValue("CameraToCarQuaternion.Q_y", 0.0);
        dval = dval < -1.0 ? -1.0 : dval > 1.0 ? 1.0 : dval;
        calData.CameraToCarQ.qVec.y = dval;
        dval = cfg->GetConfigDoubleValue("CameraToCarQuaternion.Q_z", 0.0);
        dval = dval < -1.0 ? -1.0 : dval > 1.0 ? 1.0 : dval;
        calData.CameraToCarQ.qVec.z = dval;

        dval = cfg->GetConfigDoubleValue("GyroToHeadQuaternion.Q_w", 1.0);
        dval = dval < -1.0 ? -1.0 : dval > 1.0 ? 1.0 : dval;
        calData.GyroToHeadQ.qScale = dval;
        dval = cfg->GetConfigDoubleValue("GyroToHeadQuaternion.Q_x", 0.0);
        dval = dval < -1.0 ? -1.0 : dval > 1.0 ? 1.0 : dval;
        calData.GyroToHeadQ.qVec.x = dval;
        dval = cfg->GetConfigDoubleValue("GyroToHeadQuaternion.Q_y", 0.0);
        dval = dval < -1.0 ? -1.0 : dval > 1.0 ? 1.0 : dval;
        calData.GyroToHeadQ.qVec.y = dval;
        dval = cfg->GetConfigDoubleValue("GyroToHeadQuaternion.Q_z", 0.0);
        dval = dval < -1.0 ? -1.0 : dval > 1.0 ? 1.0 : dval;
        calData.GyroToHeadQ.qVec.z = dval;
    }


    void WriteHeadOrientationCalDataToIniFile(string filename, ImageProcLibsNS::HeadOrientationCalData_t &calData)
    {
        std::ofstream cfgFile;
        ios_base::openmode fileMode = ios_base::out | ios_base::trunc;
        try
        {
            cfgFile.open(filename, fileMode);

            try
            {
                WriteHeadOrientationCalDataToFile(cfgFile, calData);
            }
            catch (exception &e)
            {
                cfgFile.close();
                throw ConfigParseException(e.what());
            }

            cfgFile.flush();
            cfgFile.close();
        }
        catch (exception &e)
        {
            std::ostringstream msgBuf;
            msgBuf << "Error Creating Camera Calibration File: " << filename
                   << " Exception: " << e.what() << endl;
            throw ConfigParseException(msgBuf.str());
        }
    }

    void WriteHeadOrientationCalDataToFile(std::ofstream &cfgFile, ImageProcLibsNS::HeadOrientationCalData_t &calData)
    {
        try
        {
            cfgFile << "#Head Orientation Calibration Data" << endl << endl;
            cfgFile << "[HeadToModelQuaternion]" << endl;
            cfgFile << "Q_w = " << calData.HeadToModelQ.qScale << endl;
            cfgFile << "Q_x = " << calData.HeadToModelQ.qVec.x << endl;
            cfgFile << "Q_y = " << calData.HeadToModelQ.qVec.y << endl;
            cfgFile << "Q_z = " << calData.HeadToModelQ.qVec.z << endl;

            cfgFile << "[CameraToCarQuaternion]" << endl;
            cfgFile << "Q_w = " << calData.CameraToCarQ.qScale << endl;
            cfgFile << "Q_x = " << calData.CameraToCarQ.qVec.x << endl;
            cfgFile << "Q_y = " << calData.CameraToCarQ.qVec.y << endl;
            cfgFile << "Q_z = " << calData.CameraToCarQ.qVec.z << endl;

            cfgFile << "[GyroToHeadQuaternion]" << endl;
            cfgFile << "Q_w = " << calData.GyroToHeadQ.qScale << endl;
            cfgFile << "Q_x = " << calData.GyroToHeadQ.qVec.x << endl;
            cfgFile << "Q_y = " << calData.GyroToHeadQ.qVec.y << endl;
            cfgFile << "Q_z = " << calData.GyroToHeadQ.qVec.z << endl;


        }
        catch(exception &e)
        {
            std::ostringstream msgBuf;
            msgBuf << "Error writing to ini Camera Calibration File, Exception: " << e.what() << endl;
            throw ConfigParseException(msgBuf.str());
        }
    }



    void ReadHeadModelFromIniFile(string filename, std::vector<cv::Point3f> &headModelData)
    {
        std::shared_ptr<ConfigData> calDataCfg = make_shared<ConfigData>();
        try
        {
            calDataCfg->ParseConfigFile(filename);
            ReadHeadModelFromConfig(calDataCfg, headModelData);
        }
        catch (exception &e)
        {
            auto msg = "Error reading Camera Calibration File: " + filename
                       + " Exception: " + e.what();
            throw ConfigParseException(msg);
        }
    }

    void ReadHeadModelFromConfig(std::shared_ptr<ConfigData> cfg, std::vector<cv::Point3f> &headModelData)
    {
        headModelData.clear();

        float valx, valy, valz;
        valx = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Left.GlyphMtx_0x0", -75.0);
        valy = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Left.GlyphMtx_0x1",  25.0 );
        valz = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Left.GlyphMtx_0x2", -50.0);
        headModelData.push_back(cv::Point3f(valx, valy, valz));

        valx = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Left.GlyphMtx_1x0", -50.0);
        valy = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Left.GlyphMtx_1x1",  22.0);
        valz = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Left.GlyphMtx_1x2", -18.0);
        headModelData.push_back(cv::Point3f(valx, valy, valz));

        valx = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Left.GlyphMtx_2x0", -51.0);
        valy = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Left.GlyphMtx_2x1", -22.0);
        valz = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Left.GlyphMtx_2x2", -18.0);
        headModelData.push_back(cv::Point3f(valx, valy, valz));

        valx = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Left.GlyphMtx_3x0", -76.0);
        valy = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Left.GlyphMtx_3x1", -19.0);
        valz = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Left.GlyphMtx_3x2", -50.0);
        headModelData.push_back(cv::Point3f(valx, valy, valz));

        valx = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Middle.GlyphMtx_0x0", -22.0);
        valy = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Middle.GlyphMtx_0x1", 22.0 );
        valz = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Middle.GlyphMtx_0x2", 0);
        headModelData.push_back(cv::Point3f(valx, valy, valz));

        valx = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Middle.GlyphMtx_1x0", 22.0);
        valy = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Middle.GlyphMtx_1x1", 22.0);
        valz = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Middle.GlyphMtx_1x2", 0);
        headModelData.push_back(cv::Point3f(valx, valy, valz));

        valx = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Middle.GlyphMtx_2x0", 22.0);
        valy = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Middle.GlyphMtx_2x1", -22.0);
        valz = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Middle.GlyphMtx_2x2", 0);
        headModelData.push_back(cv::Point3f(valx, valy, valz));

        valx = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Middle.GlyphMtx_3x0", -22.0);
        valy = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Middle.GlyphMtx_3x1", -22.0);
        valz = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Middle.GlyphMtx_3x2", 0);
        headModelData.push_back(cv::Point3f(valx, valy, valz));

        valx = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Right.GlyphMtx_0x0", -75.0);
        valy = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Right.GlyphMtx_0x1",  25.0 );
        valz = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Right.GlyphMtx_0x2", -50.0);
        headModelData.push_back(cv::Point3f(valx, valy, valz));

        valx = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Right.GlyphMtx_1x0", -50.0);
        valy = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Right.GlyphMtx_1x1",  22.0);
        valz = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Right.GlyphMtx_1x2", -18.0);
        headModelData.push_back(cv::Point3f(valx, valy, valz));

        valx = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Right.GlyphMtx_2x0", -51.0);
        valy = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Right.GlyphMtx_2x1", -22.0);
        valz = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Right.GlyphMtx_2x2", -18.0);
        headModelData.push_back(cv::Point3f(valx, valy, valz));

        valx = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Right.GlyphMtx_3x0", -76.0);
        valy = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Right.GlyphMtx_3x1", -19.0);
        valz = (float)cfg->GetConfigDoubleValue("HeadModelGlyph_Right.GlyphMtx_3x2", -50.0);
        headModelData.push_back(cv::Point3f(valx, valy, valz));
    }


}