/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: March 2017
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
 *  Routines to read and write camera calibration data files.
  *******************************************************************/
#include "CameraCalReaderWriter.h"
#include "FileUtils.h"
#include <iostream>
#include <fstream>

using namespace ImageProcLibsNS;
using namespace std;
using namespace videre;

namespace CameraCalibrationNS
{

    //Read Camera Calibration from the given file.
    //The file name is the full path to the file.
    //The calibration data is stored in the cameraCalData;
    //Calibration files can contain more than one set of calibration data... use
    //calDataIndex for an index to the calibration data.
    //throws exception:  ConfigParseException() if the file cannot be read.
    void ReadCameraCalibrationFromIniFile(std::string filename,
                                          ImageProcLibsNS::CameraCalibrationData &cameraCalData,
                                          int calDataIndex)
    {
        ConfigData calDataCfg;

        try
        {
            calDataCfg.ParseConfigFile(filename);
            ReadCameraCalibrationDataFromIniConfig(calDataCfg, cameraCalData, calDataIndex);
        }
        catch (exception &e)
        {
            auto msg = "Error reading Camera Calibration File: " + filename
                       + " Exception: " + e.what();
            throw ConfigParseException(msg);
        }
    }

    //Read Camera Calibration from the given file.
    //The file name is the full path to the file.
    //The calibration data is stored in the cameraCalData;
    //Calibration files can contain more than one set of calibration data... use
    //calDataIndex for an index to the calibration data.
    //throws exception:  ConfigParseException() if the file cannot be read.
    void ReadCameraCalibrationDataFromIniConfig(ConfigData &calDataCfg,
                                                ImageProcLibsNS::CameraCalibrationData &cameraCalData,
                                                int calDataIndex)
    {
        double value;
        double valMtx[9];
        std::ostringstream cfgPath, cfgParm;
        cfgPath << "CameraCalibration_" << calDataIndex << ".";
        std::string cfgPathStr = cfgPath.str();

        cfgParm << cfgPathStr << "FocalLength";
        value = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 1.0);
        cameraCalData.SetCameraFocalLength(value);

        cfgParm.str(std::string());  //Clear the string stream
        cfgParm << cfgPathStr << "CalScaleFactor";
        value = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 1.0);
        cameraCalData.SetCalibrationScaleFactor(value);

        cfgParm.str(std::string());  //Clear the string stream
        cfgParm << cfgPathStr << "DistCoef_0";
        valMtx[0] = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
        cfgParm.str(std::string());
        cfgParm << cfgPathStr << "DistCoef_1";
        valMtx[1] = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
        cfgParm.str(std::string());
        cfgParm << cfgPathStr << "DistCoef_2";
        valMtx[2] = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
        cfgParm.str(std::string());
        cfgParm << cfgPathStr << "DistCoef_3";
        valMtx[3] = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
        cfgParm.str(std::string());
        cfgParm << cfgPathStr << "DistCoef_4";
        valMtx[4] = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
        cameraCalData.SetDistortionCalibrationData(valMtx);

        cfgParm.str(std::string());
        cfgParm << cfgPathStr << "IntrinsicCalMtx_0x0";
        valMtx[0] = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
        cfgParm.str(std::string());
        cfgParm << cfgPathStr << "IntrinsicCalMtx_0x1";
        valMtx[1] = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
        cfgParm.str(std::string());
        cfgParm << cfgPathStr << "IntrinsicCalMtx_0x2";
        valMtx[2] = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
        cfgParm.str(std::string());
        cfgParm << cfgPathStr << "IntrinsicCalMtx_1x0";
        valMtx[3] = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
        cfgParm.str(std::string());
        cfgParm << cfgPathStr << "IntrinsicCalMtx_1x1";
        valMtx[4] = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
        cfgParm.str(std::string());
        cfgParm << cfgPathStr << "IntrinsicCalMtx_1x2";
        valMtx[5] = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
        cfgParm.str(std::string());
        cfgParm << cfgPathStr << "IntrinsicCalMtx_2x0";
        valMtx[6] = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
        cfgParm.str(std::string());
        cfgParm << cfgPathStr << "IntrinsicCalMtx_2x1";
        valMtx[7] = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
        cfgParm.str(std::string());
        cfgParm << cfgPathStr << "IntrinsicCalMtx_2x2";
        valMtx[8] = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
        cameraCalData.SetIntrinsicCalibrationData(valMtx);

        cfgParm.str(std::string());  //Clear the string stream
        cfgParm << cfgPathStr << "UseCameraMountingCorrection";
        cameraCalData.UseCameraMountingCorrection = calDataCfg.GetConfigBoolValue(cfgParm.str(), false);

        if( cameraCalData.UseCameraMountingCorrection )
        {
            cfgParm.str(std::string());  //Clear the string stream
            cfgParm << cfgPathStr << "YawCorrectionDegrees";
            value = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
            cameraCalData.SetYawCorrectionDegrees(value);

            cfgParm.str(std::string());  //Clear the string stream
            cfgParm << cfgPathStr << "PitchCorrectionDegrees";
            value = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
            cameraCalData.SetPitchCorrectionDegrees(value);

            cfgParm.str(std::string());  //Clear the string stream
            cfgParm << cfgPathStr << "RollCorrectionDegrees";
            value = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
            cameraCalData.SetRollCorrectionDegrees(value);

            cfgParm.str(std::string());  //Clear the string stream
            cfgParm << cfgPathStr << "DelXCorrectionCentiMeters";
            value = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
            cameraCalData.SetDelXCorrectionCentiMeters(value);

            cfgParm.str(std::string());  //Clear the string stream
            cfgParm << cfgPathStr << "DelYCorrectionCentiMeters";
            value = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
            cameraCalData.SetDelYCorrectionCentiMeters(value);

            cfgParm.str(std::string());  //Clear the string stream
            cfgParm << cfgPathStr << "DelZCorrectionCentiMeters";
            value = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
            cameraCalData.SetDelZCorrectionCentiMeters(value);

            cameraCalData.GenerateRotationXlationCalFromCameraMountingCorrection();
        }
        else
        {
            cfgParm.str(std::string());
            cfgParm << cfgPathStr << "CameraRotationMtx_0x0";
            valMtx[0] = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
            cfgParm.str(std::string());
            cfgParm << cfgPathStr << "CameraRotationMtx_0x1";
            valMtx[1] = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
            cfgParm.str(std::string());
            cfgParm << cfgPathStr << "CameraRotationMtx_0x2";
            valMtx[2] = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
            cfgParm.str(std::string());
            cfgParm << cfgPathStr << "CameraRotationMtx_1x0";
            valMtx[3] = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
            cfgParm.str(std::string());
            cfgParm << cfgPathStr << "CameraRotationMtx_1x1";
            valMtx[4] = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
            cfgParm.str(std::string());
            cfgParm << cfgPathStr << "CameraRotationMtx_1x2";
            valMtx[5] = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
            cfgParm.str(std::string());
            cfgParm << cfgPathStr << "CameraRotationMtx_2x0";
            valMtx[6] = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
            cfgParm.str(std::string());
            cfgParm << cfgPathStr << "CameraRotationMtx_2x1";
            valMtx[7] = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
            cfgParm.str(std::string());
            cfgParm << cfgPathStr << "CameraRotationMtx_2x2";
            valMtx[8] = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
            cameraCalData.SetRotationCalibrationData(valMtx);

            cfgParm.str(std::string());
            cfgParm << cfgPathStr << "CameraTranslationVec_0";
            valMtx[0] = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
            cfgParm.str(std::string());
            cfgParm << cfgPathStr << "CameraTranslationVec_1";
            valMtx[1] = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
            cfgParm.str(std::string());
            cfgParm << cfgPathStr << "CameraTranslationVec_2";
            valMtx[2] = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
            cameraCalData.SetTranslationCalibrationData(valMtx);
        }
    }


    //Write the Camera Calibration Data to an ini file.
    //The full path for the file is: filename
    //Throws excecption:  ConfigParseException() if a file cannot be opened or there
    //is errors writing to the file.
    void WriteCameraCalibrationToIniFile(std::string filename,
                                         const ImageProcLibsNS::CameraCalibrationData &cameraCalData)
    {
        std::ofstream cfgFile;
        ios_base::openmode fileMode = ios_base::out | ios_base::trunc;
        try
        {
            cfgFile.open(filename, fileMode);

            try
            {
                WriteCameraCalibrationDataToIniFile(cfgFile, cameraCalData, 0);
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

    //Write the Camera Calibration Data to an ini file.
    //The full path for the file is: filename
    //Throws excecption:  ConfigParseException() if a file cannot be opened or there
    //is errors writing to the file.
    void WriteCameraCalibrationDataToIniFile(std::ofstream &cfgFile,
                                             const ImageProcLibsNS::CameraCalibrationData &cameraCalData,
                                             int calDataIndex)
    {
        try
        {
            cfgFile << "#Camera Calibration Data" << endl << endl;
            cfgFile << "[CameraCalibration_" << calDataIndex << "]" << endl;

            cfgFile << "FocalLength = " << cameraCalData.GetCameraFocalLength() << endl;
            cfgFile << "CalScaleFactor = " << cameraCalData.GetCalibrationScaleFactor() << endl;

            cfgFile << endl << "#Distortion Coefficient Vector" << endl;
            cfgFile << "DistCoef_0 = " << cameraCalData.cvDistortionCoeff.at<double>(0,0) << endl;
            cfgFile << "DistCoef_1 = " << cameraCalData.cvDistortionCoeff.at<double>(0,1) << endl;
            cfgFile << "DistCoef_2 = " << cameraCalData.cvDistortionCoeff.at<double>(0,2) << endl;
            cfgFile << "DistCoef_3 = " << cameraCalData.cvDistortionCoeff.at<double>(0,3) << endl;
            cfgFile << "DistCoef_4 = " << cameraCalData.cvDistortionCoeff.at<double>(0,4) << endl;

            cfgFile << endl << "#Intrinsic Calibration Matrix 3x3" << endl;
            cfgFile << "IntrinsicCalMtx_0x0 = " << cameraCalData.cvIntrinsicCalM.at<double>(0,0) << endl;
            cfgFile << "IntrinsicCalMtx_0x1 = " << cameraCalData.cvIntrinsicCalM.at<double>(0,1) << endl;
            cfgFile << "IntrinsicCalMtx_0x2 = " << cameraCalData.cvIntrinsicCalM.at<double>(0,2) << endl;
            cfgFile << "IntrinsicCalMtx_1x0 = " << cameraCalData.cvIntrinsicCalM.at<double>(1,0) << endl;
            cfgFile << "IntrinsicCalMtx_1x1 = " << cameraCalData.cvIntrinsicCalM.at<double>(1,1) << endl;
            cfgFile << "IntrinsicCalMtx_1x2 = " << cameraCalData.cvIntrinsicCalM.at<double>(1,2) << endl;
            cfgFile << "IntrinsicCalMtx_2x0 = " << cameraCalData.cvIntrinsicCalM.at<double>(2,0) << endl;
            cfgFile << "IntrinsicCalMtx_2x1 = " << cameraCalData.cvIntrinsicCalM.at<double>(2,1) << endl;
            cfgFile << "IntrinsicCalMtx_2x2 = " << cameraCalData.cvIntrinsicCalM.at<double>(2,2) << endl;

            cfgFile << endl << "#Use Camera Mounting Correction (true)," << endl;
            cfgFile << endl << "#otherwise enter CameraRotationMtx and CameraTranslationVec directly (false)" << endl;
            cfgFile << "UseCameraMountingCorrection = "
                    << (cameraCalData.UseCameraMountingCorrection ? "true" : "false") << endl << endl;

            //if(cameraCalData.UseCameraMountingCorrection)
            {
                cfgFile << "YawCorrectionDegrees = " << cameraCalData.GetYawCorrectionDegrees() << endl;
                cfgFile << "PitchCorrectionDegrees = " << cameraCalData.GetPitchCorrectionDegrees() << endl;
                cfgFile << "RollCorrectionDegrees = " << cameraCalData.GetRollCorrectionDegrees() << endl;
                cfgFile << "DelXCorrectionCentiMeters = " << cameraCalData.GetDelXCorrectionCentiMeters() << endl;
                cfgFile << "DelYCorrectionCentiMeters = " << cameraCalData.GetDelYCorrectionCentiMeters() << endl;
                cfgFile << "DelZCorrectionCentiMeters = " << cameraCalData.GetDelZCorrectionCentiMeters() << endl;
                cfgFile << endl;
            }
            //else
            {
                cfgFile << endl << "#Rotation matrix from the camera coordinate frame" << endl;
                cfgFile << "#to the uav coordinate frame, 3x3" << endl;
                cfgFile << "CameraRotationMtx_0x0 = " << cameraCalData.cvRotationCalM.at<double>(0, 0) << endl;
                cfgFile << "CameraRotationMtx_0x1 = " << cameraCalData.cvRotationCalM.at<double>(0, 1) << endl;
                cfgFile << "CameraRotationMtx_0x2 = " << cameraCalData.cvRotationCalM.at<double>(0, 2) << endl;
                cfgFile << "CameraRotationMtx_1x0 = " << cameraCalData.cvRotationCalM.at<double>(1, 0) << endl;
                cfgFile << "CameraRotationMtx_1x1 = " << cameraCalData.cvRotationCalM.at<double>(1, 1) << endl;
                cfgFile << "CameraRotationMtx_1x2 = " << cameraCalData.cvRotationCalM.at<double>(1, 2) << endl;
                cfgFile << "CameraRotationMtx_2x0 = " << cameraCalData.cvRotationCalM.at<double>(2, 0) << endl;
                cfgFile << "CameraRotationMtx_2x1 = " << cameraCalData.cvRotationCalM.at<double>(2, 1) << endl;
                cfgFile << "CameraRotationMtx_2x2 = " << cameraCalData.cvRotationCalM.at<double>(2, 2) << endl;

                cfgFile << endl << "#Translation vector from the camera coordinate frame to the uav coordinate frame"
                        << endl;
                cfgFile << "CameraTranslationVec_0 = " << cameraCalData.cvTranslationCalM.at<double>(0) << endl;
                cfgFile << "CameraTranslationVec_1 = " << cameraCalData.cvTranslationCalM.at<double>(1) << endl;
                cfgFile << "CameraTranslationVec_2 = " << cameraCalData.cvTranslationCalM.at<double>(2) << endl;
                cfgFile << endl;
            }
        }
        catch (exception &e)
        {
            std::ostringstream msgBuf;
            msgBuf << "Error writing to ini Camera Calibration File, Exception: " << e.what() << endl;
            throw ConfigParseException(msgBuf.str());
        }
    }

}
