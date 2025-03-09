//
// Created by wavelet on 3/29/17.
//

#include "TestCameraCalReaderWriter.h"
#include "CameraCalReaderWriter.h"
#include "config_data.h"
#include "CommonImageProcTypesDefs.h"
#include "global_defines.h"
#include <memory>
#include <string>
#include <boost/filesystem.hpp>
#include "boost/date_time.hpp"
#include "math.h"

using namespace std;
using namespace VidereFileUtils;
using namespace boost::filesystem;
using namespace CameraCalibrationNS;
using namespace ImageProcLibsNS;
using namespace videre;
using namespace cv;

bool compareEqualDouble(double x, double y, double epsolon = 1.0e-6)
{
    bool equal = false;
    if( abs(x - y) < epsolon)
        equal = true;
    return equal;
}

TEST_F(TestCameraCalReaderWriter, TestWriteReadCameraCalDataFiles)
{
    string directory = "CameraCalDir";
    string baseFname = "CameraCalData";
    path dirpath(directory);
    path filepath;
    CameraCalibrationData camCalD1;
    CameraCalibrationData camCalD2;

    if( VidereFileUtils::CreateDirectory(dirpath) )
    {
        filepath = dirpath;
        string filename = VidereFileUtils::AddOrReplaceFileExtention(baseFname, CAMERA_CAL_FILE_EXT);
        filepath /= filename;

        camCalD1.SetCalibrationScaleFactor(2.575);
        camCalD1.SetCameraFocalLength(5.85);
        camCalD1.cvDistortionCoeff.at<double>(0) = 1.235;
        camCalD1.cvDistortionCoeff.at<double>(1) = 2.235;
        camCalD1.cvDistortionCoeff.at<double>(2) = 3.235;
        camCalD1.cvDistortionCoeff.at<double>(3) = 4.235;
        camCalD1.cvDistortionCoeff.at<double>(4) = 5.235;

        camCalD1.cvIntrinsicCalM.at<double>(0,0) = 0.075;
        camCalD1.cvIntrinsicCalM.at<double>(0,1) = 0.175;
        camCalD1.cvIntrinsicCalM.at<double>(0,2) = 0.275;
        camCalD1.cvIntrinsicCalM.at<double>(1,0) = 1.085;
        camCalD1.cvIntrinsicCalM.at<double>(1,1) = 1.185;
        camCalD1.cvIntrinsicCalM.at<double>(1,2) = 1.285;
        camCalD1.cvIntrinsicCalM.at<double>(2,0) = 2.095;
        camCalD1.cvIntrinsicCalM.at<double>(2,1) = 2.195;
        camCalD1.cvIntrinsicCalM.at<double>(2,2) = 2.295;

        camCalD1.cvRotationCalM.at<double>(0,0) = 10.075;
        camCalD1.cvRotationCalM.at<double>(0,1) = 10.175;
        camCalD1.cvRotationCalM.at<double>(0,2) = 10.275;
        camCalD1.cvRotationCalM.at<double>(1,0) = 11.085;
        camCalD1.cvRotationCalM.at<double>(1,1) = 11.185;
        camCalD1.cvRotationCalM.at<double>(1,2) = 11.285;
        camCalD1.cvRotationCalM.at<double>(2,0) = 12.095;
        camCalD1.cvRotationCalM.at<double>(2,1) = 12.195;
        camCalD1.cvRotationCalM.at<double>(2,2) = 12.295;

        camCalD1.cvTranslationCalM.at<double>(0) = 21.235;
        camCalD1.cvTranslationCalM.at<double>(1) = 22.235;
        camCalD1.cvTranslationCalM.at<double>(2) = 23.235;

        try
        {
            WriteCameraCalibrationToIniFile(filepath.string(), camCalD1);
        }
        catch(exception &e)
        {
            cout << "TestWriteReadCameraCal Write Error: " << e.what() << endl;
            return;
        }

        try
        {
            ReadCameraCalibrationFromIniFile(filepath.string(), camCalD2);
            //Validate cal data
            if(!compareEqualDouble(camCalD1.GetCalibrationScaleFactor(), camCalD2.GetCalibrationScaleFactor()))
            {
                cout << "TestWriteReadCameraCal CalibrationScaleFactor Not Equal: D1="
                     << camCalD1.GetCalibrationScaleFactor()
                     << " D2="  << camCalD2.GetCalibrationScaleFactor() << endl;
            }
            if(!compareEqualDouble(camCalD1.GetCameraFocalLength(), camCalD2.GetCameraFocalLength()))
            {
                cout << "TestWriteReadCameraCal CameraFocalLength Not Equal: D1="
                     << camCalD1.GetCameraFocalLength()
                     << " D2="  << camCalD2.GetCameraFocalLength() << endl;
            }

            for(int i = 0; i < 5; i++)
            {
                if(!compareEqualDouble(camCalD1.cvDistortionCoeff.at<double>(i), camCalD2.cvDistortionCoeff.at<double>(i)))
                {
                    cout << "TestWriteReadCameraCal cvDistortionCoeff Not Equal: D1="
                         << camCalD1.cvDistortionCoeff.at<double>(i)
                         << " D2="  << camCalD2.cvDistortionCoeff.at<double>(i) << endl;
                }
            }

            for(int i = 0; i < 3; i++)
            {
                if(!compareEqualDouble(camCalD1.cvTranslationCalM.at<double>(i), camCalD2.cvTranslationCalM.at<double>(i)))
                {
                    cout << "TestWriteReadCameraCal cvTranslationCalM Not Equal: D1="
                         << camCalD1.cvTranslationCalM.at<double>(i)
                         << " D2="  << camCalD2.cvTranslationCalM.at<double>(i) << endl;
                }
            }

            for(int i = 0; i < 3; i++)
            {
                for(int j = 0; j < 3; j++)
                {
                    if (!compareEqualDouble(camCalD1.cvIntrinsicCalM.at<double>(i,j),
                                            camCalD2.cvIntrinsicCalM.at<double>(i,j)))
                    {
                        cout << "TestWriteReadCameraCal cvIntrinsicCalM Not Equal: D1="
                             << camCalD1.cvIntrinsicCalM.at<double>(i,j)
                             << " D2=" << camCalD2.cvIntrinsicCalM.at<double>(i,j) << endl;
                    }
                }
            }

            for(int i = 0; i < 3; i++)
            {
                for(int j = 0; j < 3; j++)
                {
                    if (!compareEqualDouble(camCalD1.cvRotationCalM.at<double>(i,j),
                                            camCalD2.cvRotationCalM.at<double>(i,j)))
                    {
                        cout << "TestWriteReadCameraCal cvRotationCalM Not Equal: D1="
                             << camCalD1.cvRotationCalM.at<double>(i,j)
                             << " D2=" << camCalD2.cvRotationCalM.at<double>(i,j) << endl;
                    }
                }
            }

        }
        catch(exception &e)
        {
            cout << "TestWriteReadCameraCal Write Error: " << e.what() << endl;
            return;
        }
    }
    else
    {
        cout << "Dould not create directory: " << directory;
    }

}