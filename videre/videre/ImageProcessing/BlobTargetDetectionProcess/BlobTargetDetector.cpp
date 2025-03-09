/* ****************************************************************
 * Blob Target Detector
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Sept 2016
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
 * The Blob Target Detector was orginally developed for the JHART
 * and earlier Huntsman programs by Dr. Hyukseong Kwon in the June 2012
 * timeframe.   It was updated for the Dominator project by Dr. Gruber
 * in March 2014.
 *
 * The Blob Target Detector is being optimized to use NVidia's GPGPU
 * processor technology.  OpenCV has optimizations for NVidia's GPGPU
 * in addition other blob detection algorithms are being optimized
 * to use the GPGPU.
 *
 * The Blob Target Detector makes use of a Blob Library.  I do not have
 * the details of where this library came from other than the header
 * information in the files.  Modifications and optimizations to this
 * library are being made as necessary to suport the Blob Target Detector.
 *
  *******************************************************************/

#include "BlobTargetDetector.h"
#include <math.h>
#include <GeoCoordinateSystem.h>


using namespace std;
using namespace cv;
using namespace VidereImageprocessing;
using namespace GeoCoordinateSystemNS;


namespace TargetDetectorNS
{
    //There were two Blob Target Detectors to choose from.  The Old/Std one
    //is being dropped because is uses old opencv routines that are no longer
    //supported.  Right now I am cheating and selecting the newer OpenCVSimple 
    //blob detector when the old one is requested.

    BlobTargetDetector::BlobTargetDetector(Rabit::RabitManager* mgrPtr,
                                           std::shared_ptr<ConfigData> config,
                                           bool useOpenCVBlobDetector,
                                           bool useGPGPUAcceleration)
        :TargetDetector(mgrPtr, config)
    {
        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        _videreCfg = config;

#ifdef CUDA
        if(useGPGPUAcceleration)
        {
            _cudaBlobTargetDetectorOpenCVPtr = new CudaBlobTargetDetectorOpenCVNS::CudaBlobTargetDetectorOpenCV();
            _cudaTargetDetectorEnabled = true;
            LOGINFO("BlobTargetDetector: (OpenCV Cuda) Created!");
            cout << "BlobTargetDetector: (OpenCV Cuda) Created!" << endl;
        }
        else
        {
	    _stdBlobTargetDetectorOpenCVPtr = new StdBlobTargetDetectorOpenCVNS::StdBlobTargetDetectorOpenCVSimple();
	    _cudaTargetDetectorEnabled = false;
	    LOGINFO("BlobTargetDetector: (OpenCV Std) Created.");
	    cout << "BlobTargetDetector: (OpenCV Std) Created." << endl;
        }
#else
        _stdBlobTargetDetectorOpenCVPtr = new StdBlobTargetDetectorOpenCVNS::StdBlobTargetDetectorOpenCVSimple();
        _cudaTargetDetectorEnabled = false;
        LOGINFO("BlobTargetDetector: (OpenCV Std) Created.");
        cout << "BlobTargetDetector: (OpenCV Std) Created." << endl;
#endif

    }

    BlobTargetDetector::~BlobTargetDetector()
    {
        if(_stdBlobTargetDetectorOpenCVPtr != nullptr)
        {
            _stdBlobTargetDetectorOpenCVPtr->Close();
            delete(_stdBlobTargetDetectorOpenCVPtr);
        }
#ifdef CUDA
        if(_cudaBlobTargetDetectorOpenCVPtr != nullptr)
        {
            _cudaBlobTargetDetectorOpenCVPtr->Close();
            delete(_cudaBlobTargetDetectorOpenCVPtr);
        }
#endif
    }


    void BlobTargetDetector::ReadBlobDetectorConfigParameters()
    {
        PixelColorValue_t colorVal;
        u_char R, G, B;
        string colorFormat;
        ImageColorFormat_e cf = ImageColorFormat_e::IPCF_RGB;

        BlobDetectorParameters.UseGausianFilter = _videreCfg->GetConfigBoolValue("BlobDetectorParameters.UseGausianFilter", false);
        BlobDetectorParameters.setGausianFilterKernalSize(_videreCfg->GetConfigIntValue("BlobDetectorParameters.GausianFilterKernalSize", 0));

        BlobDetectorParameters.setBlobDetectorFilterByArea(true);
        BlobDetectorParameters.setBlobDetectorMinArea(_videreCfg->GetConfigDoubleValue("BlobDetectorParameters.MinArea", 25.0));
        BlobDetectorParameters.setBlobDetectorMaxArea(_videreCfg->GetConfigDoubleValue("BlobDetectorParameters.MaxArea", 500.0));

        BlobDetectorParameters.setBlobDetectorMinDistBetweenBlobs(_videreCfg->GetConfigDoubleValue("BlobDetectorParameters.MinDistBetweenBlobs", 100.0));

        BlobDetectorParameters.setBlobDetectorFilterByCircularity(_videreCfg->GetConfigBoolValue("BlobDetectorParameters.FilterByCircularity", false));
        BlobDetectorParameters.setBlobDetectorMinCircularity(_videreCfg->GetConfigDoubleValue("BlobDetectorParameters.MinCircularity", 0.1));
        BlobDetectorParameters.setBlobDetectorMaxCircularity(_videreCfg->GetConfigDoubleValue("BlobDetectorParameters.MaxCircularity", 1.0));

        BlobDetectorParameters.setBlobDetectorFilterByConvexity(_videreCfg->GetConfigBoolValue("BlobDetectorParameters.FilterByConvexity", false));
        BlobDetectorParameters.setBlobDetectorMinConvexity(_videreCfg->GetConfigDoubleValue("BlobDetectorParameters.MinConvexity", 0.1));
        BlobDetectorParameters.setBlobDetectorMaxConvexity(_videreCfg->GetConfigDoubleValue("BlobDetectorParameters.MaxConvexity", 1.0));

        BlobDetectorParameters.setBlobDetectorFilterByInertia(_videreCfg->GetConfigBoolValue("BlobDetectorParameters.FilterByInertia", false));
        BlobDetectorParameters.setBlobDetectorMinInertiaRatio(_videreCfg->GetConfigDoubleValue("BlobDetectorParameters.MinInertiaRatio", 0.1));
        BlobDetectorParameters.setBlobDetectorMaxInertiaRatio(_videreCfg->GetConfigDoubleValue("BlobDetectorParameters.MaxInertiaRatio", 1.0));

        colorFormat = _videreCfg->GetConfigStringValue("BlobDetectorParameters.ColorFormat", "RGB");
        if(colorFormat == "HSV") cf = ImageColorFormat_e::IPCF_HSV;
        else if(colorFormat == "HSL") cf = ImageColorFormat_e::IPCF_HSL;
        else if(colorFormat == "HSI") cf = ImageColorFormat_e::IPCF_HSI;
        else if(colorFormat == "YCrCb") cf = ImageColorFormat_e::IPCF_YCrCb;
        else cf = ImageColorFormat_e::IPCF_RGB;

        if(cf == ImageColorFormat_e::IPCF_RGB)
        {
            R = (u_char) _videreCfg->GetConfigIntValue("BlobDetectorParameters.MinRedValue", 150);
            G = (u_char) _videreCfg->GetConfigIntValue("BlobDetectorParameters.MinGreenValue", 0);
            B = (u_char) _videreCfg->GetConfigIntValue("BlobDetectorParameters.MinBlueValue", 0);
            BlobDetectorParameters.BlobMinColorValue.setRGBColor(R, G, B);

            R = (u_char) _videreCfg->GetConfigIntValue("BlobDetectorParameters.MaxRedValue", 255);
            G = (u_char) _videreCfg->GetConfigIntValue("BlobDetectorParameters.MaxGreenValue", 80);
            B = (u_char) _videreCfg->GetConfigIntValue("BlobDetectorParameters.MaxBlueValue", 80);
            BlobDetectorParameters.BlobMaxColorValue.setRGBColor(R, G, B);
        }
        else if(cf == ImageColorFormat_e::IPCF_YCrCb)
        {
            R = (u_char) _videreCfg->GetConfigIntValue("BlobDetectorParameters.MinYValue", 0);
            G = (u_char) _videreCfg->GetConfigIntValue("BlobDetectorParameters.MinCrValue", 160);
            B = (u_char) _videreCfg->GetConfigIntValue("BlobDetectorParameters.MinCbValue", 0);
            BlobDetectorParameters.BlobMinColorValue.setRGBColor(R, G, B);
            BlobDetectorParameters.BlobMinColorValue.colorFormat = ImageColorFormat_e::IPCF_YCrCb;

            R = (u_char)_videreCfg->GetConfigIntValue("BlobDetectorParameters.MaxYValue", 255);
            G = (u_char)_videreCfg->GetConfigIntValue("BlobDetectorParameters.MaxCrValue", 255);
            B = (u_char)_videreCfg->GetConfigIntValue("BlobDetectorParameters.MaxCbValue", 145);
            BlobDetectorParameters.BlobMaxColorValue.setRGBColor(R, G, B);
            BlobDetectorParameters.BlobMaxColorValue.colorFormat = ImageColorFormat_e::IPCF_YCrCb;
        }
        else  //Assume HSV, HSL or HSI format
        {
            double Hue, Sat, Val;
            Hue = _videreCfg->GetConfigDoubleValue("BlobDetectorParameters.MinHue", 330.0);
            Sat = _videreCfg->GetConfigDoubleValue("BlobDetectorParameters.MinSaturation", 0.0);
            Val = _videreCfg->GetConfigDoubleValue("BlobDetectorParameters.MinLum", 0.0);
            BlobDetectorParameters.BlobMinColorValue.setHSxColor(Hue, Sat, Val, cf);

            Hue = _videreCfg->GetConfigDoubleValue("BlobDetectorParameters.MaxHue", 15.0);
            Sat = _videreCfg->GetConfigDoubleValue("BlobDetectorParameters.MaxSaturation", 100.0);
            Val = _videreCfg->GetConfigDoubleValue("BlobDetectorParameters.MaxLum", 75.0);
            BlobDetectorParameters.BlobMaxColorValue.setHSxColor(Hue, Sat, Val, cf);
        }

        R = (u_char)_videreCfg->GetConfigIntValue("BlobDetectorParameters.DispTargetRedValue", 0);
        G = (u_char)_videreCfg->GetConfigIntValue("BlobDetectorParameters.DispTargetGreenValue", 0);
        B = (u_char)_videreCfg->GetConfigIntValue("BlobDetectorParameters.DispTargetBlueValue", 0);
        TargetColorVal.setRGBColor(R, G, B);

    }



    bool BlobTargetDetector::Initialize()
    {
        bool error = false;

        _cameraOrientationValidation.ReadValsFromConfigFile(_videreCfg);
        //Keep the _imageProcessingParamsMsg from changing the parameters...
        //A new message can be used to update the parameters if the new
        //messasge arrives after setting the default config values.
        _imageProcessingParamsMsg->FetchMessage();
        ReadBlobDetectorConfigParameters();

#ifdef CUDA
        if(_cudaBlobTargetDetectorOpenCVPtr != nullptr)
        {
            error = _cudaBlobTargetDetectorOpenCVPtr->Initialize();
        }
#endif
        if(_stdBlobTargetDetectorOpenCVPtr != nullptr)
        {
            error = _stdBlobTargetDetectorOpenCVPtr->Initialize();
        }
        return error;
    }

    void BlobTargetDetector::Close()
    {
#ifdef CUDA
        if(_cudaBlobTargetDetectorOpenCVPtr != nullptr)
        {
            _cudaBlobTargetDetectorOpenCVPtr->Close();
        }
#endif
        if(_stdBlobTargetDetectorOpenCVPtr != nullptr)
        {
            _stdBlobTargetDetectorOpenCVPtr->Close();
        }
    }


    //Target Detectors may be holding resources such as cv:Mat memory
    //and other resourses that should be released when changing to another
    //image processing routine.
    void BlobTargetDetector::ReleaseResources()
    {
#ifdef CUDA
        if(_cudaBlobTargetDetectorOpenCVPtr != nullptr)
        {
            _cudaBlobTargetDetectorOpenCVPtr->releaseResources();
        }
#endif
        if(_stdBlobTargetDetectorOpenCVPtr != nullptr)
        {
            _stdBlobTargetDetectorOpenCVPtr->releaseResources();
        }
    }

    //Check and update target and image processing prameters if necessary.
    void BlobTargetDetector::CheckUpdateTargetParameters(bool forceUpdates)
    {
        bool pchange = forceUpdates;
        //pchange |= _targetType1ParamsMsg->FetchMessage();
        //pchange |= _targetType2ParamsMsg->FetchMessage();
        //pchange |= _targetType3ParamsMsg->FetchMessage();
        //pchange |= _targetType4ParamsMsg->FetchMessage();
        pchange |= _imageProcessingParamsMsg->FetchMessage();
        if(pchange)
        {

            BlobDetectorParameters.UseGausianFilter = _imageProcessingParamsMsg->ParamI_1 != 0;
            BlobDetectorParameters.setGausianFilterKernalSize(_imageProcessingParamsMsg->ParamI_1);

            BlobDetectorParameters.setBlobDetectorFilterByArea(true);
            BlobDetectorParameters.setBlobDetectorMinArea(_imageProcessingParamsMsg->ParamF_10);
            BlobDetectorParameters.setBlobDetectorMaxArea(_imageProcessingParamsMsg->ParamF_11);

            BlobDetectorParameters.setBlobDetectorMinDistBetweenBlobs(_imageProcessingParamsMsg->ParamF_12);

            BlobDetectorParameters.setBlobDetectorFilterByCircularity(_imageProcessingParamsMsg->ParamF_14 > 0);
            BlobDetectorParameters.setBlobDetectorMinCircularity(_imageProcessingParamsMsg->ParamF_14);
            BlobDetectorParameters.setBlobDetectorMaxCircularity(_imageProcessingParamsMsg->ParamF_15);

            BlobDetectorParameters.setBlobDetectorFilterByConvexity(_imageProcessingParamsMsg->ParamF_16 > 0);
            BlobDetectorParameters.setBlobDetectorMinConvexity(_imageProcessingParamsMsg->ParamF_16);
            BlobDetectorParameters.setBlobDetectorMaxConvexity(_imageProcessingParamsMsg->ParamF_17);

            BlobDetectorParameters.setBlobDetectorFilterByInertia(_imageProcessingParamsMsg->ParamF_18 > 0);
            BlobDetectorParameters.setBlobDetectorMinInertiaRatio(_imageProcessingParamsMsg->ParamF_18);
            BlobDetectorParameters.setBlobDetectorMaxInertiaRatio(_imageProcessingParamsMsg->ParamF_19);

            //Color Boundaries
            BlobDetectorParameters.BlobMinColorValue.setColorWithFormat((unsigned int)_imageProcessingParamsMsg->ParamI_2);
            BlobDetectorParameters.BlobMaxColorValue .setColorWithFormat((unsigned int)_imageProcessingParamsMsg->ParamI_3);


#ifdef CUDA
            if(_cudaBlobTargetDetectorOpenCVPtr != nullptr)
            {
                _cudaBlobTargetDetectorOpenCVPtr->BlobDetectorParameters = BlobDetectorParameters;
            }
#endif
            if(_stdBlobTargetDetectorOpenCVPtr != nullptr)
            {
                _stdBlobTargetDetectorOpenCVPtr->BlobDetectorParameters = BlobDetectorParameters;
            }
        }
    }


    //Target Detection Method
    //Handles the Image Processing for detecting targets
    //and returns the results in the provide message.
    //Returns the number of targets found
    //A number less than zero indicates an error.
    int BlobTargetDetector::DetectTargets(ImagePlusMetadataMessage* imagePlusMetaData,
                                          ImageProcTargetInfoResultsMessage* targetResultsMsg,
                                          std::shared_ptr<ImagePixelLocationToRealWorldLocation> pixeToRealWorldConvertor_sptr)
    {
        int numberOfTargetsFound = 0;
        int imgHeight;
        int imgWidth;
        std::vector<BlobTargetInfo_t> tgtResults;

        _targetResultsMsg = targetResultsMsg;
        _pixeToRealWorldConvertor = pixeToRealWorldConvertor_sptr;

        //First check to ensure the UAV is at a reasonable altitude and
        //bank angle... otherwise the image processing will be meaningless.
        if( !_cameraOrientationValidation.IsCameraOrientationInRange(imagePlusMetaData))
        {
            LOGINFO("DetectTargets: Vehicle Altitude, Roll, Pitch, or Camera Angle is out of rnage.")
            return 0;
        }

        //Get a reference to the Sensor Image passed into the method.
        //imgInpRGB = &imagePlusMetaData->ImageFrame;
#ifdef CUDA
        if(_cudaBlobTargetDetectorOpenCVPtr != nullptr)
        {
            numberOfTargetsFound = _cudaBlobTargetDetectorOpenCVPtr->DetectTargets(&imagePlusMetaData->GetCudaImageMat(),
                                                                                   &TargetResults);
        }
        else if(_stdBlobTargetDetectorOpenCVPtr != nullptr)
        {
            numberOfTargetsFound = _stdBlobTargetDetectorOpenCVPtr->DetectTargets(&imagePlusMetaData->ImageFrame,
                                                                                  &TargetResults);
        }
#else
        if(_stdBlobTargetDetectorOpenCVPtr != nullptr)
        {
            numberOfTargetsFound = _stdBlobTargetDetectorOpenCVPtr->DetectTargets(&imagePlusMetaData->ImageFrame, &TargetResults);
        }
#endif
        if(numberOfTargetsFound > 0)
        {
            //ToDo:  establish target type.
            AddTargetsFoundToTargetList(1, 4);
        }

        return numberOfTargetsFound;
    }


    int BlobTargetDetector::AddTargetsFoundToTargetList(int targetType, int maxNoTgtsToAdd)
    {
        int NoTgtsAdded = 0;
        int brSortedIdexes[MAX_NUMBER_TARGETS_PER_IMAGE];
        double brAreaFitError[MAX_NUMBER_TARGETS_PER_IMAGE];
        XYZCoord_t xyzCoord;
        LatLonAltCoord_t latLonAltCoord;
        AzimuthElevation_t azimuthElevation;
        GeoCoordinateSystem *gcsPtr;

        double parimeterRatioMin = 0;
        double desiredArea = 0;
        double desiredParimeter = 0;
        double tmpDelArea = 0;
        int tmpIdx = 0;

        if(targetType == 1)
        {
            desiredArea = _targetType1ParamsMsg->TargetPerimeterOrLenght * _targetType1ParamsMsg->TargetSizeOrWidth;
            desiredParimeter = 2 * _targetType1ParamsMsg->TargetPerimeterOrLenght + 2.0 * _targetType1ParamsMsg->TargetSizeOrWidth;
            parimeterRatioMin = PATCH_TYPE1_CIRCULAR_RATE;
        }
        else if(targetType == 2)
        {
            desiredArea = _targetType2ParamsMsg->TargetPerimeterOrLenght * _targetType2ParamsMsg->TargetSizeOrWidth;
            desiredParimeter = 2 * _targetType2ParamsMsg->TargetPerimeterOrLenght + 2.0 * _targetType2ParamsMsg->TargetSizeOrWidth;
            parimeterRatioMin = PATCH_TYPE2_CIRCULAR_RATE;
        }
        else
        {
            desiredArea = _targetType1ParamsMsg->TargetPerimeterOrLenght * _targetType1ParamsMsg->TargetSizeOrWidth;
            desiredParimeter = 2 * _targetType1ParamsMsg->TargetPerimeterOrLenght + 2.0 * _targetType1ParamsMsg->TargetSizeOrWidth;
            parimeterRatioMin = PATCH_TYPE1_CIRCULAR_RATE;
        }

        if( _targetResultsMsg != nullptr )
        {
            int N = TargetResults.size();
            if( maxNoTgtsToAdd > MAX_NUMBER_TARGETS_PER_IMAGE
                    || maxNoTgtsToAdd < 1)
            {
                maxNoTgtsToAdd = MAX_NUMBER_TARGETS_PER_IMAGE;
            }

            for(int i = 0; i < N; i++)
            {
                double blobArea = TargetResults[i].TgtAreaSqPixels;
                double blobParimeter = TargetResults[i].TgtParimeterPixels;
                if (blobParimeter > 0)
                {
                    double parimRatio = sqrt(blobArea) / blobParimeter;
                    if (parimRatio > parimeterRatioMin)
                    {
                        //Sort targets by closeness to the desired Area.
                        double delArea = fabs(blobArea - desiredArea);
                        int blobIdx = i;
                        int idx = 0;
                        bool minSet = false;
                        for (; idx < NoTgtsAdded; idx++)
                        {
                            if (delArea < brAreaFitError[idx])
                            {
                                tmpDelArea = brAreaFitError[idx];
                                tmpIdx = brSortedIdexes[idx];
                                brAreaFitError[idx] = delArea;
                                brSortedIdexes[idx] = blobIdx;
                                delArea = tmpDelArea;
                                blobIdx = tmpIdx;
                                minSet = true;
                                break;
                            }
                        }
                        if (minSet)
                        {
                            //shift target info up.
                            for (int j = idx + 1; j < NoTgtsAdded; j++)
                            {
                                tmpDelArea = brAreaFitError[j];
                                tmpIdx = brSortedIdexes[j];
                                brAreaFitError[j] = delArea;
                                brSortedIdexes[j] = blobIdx;
                                delArea = tmpDelArea;
                                blobIdx = tmpIdx;
                            }
                        }
                        if (NoTgtsAdded < maxNoTgtsToAdd)
                        {
                            brAreaFitError[NoTgtsAdded] = delArea;
                            brSortedIdexes[NoTgtsAdded] = blobIdx;
                            ++NoTgtsAdded;
                        }
                    }
                }
            }

            for(int i = 0; i < NoTgtsAdded; i++)
            {
                int idx = brSortedIdexes[i];
                TargetResults[idx].IsTarget = true;
                TargetResults[idx].TargetType = targetType;
                double blob_XPix = TargetResults[idx].TgtCenterPixel_x;
                double blob_YPix = TargetResults[idx].TgtCenterPixel_y;
                double orientationRad = TargetResults[idx].TgtOrientationAngleDeg;

                if (_pixeToRealWorldConvertor != nullptr)
                {
                    _pixeToRealWorldConvertor->CalculateRealWorldLocation(blob_XPix, blob_YPix,
                                                                          &xyzCoord,
                                                                          &azimuthElevation);
                    gcsPtr = GeoCoordinateSystem::GetGeoCoordinateSystemReference();
                    latLonAltCoord = gcsPtr->XYZToLatLonAlt(xyzCoord);
                }

                if( _targetResultsMsg->AddTarget(targetType,
                                             blob_XPix, blob_YPix,
                                             orientationRad,
                                             latLonAltCoord,
                                             azimuthElevation) )
                {
                    LOGERROR("AddTargetsFoundToTargetList: Error addting target to Target Info Message")
                }
            }
        }

        return NoTgtsAdded;
    }


    //Check to see if there is an intermediate image (an image created as part of the
    //taget processing) available.  This image can be used for an output display.
    //Image numbers: [0...N)
    bool BlobTargetDetector::IsIntermediateImageAvailable(int imgNumber)
    {

        return imgNumber == 0 ? true : false;
    }

    //Get an intermediate image.  Return true if image is available and obtained.
    //return false if no image is available.  An image copy is made if makecopy = true.
    //Image numbers: [0...N)
    bool BlobTargetDetector::GetIntermediateImage(int imgNumber, cv::Mat &outImg)
    {
        bool imgObtained = false;
#ifdef CUDA
        if(_cudaBlobTargetDetectorOpenCVPtr != nullptr)
        {
            imgObtained = _cudaBlobTargetDetectorOpenCVPtr->GetIntermediateImage(imgNumber, outImg);
        }
#endif
        if(_stdBlobTargetDetectorOpenCVPtr != nullptr)
        {
            imgObtained = _stdBlobTargetDetectorOpenCVPtr->GetIntermediateImage(imgNumber, outImg);
        }

        return imgObtained;
    }


    void BlobTargetDetector::MarkTargetsOnImage(cv::Mat &image, bool targetsOnly)
    {
        std::vector<cv::KeyPoint> tgtLocs;
        if(TargetResults.size() > 0)
        {
            try
            {
                cv::KeyPoint keypt;
                for(int i = 0; i < TargetResults.size(); i++)
                {
                    if (!targetsOnly || TargetResults[i].IsTarget)
                    {
                        keypt.pt.x = TargetResults[i].TgtCenterPixel_x;
                        keypt.pt.y = TargetResults[i].TgtCenterPixel_y;
                        keypt.size = TargetResults[i].TgtDiameterPixels;
                        keypt.angle = TargetResults[i].TgtOrientationAngleDeg;
                        tgtLocs.push_back(keypt);
                    }
                }
                cv::DrawMatchesFlags flags = cv::DrawMatchesFlags::DRAW_OVER_OUTIMG;
                flags |= cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS;
                 cv::drawKeypoints(image, tgtLocs, image,
                                   Scalar(TargetColorVal.c2,TargetColorVal.c1,TargetColorVal.c0),
                                   flags);
            }
            catch (std::exception &e)
            {
                LOGERROR("BlobTargetDetector:MarkTargetsOnImage: Exception: " << e.what());
            }
        }

    }
}
