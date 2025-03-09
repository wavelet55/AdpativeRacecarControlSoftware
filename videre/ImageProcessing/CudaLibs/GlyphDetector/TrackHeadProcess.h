/* ****************************************************************
 * Blob Target Detector
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Randy Direen PhD
 * Date: July 2018
 *
 *******************************************************************/

#ifndef VIDERE_DEV_TRACKHEADPROCESS_H
#define VIDERE_DEV_TRACKHEADPROCESS_H

#include <vector>
#include <opencv2/core.hpp>
#include "CommonImageProcTypesDefs.h"
#include "opencv2/cudaimgproc.hpp"
#include "DetectGlyphs.h"
#include "HeadPosition.h"
#include "CameraCalibrationData.h"
#include "GlyphModel.h"

namespace CudaImageProcLibsTrackHeadNS
{

    class TrackHeadProcess
    {
    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        //The glyph detector
        DetectGlyphs _dglyphs;

        HeadPosition _headPosition;

        bool _debugMode = false;

        ImageProcLibsNS::HeadTrackingImageDisplayType_e _headTrackingImageDisplayType;

        //A temporary computed image.   This cv::Mat
        cv::Mat _grayImage;

        cv::Mat _outpImage;

        //Cuda Items
        cv::cuda::GpuMat _cframe;
        cv::cuda::GpuMat _cgrayImage;


        bool _useGPU = false;

    public:

        TrackHeadProcess();

        ~TrackHeadProcess();

        //Initialize resources as need.
        //Returns false if ok, true if there was an error.
        bool Initialize(ImageProcLibsNS::CameraCalibrationData &cameraCalData,
                        GlyphModel& glyphModel,
                        ImageProcLibsNS::HeadTrackingParameters_t &headTrackingParameters);

        void SetHeadTrackingParameters(ImageProcLibsNS::HeadTrackingParameters_t &headTrackingParameters);

        void SetHeadTrackingImageDisplayType(ImageProcLibsNS::HeadTrackingImageDisplayType_e dispType);

        //Close out resources used by the Cuda Target Detectory
        void Close();

        void SetUseGPU(bool useGpu)
        {
            _useGPU = useGpu;
            _dglyphs.UseGPU = useGpu;
            _headPosition.UseGPU = useGpu;
        }

        bool GetUseGPU() { return _useGPU; }

        ImageProcLibsNS::TrackHeadOrientationData_t TrackHeadPostion(cv::Mat &imgInpRGB);

        //Get an internal Created Image for output display.  The the output image
        //will be a copy of the internal image.  The imgNumber can be used to select
        //which image to obtain.
        //Returns true if the image was obtained, false otherwise.
        bool GetIntermediateImage(int imgNumber, cv::OutputArray outImg);

    };
}

#endif //VIDERE_DEV_TRACKHEADPROCESS_H
