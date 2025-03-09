/* ****************************************************************
 * Blob Target Detector
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Randy Direen PhD
 * Date: July 2018
 *
 *******************************************************************/

#include "TrackHeadProcess.h"


using namespace cv;
using namespace std;
using namespace ImageProcLibsNS;

namespace CudaImageProcLibsTrackHeadNS
{

    TrackHeadProcess::TrackHeadProcess()
        : _dglyphs()
    {
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);
        LOGINFO("Track Head Processor Created.")
    }

    TrackHeadProcess::~TrackHeadProcess()
    {
        Close();
    };

    //Initialize resources as need.
    //Returns false if ok, true if there was an error.
    bool TrackHeadProcess::Initialize(ImageProcLibsNS::CameraCalibrationData &cameraCalData,
                                      GlyphModel& glyphModel,
                                      ImageProcLibsNS::HeadTrackingParameters_t &headTrackingParameters)
    {
        bool error = true;

        _debugMode = false;

        _dglyphs.set_glyph_model(glyphModel);
        _dglyphs.init();

        _headPosition.init(cameraCalData, glyphModel);
        _headPosition.setDebugInMat(_dglyphs.debugMat());

        SetHeadTrackingParameters(headTrackingParameters);
        SetHeadTrackingImageDisplayType(HeadTrackingImageDisplayType_e::HTID_None);

        LOGINFO("Track Head Processor Initialized.")
        return error;
    };

    void TrackHeadProcess::SetHeadTrackingParameters(ImageProcLibsNS::HeadTrackingParameters_t &headTrackingParameters)
    {
        _dglyphs.set_canny_low_thresh(headTrackingParameters.Canny_low);
        _dglyphs.set_canny_high_thresh(headTrackingParameters.Canny_high);
        _dglyphs.set_canny_kernel(3);
        _dglyphs.set_min_glyph_area(headTrackingParameters.GlyphAreaPixels_min);
        _dglyphs.set_max_glyph_area(headTrackingParameters.GlyphAreaPixels_max);

        _headPosition.set_iterations_count(headTrackingParameters.NumberOfIterations);
        _headPosition.set_reprojection_error(headTrackingParameters.ReprojectionErrorDistance);
        _headPosition.set_confidence(headTrackingParameters.ConfidencePercent);
    }


    //Close out resources used by the Cuda Target Detectory
    void TrackHeadProcess::Close()
    {
        _grayImage.release();
        _outpImage.release();

        LOGINFO("Track Head Processor Closed.")
    }

    void TrackHeadProcess::SetHeadTrackingImageDisplayType(HeadTrackingImageDisplayType_e dispType)
    {
        _headTrackingImageDisplayType = dispType;
        _dglyphs.DisplayType = dispType;
        _headPosition.DisplayType = dispType;
    }


    TrackHeadOrientationData_t TrackHeadProcess::TrackHeadPostion(cv::Mat &imgInpRGB)
    {
        bool glyphsUpdateOk = false;
        try
        {
            _headPosition.Clear();
            _headPosition.setInputImageSize(imgInpRGB);
            _outpImage = imgInpRGB;

            if(_useGPU)
            {
                _dglyphs.UseGPU = true;
                _headPosition.UseGPU = true;
                _cframe.upload(imgInpRGB);
                _cgrayImage.upload(_grayImage);
                cuda::cvtColor(_cframe,_cgrayImage, COLOR_BGR2GRAY);
                glyphsUpdateOk = _dglyphs.update(_cgrayImage);
                _cgrayImage.download(_grayImage);
            }
            else
            {
                _dglyphs.UseGPU = false;
                _headPosition.UseGPU = false;
                cvtColor(imgInpRGB, _grayImage, COLOR_BGR2GRAY);
                glyphsUpdateOk = _dglyphs.update(_grayImage);
            }

            if( glyphsUpdateOk )
            {
                bool dataValid = _headPosition.update(_dglyphs.indices(), _dglyphs.points());

                //Create an output image
                if(_headTrackingImageDisplayType == HeadTrackingImageDisplayType_e::HTID_HighLightGlyphs)
                {
                    vector<array<Point, 4> > squares = _dglyphs.squares();
                    vector<vector<Point> > fcontours = _dglyphs.contours();
                    vector<Vec4i> fhierarchy = _dglyphs.hierarchy();
                    vector<Point2f> mc = _dglyphs.center_mass();

                    Scalar color = Scalar(0, 255, 0);

                    cvtColor(_dglyphs.edge(), _outpImage, cv::COLOR_GRAY2BGR);
                    for(int i = 0; i < fcontours.size(); ++i)
                    {
                        drawContours(_outpImage, fcontours, i, color, 2, 8, fhierarchy, 0);
                        circle(_outpImage, mc[i], 2, Scalar(0, 0, 255), 2, 7, 0);
                    }

                    for(int i = 0; i < squares.size(); ++i)
                    {
                        const Point *p = &squares[i][0];

                        int n = (int) squares[i].size();
                        if(p->x > 3 && p->y > 3)
                        {
                            polylines(_outpImage, &p, &n, 1, true, Scalar(255, 0, 0), 3, LINE_AA);
                        }

                        for(int j = 0; j < 4; ++j)
                        {
                            circle(_outpImage, squares[i][j], 2, Scalar(0, 0, 255), 2, 7, 0);
                        }
                    }
                }
            }
            else
            {
                //Invalid Data... No Glyphs found?
                _headPosition.HeadOrientationData.Clear();
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("TrackHeadProcess:TrackHeadPostion: Exception: " << e.what());
            _headPosition.HeadOrientationData.Clear();
        }
        return _headPosition.HeadOrientationData;
    }

    bool TrackHeadProcess::GetIntermediateImage(int imgNumber, cv::OutputArray outImg)
    {
        bool imgObtained = false;
        if(_headTrackingImageDisplayType == HeadTrackingImageDisplayType_e::HTID_HighLightGlyphs)
        {
            if(_outpImage.rows > 0 && _outpImage.cols > 0)
            {
                _outpImage.copyTo(outImg);
                imgObtained = true;
            }
        }
        if(_headTrackingImageDisplayType == HeadTrackingImageDisplayType_e::HTID_HeadOrientationVector)
        {
            if(_headPosition.debugMat().rows > 0 && _headPosition.debugMat().cols > 0)
            {
                _headPosition.debugMat().copyTo(outImg);
                imgObtained = true;
            }
        }
        return imgObtained;
    }


}