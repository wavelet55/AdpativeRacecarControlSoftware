/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: July. 2018
 *
 *******************************************************************/

#ifndef VIDERE_DEV_HEADTRACKINGPROCESSOR_H
#define VIDERE_DEV_HEADTRACKINGPROCESSOR_H

#include "VisionProcessResources.h"
#include "VisionProcessorAbstract.h"
#include "Utilities/ImageMetadataWriter.h"
#include "GlyphDetector/TrackHeadProcess.h"
#include "CameraCalibrationData.h"
#include "HeadTrackingOrientationMessage.h"
#include "HeadTrackingControlMessage.h"
#include "HeadOrientationControlMessage.h"
#include "DataRecorder.h"
#include "DataRecorderStdHeader.h"
#include "TrackHeadDataRecord.h"
#include "GlyphDetector/GlyphModel.h"
#include "global_defines.h"

namespace videre
{


    class HeadTrackingProcessor : public VisionProcessorAbstract
    {
    private:

        CudaImageProcLibsTrackHeadNS::GlyphModel _headModelData;

        CudaImageProcLibsTrackHeadNS::TrackHeadProcess _trackHeadProcess;

        std::shared_ptr<TrackHeadOrientationMessage> _trackHeadOrientationMsg;

        std::shared_ptr<HeadTrackingControlMessage> _headTrackingControlMsg;

        std::shared_ptr<HeadOrientationControlMessage> _headOrientationControlMsg;

        std::string _htParamsFilename = "HeadTrackingConfig.ini";

        std::string _glyphModelDirectoryName;
        int _NoOfGlyphModels = 2;
        int _selectedGlyphModel = 0;

        std::string _glyphModelFileNames[MAXNUMBERGLYPHMODELS];
        int _glyphScaleVals[MAXNUMBERGLYPHMODELS];


        DataRecorder _dataRecorder;
        DataRecorderStdHeader _dataRecorderStdHeader;
        TrackHeadDataRecord _trackHeadDataRecord;
        bool EnableTrackHeadLogging = false;

        bool SendTrackHeadDataOut = false;

    public:



        HeadTrackingProcessor(VisionProcessResources* vpResources);

        ~HeadTrackingProcessor();

        //Intialize any resources at the start of operation.
        virtual void Intialize() final;


        //Reset the Vision Processor.
        //Reset must return all resources obtained during operation...
        //such a image messages, target result messages and the like.
        virtual void Reset();

        std::string getGlyphModelFilename(int modelIdx);

        //This is the primary call for running the Vision Processor
        //each time through the Manager's loop.
        virtual VPProcessResult_e ExecuteUnitOfWork();


        bool PostProcessImageToBeSentOut(ImagePlusMetadataMessage *inputMsg,
                                         ImagePlusMetadataMessage *outputMsg);


    };

}
#endif //VIDERE_DEV_HEADTRACKINGPROCESSOR_H
