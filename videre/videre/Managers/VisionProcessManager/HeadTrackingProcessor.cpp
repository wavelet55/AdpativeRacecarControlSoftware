/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: July. 2018
 *
 *******************************************************************/

#include "HeadTrackingProcessor.h"
#include "Utilities/HeadTrackingCalParamsReaderWriter.h"

using namespace CudaImageProcLibsTrackHeadNS;
using namespace std;
using namespace ImageProcLibsNS;

namespace videre
{

    HeadTrackingProcessor::HeadTrackingProcessor(VisionProcessResources* vpResources)
            : VisionProcessorAbstract(vpResources), _trackHeadProcess(), _dataRecorder(),
              _dataRecorderStdHeader("Image Based Track Head Data  Log", 0), _trackHeadDataRecord()
    {

        _trackHeadOrientationMsg = make_shared<TrackHeadOrientationMessage>();
        VPRptr->GetMgrPtr()->AddPublishSubscribeMessage("TrackHeadOrientationMessage", _trackHeadOrientationMsg);

        _headTrackingControlMsg = make_shared<HeadTrackingControlMessage>();
        VPRptr->GetMgrPtr()->AddPublishSubscribeMessage("HeadTrackingControlMessage", _headTrackingControlMsg);

        _headOrientationControlMsg = make_shared<HeadOrientationControlMessage>();
        VPRptr->GetMgrPtr()->AddPublishSubscribeMessage("HeadOrientationControlMessage", _headOrientationControlMsg);

        _dataRecorder.setDirectory(DataLogDirectory);
        string fn = VPRptr->GetConfig()->GetConfigStringValue("HeadTrackingParameters.DataLogBaseFilename", "TrackHeadDataLog");
        _dataRecorder.setBaseFilename(fn);
        _dataRecorder.setHeaderRecord(&_dataRecorderStdHeader);

        _trackHeadDataRecord.TrackHeadOrientationMsg = _trackHeadOrientationMsg;
        EnableTrackHeadLogging = VPRptr->GetConfig()->GetConfigBoolValue("HeadTrackingParameters.EnableTrackHeadLogging", true);
        SendTrackHeadDataOut = VPRptr->GetConfig()->GetConfigBoolValue("HeadTrackingParameters.SendTrackHeadDataOut", false);

        _glyphModelDirectoryName = VPRptr->GetConfig()->GetConfigStringValue("HeadTrackingParameters.GlyphModelsDirectory", "GlyphModels");
        _NoOfGlyphModels = VPRptr->GetConfig()->GetConfigIntValue("HeadTrackingParameters.NumberOfGlyphModels", 1);
        _NoOfGlyphModels = _NoOfGlyphModels < 1 ? 1 : _NoOfGlyphModels > MAXNUMBERGLYPHMODELS ? MAXNUMBERGLYPHMODELS : _NoOfGlyphModels;

        _selectedGlyphModel = 0;
        for(int i = 0; i < _NoOfGlyphModels; i++)
        {
            std::ostringstream cfgParm;

            cfgParm.str(std::string());  //Clear the string stream
            cfgParm << "HeadTrackingParameters.GlyphModelFilename_" << (i + 1);
            _glyphModelFileNames[i] = VPRptr->GetConfig()->GetConfigStringValue(cfgParm.str(), "GlyphModelXLHelmet.glf");

            cfgParm.str(std::string());  //Clear the string stream
            cfgParm << "HeadTrackingParameters.GlyphScale_" << (i + 1);
            _glyphScaleVals[i] = VPRptr->GetConfig()->GetConfigIntValue(cfgParm.str(), 30);
        }

    }

    HeadTrackingProcessor::~HeadTrackingProcessor()
    {
        _trackHeadProcess.Close();
    }

    std::string HeadTrackingProcessor::getGlyphModelFilename(int modelIdx)
    {
        if( modelIdx < 0 || modelIdx >= _NoOfGlyphModels)
            modelIdx = 0;

        boost::filesystem::path filename(_glyphModelDirectoryName);
        filename /= _glyphModelFileNames[modelIdx];
        return filename.string();
    }

    void HeadTrackingProcessor::Intialize()
    {
        //ReadHeadModelFromConfig(VPRptr->GetConfig(), _headModelData);

        _htParamsFilename = VPRptr->GetConfig()->GetConfigStringValue("HeadTrackingParameters.HTParametersFilename", "HeadTrackingConfig.ini");

        std::string htParamsFilename = VPRptr->GetConfig()->GetConfigStringValue("HeadTrackingParameters.HTParametersFilename", "HeadTrackingConfig.ini");
        _headTrackingControlMsg->HeadTrackingParameters.SetDefaults();
        //If there is a Config file... read parameters from the config file
        try
        {
            ReadHeadTrackingParametersFromIniFile(htParamsFilename, _headTrackingControlMsg->HeadTrackingParameters);

            string glyphFilename = getGlyphModelFilename(_selectedGlyphModel);
            int scale = _glyphScaleVals[_selectedGlyphModel];
            if( !_headModelData.load(glyphFilename, scale) )
            {
                LOGERROR("HeadTrackingProcessor Error loading Glyph Model file: " << glyphFilename);
            }

        }
        catch (exception &e)
        {
            LOGWARN("Could not read model date for target")
            ReadHeadTrackingParametersFromConfig(VPRptr->GetConfig(), _headTrackingControlMsg->HeadTrackingParameters);
        }



        _headTrackingControlMsg->HeadTrackingImageDisplayType = HeadTrackingImageDisplayType_e::HTID_HeadOrientationVector;
        _headTrackingControlMsg->PostMessage();  //Make sure the rest of videre has these values if needed.


        if(_trackHeadProcess.Initialize(VPRptr->CameraCalData, _headModelData,
                                        _headTrackingControlMsg->HeadTrackingParameters))
        {
            LOGERROR("Track Head Process Initialize Error.")
        }
        _trackHeadProcess.SetHeadTrackingImageDisplayType(_headTrackingControlMsg->HeadTrackingImageDisplayType);
        _trackHeadProcess.SetUseGPU(false);
    }


    //Reset the Vision Processor.
    //Reset must return all resources obtained during operation...
    //such a image messages, target result messages and the like.
    void HeadTrackingProcessor::Reset()
    {

        _trackHeadProcess.Close();
    }

    //This is the primary call for running the Vision Processor
    //each time through the Manager's loop.
    VPProcessResult_e HeadTrackingProcessor::ExecuteUnitOfWork()
    {
        VPProcessResult_e result = VPProcessResult_e::VPPR_none;
        ImagePlusMetadataMessage *imgMsgPtr = nullptr;
        ImagePlusMetadataMessage *imgPMetadataMsg2Ptr = nullptr;
        ImageProcTargetInfoResultsMessage *tgtMsgPtr;
        Rabit::RabitMessage *tmpMsgPtr;
        TrackHeadOrientationData_t headOrientationData;

        VPRptr->StreamRecordControlMsg->FetchMessage();
        VPRptr->LoggingControlMsg->FetchMessage();

        VPRptr->AddEmptyImageMsgsToQueue();
        VPRptr->CheckForEmptyTargetMessages();

        if(VPRptr->ImageProcessControlMsgChanged)
        {
            VPRptr->ImageProcessControlMsg->FetchMessage();

            //The GPU Process does not work as well ... so force this to be false.
            //_trackHeadProcess.SetUseGPU(VPRptr->ImageProcessControlMsg->GPUProcessingEnabled);
            _trackHeadProcess.SetUseGPU(false);
        }

        if(_headTrackingControlMsg->FetchMessage())
        {
            VPRptr->ImageProcessControlMsg->FetchMessage();
            _trackHeadProcess.SetHeadTrackingParameters(_headTrackingControlMsg->HeadTrackingParameters);
            _trackHeadProcess.SetHeadTrackingImageDisplayType(_headTrackingControlMsg->HeadTrackingImageDisplayType);

            //The GPU Process does not work as well ... so force this to be false.
            //_trackHeadProcess.SetUseGPU(VPRptr->ImageProcessControlMsg->GPUProcessingEnabled);
            _trackHeadProcess.SetUseGPU(false);

            try
            {
                WriteHeadTrackingParametersToIniFile(_htParamsFilename,
                                                     _headTrackingControlMsg->HeadTrackingParameters);
            }
            catch (exception &e)
            {
                LOGWARN("Could not write HeadTrackingConfig.ini Config")
                ReadHeadTrackingParametersFromConfig(VPRptr->GetConfig(), _headTrackingControlMsg->HeadTrackingParameters);
            }


            if( _headTrackingControlMsg->GlyphModelIndex != _selectedGlyphModel)
            {
                int gmIdx = _headTrackingControlMsg->GlyphModelIndex;
                gmIdx = gmIdx >= _NoOfGlyphModels - 1 ? _NoOfGlyphModels - 1 : gmIdx;
                gmIdx = gmIdx < 0 ? 0 : gmIdx;
                string glyphFilename = getGlyphModelFilename(gmIdx);
                int scale = _glyphScaleVals[gmIdx];
                if( !_headModelData.load(glyphFilename, scale) )
                {
                    LOGERROR("HeadTrackingProcessor Error loading Glyph Model file: " << glyphFilename);
                }
                _selectedGlyphModel = gmIdx;
                _headTrackingControlMsg->GlyphModelIndex = gmIdx;
            }
        }

        //Get the message even if it is not going to be processed... it will be
        //effectively thrown away so that when processing is enabled... we will have
        //fresh images.
        imgMsgPtr = VPRptr->GetNextIncomingImagePlusMetadataMessage();
        if(imgMsgPtr != nullptr)
        {

            if (VPRptr->ImageProcessControlMsg->TargetImageProcessingEnabled)
            {
                tgtMsgPtr = nullptr;
                try
                {
                    _trackHeadOrientationMsg->TrackHeadOrientationData = _trackHeadProcess.TrackHeadPostion(imgMsgPtr->ImageFrame);
                    _trackHeadOrientationMsg->ImageNumber = imgMsgPtr->ImageNumber;
                    _trackHeadOrientationMsg->ImageCaptureTimeStampSec = imgMsgPtr->ImageCaptureTimeStampSec;
                    _trackHeadOrientationMsg->PostMessage();

                    _headOrientationControlMsg->FetchMessage();
                    if( SendTrackHeadDataOut
                        && _headOrientationControlMsg->HeadOrientationOutputSelect == HeadOrientationOutputSelect_e::ImageProcTrackHead)
                    {
                        std::shared_ptr<TrackHeadOrientationMessage> htOutMsg;
                        htOutMsg = std::make_shared<TrackHeadOrientationMessage>();
                        htOutMsg->CopyMessage(_trackHeadOrientationMsg.get());
                        auto rmsgPtr = dynamic_pointer_cast<Rabit::RabitMessage, TrackHeadOrientationMessage>(htOutMsg);
                        VPRptr->GetMgrPtr()->AddMessageToQueue("ZMQPublishOutMsgQueue", rmsgPtr);
                    }

                    if( EnableTrackHeadLogging && VPRptr->LoggingControlMsg->EnableLogging)
                    {
                        _dataRecorder.writeDataRecord(_trackHeadDataRecord);
                    }
                    else if(!VPRptr->LoggingControlMsg->EnableLogging)
                    {
                        _dataRecorder.closeLogFile();
                    }

                    if(VPRptr->PostProcessImages)
                    {

                        //Send a copy of the Image Plus Metadata to the Stream Record Manager.
                        if (VPRptr->StreamRecordEmptyIPMDMsgQueue->GetMessage(tmpMsgPtr))
                        {
                            imgPMetadataMsg2Ptr = static_cast<ImagePlusMetadataMessage *>(tmpMsgPtr);
                            if(PostProcessImageToBeSentOut(imgMsgPtr, imgPMetadataMsg2Ptr))
                            {
                                imgPMetadataMsg2Ptr->CopyMessage(imgMsgPtr);
                            }
                            //Send the message back to the Stream Manager even if it is bad,
                            //otherwise we will run out of the messages.
                            VPRptr->StreamRecordRxIPMDMsgQueue->AddMessage(imgPMetadataMsg2Ptr);
                        }
                    }
                    if (imgMsgPtr->ImageNumber % 100 == 0)
                    {
                        LOGINFO("HeadTrackingProcessor Image Number: " << imgMsgPtr->ImageNumber);
                    }
                }
                catch (std::exception &e)
                {
                    LOGERROR("HeadTrackingProcessor: Exception: " << e.what());
                }
                //Add our spent Image Message back on the Empty Queue
                //for Image Capture Manager to Re-populate
                VPRptr->EmptyImagePlusMetadataQueue->AddMessage(imgMsgPtr);
            }
            else
            {
                //Now return our ImagePlusMetadataMessage back to the pool to be used again.
                VPRptr->CheckinImagePlusMetadataMessageToPool(imgMsgPtr);
            }
        }
        return result;
    }


    bool HeadTrackingProcessor::PostProcessImageToBeSentOut(ImagePlusMetadataMessage *inputMsg,
                                                             ImagePlusMetadataMessage *outputMsg)
    {
        bool error = true;
        cv::Mat outpImage;
        if (inputMsg != nullptr && outputMsg != nullptr)
        {
            try
            {
                _headTrackingControlMsg->FetchMessage();
                switch(_headTrackingControlMsg->HeadTrackingImageDisplayType)
                {
                    case HeadTrackingImageDisplayType_e::HTID_None:
                        outputMsg->CopyMessage(inputMsg);
                        break;

                    case HeadTrackingImageDisplayType_e::HTID_HighLightGlyphs:
                        if( _trackHeadProcess.GetIntermediateImage(0, outputMsg->ImageFrame))
                        {
                            outputMsg->CopyMessageWithoutImage(inputMsg);
                        }
                        else
                        {
                            outputMsg->CopyMessage(inputMsg);
                        }
                        break;
                    case HeadTrackingImageDisplayType_e::HTID_HeadOrientationVector:
                        if( _trackHeadProcess.GetIntermediateImage(1, outputMsg->ImageFrame))
                        {
                            outputMsg->CopyMessageWithoutImage(inputMsg);
                        }
                        else
                        {
                            outputMsg->CopyMessage(inputMsg);
                        }
                        break;
                }
                error = false;
            }
            catch (std::exception &e)
            {
                LOGERROR("HeadTrackingProcessor:PostProcessImageToBeSentOut: Exception: " << e.what());
                error = true;
            }
        }
        return error;
    }




}