/* ****************************************************************
 * Standard Target Detector Vision Processor
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Feb. 2017
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
  *******************************************************************/

#include "StdTargetDectectorVisionProcessor.h"

namespace videre
{

    StdTargetDectectorVisionProcessor::StdTargetDectectorVisionProcessor(VisionProcessResources* vpResources)
        : VisionProcessorAbstract(vpResources),
          _targetDetectorProcessControl(vpResources->GetMgrPtr(), vpResources->GetConfig())
    {

    }

    StdTargetDectectorVisionProcessor::~StdTargetDectectorVisionProcessor()
    {

    }

    //Reset the Vision Processor.
    //Reset must return all resources obtained during operation...
    //such a image messages, target result messages and the like.
    void StdTargetDectectorVisionProcessor::Reset()
    {

    }

    //This is the primary call for running the Vision Processor
    //each time through the Manager's loop.
    VPProcessResult_e StdTargetDectectorVisionProcessor::ExecuteUnitOfWork()
    {
        VPProcessResult_e result = VPProcessResult_e::VPPR_none;
        ImagePlusMetadataMessage *imgMsgPtr = nullptr;
        ImagePlusMetadataMessage *imgPMetadataMsg2Ptr = nullptr;
        ImageProcTargetInfoResultsMessage *tgtMsgPtr;
        Rabit::RabitMessage *tmpMsgPtr;
        VPRptr->StreamRecordControlMsg->FetchMessage();

        VPRptr->AddEmptyImageMsgsToQueue();
        VPRptr->CheckForEmptyTargetMessages();

        if(VPRptr->ImageProcessControlMsgChanged)
        {
            //Release resources of the current target detector before switching to a new one.
            _targetDetectorProcessControl.SetTargetProcessingMode(VPRptr->ImageProcessControlMsg->TargetProcessingMode,
                                                                  VPRptr->ImageProcessControlMsg->GPUProcessingEnabled);

            VPRptr->SetTargetProcesModeInImageProcessControlStatusMsg(_targetDetectorProcessControl.GetActiveTargetProcessingMode());
            VPRptr->SetGPUProcessingEnabledInImageProcessControlStatusMsg(_targetDetectorProcessControl.IsGPUAcceleratedAlgorithm());
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
                    //Build a Target Message and send it out.
                    tgtMsgPtr = VPRptr->CheckoutImageProcTargetInfoResultsMessageFromPool();
                    if( tgtMsgPtr != nullptr)
                    {
                        tgtMsgPtr->Clear();
                        tgtMsgPtr->SetImageMetaDataInfo(imgMsgPtr);

                        if (VPRptr->ImageProcessControlMsg->TargetProcessingMode !=
                                                        TargetProcessingMode_e::TgtProcMode_None)
                        {
                            _targetDetectorProcessControl.GetActiveTargetDetector()->CheckUpdateTargetParameters();

                            _targetDetectorProcessControl.GetActiveTargetDetector()->DetectTargets(imgMsgPtr,
                                                                                                   tgtMsgPtr,
                                                                                                   imgMsgPtr->PixelToRWTranslator);
                        }

                        //Send the Target Results out... even if no targets are found.
                        VPRptr->TgtMsgsToBeSentQueue->AddMessage(tgtMsgPtr);

                        if(VPRptr->PostProcessImages)
                        {
                            //Send a copy of the Image Plus Metadata to the Stream Record Manager.
                            if (VPRptr->StreamRecordEmptyIPMDMsgQueue->GetMessage(tmpMsgPtr))
                            {
                                imgPMetadataMsg2Ptr = static_cast<ImagePlusMetadataMessage *>(tmpMsgPtr);
                                PostProcessImageToBeSentOut(imgMsgPtr, imgPMetadataMsg2Ptr);
                                //Send the message back to the Stream Manager even if it is bad,
                                //otherwise we will run out of the messages.
                                VPRptr->StreamRecordRxIPMDMsgQueue->AddMessage(imgPMetadataMsg2Ptr);
                            }
                        }
                    }
                    else
                    {
                        LOGERROR("The TargetInfoResultsMsgPool is Empty");
                        cout << "The TargetInfoResultsMsgPool is Empty" << std::endl;
                    }

                    if (imgMsgPtr->ImageNumber % 25 == 0)
                    {
                        LOGINFO("VisionProcess Image Number: " << imgMsgPtr->ImageNumber);
                        cout << "VisionProcess Image Number: " << imgMsgPtr->ImageNumber << std::endl;
                    }
                }
                catch (std::exception &e)
                {
                    LOGERROR("VisionProcessManager: Exception: " << e.what());
                    VPRptr->CheckinImageProcTargetInfoResultsMessageToPool(tgtMsgPtr);
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


    bool StdTargetDectectorVisionProcessor::PostProcessImageToBeSentOut(ImagePlusMetadataMessage *inputMsg,
                                                                        ImagePlusMetadataMessage *outputMsg)
    {
        bool error = true;
        if (inputMsg != nullptr && outputMsg != nullptr)
        {
            try
            {
                VPRptr->FeatureMatchProcCtrlMsg->FetchMessage();
                if ((int)VPRptr->FeatureMatchProcCtrlMsg->FMImagePostProcessMethod == 1)
                {
                    outputMsg->CopyMessageWithoutImage(inputMsg);
                    //Modify the image with the feature locations.
                    bool gotImg = _targetDetectorProcessControl.GetActiveTargetDetector()->GetIntermediateImage(0, outputMsg->ImageFrame);
                    if(gotImg)
                    {
                        _targetDetectorProcessControl.GetActiveTargetDetector()->MarkTargetsOnImage(outputMsg->ImageFrame, false);
                    }
                    else
                    {
                        inputMsg->ImageFrame.copyTo(outputMsg->ImageFrame);
                    }
                }
                else if ((int)VPRptr->FeatureMatchProcCtrlMsg->FMImagePostProcessMethod == 2 )
                {
                    outputMsg->CopyMessage(inputMsg);
                    _targetDetectorProcessControl.GetActiveTargetDetector()->MarkTargetsOnImage(outputMsg->ImageFrame, true);
                }
                else if ((int)VPRptr->FeatureMatchProcCtrlMsg->FMImagePostProcessMethod == 3 )
                {
                    outputMsg->CopyMessage(inputMsg);
                    _targetDetectorProcessControl.GetActiveTargetDetector()->MarkTargetsOnImage(outputMsg->ImageFrame, false);
                }
                else
                {
                    //No extra processing
                    outputMsg->CopyMessage(inputMsg);
                }
                error = false;
           }
            catch (std::exception &e)
            {
                LOGERROR("PostProcessImageToBeSentOut: Exception: " << e.what());
                error = true;
            }
        }
        return error;
    }

}