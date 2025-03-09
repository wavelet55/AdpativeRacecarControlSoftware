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

#include "NullVisionProcessor.h"
#include "Utilities/ImageMetadataWriter.h"

namespace videre
{

    NullVisionProcessor::NullVisionProcessor(VisionProcessResources* vpResources)
        : VisionProcessorAbstract(vpResources)
    {
        _imgTxtWriter.SetTextScreenLocation(true, true, true, 640, 480);
    }

    NullVisionProcessor::~NullVisionProcessor()
    {

    }

    //Reset the Vision Processor.
    //Reset must return all resources obtained during operation...
    //such a image messages, target result messages and the like.
    void NullVisionProcessor::Reset()
    {
        _startTimeStampSec = 0;
        _startImageNumber = 0;
    }

    //This is the primary call for running the Vision Processor
    //each time through the Manager's loop.
    VPProcessResult_e NullVisionProcessor::ExecuteUnitOfWork()
    {
        VPProcessResult_e result = VPProcessResult_e::VPPR_none;
        ImagePlusMetadataMessage *imgMsgPtr = nullptr;
        ImagePlusMetadataMessage *imgPMetadataMsg2Ptr = nullptr;
        ImageProcTargetInfoResultsMessage *tgtMsgPtr;
        Rabit::RabitMessage *tmpMsgPtr;

        VPRptr->StreamRecordControlMsg->FetchMessage();

        VPRptr->AddEmptyImageMsgsToQueue();
        VPRptr->CheckForEmptyTargetMessages();

        //Get the message even if it is not going to be processed... it will be
        //effectively thrown away so that when processing is enabled... we will have
        //fresh images.
        imgMsgPtr = VPRptr->GetNextIncomingImagePlusMetadataMessage();
        if(imgMsgPtr != nullptr)
        {
            if( _startTimeStampSec <= 0)
            {
                _startTimeStampSec = imgMsgPtr->ImageCaptureTimeStampSec;
                _startImageNumber = imgMsgPtr->ImageNumber;
                _imgTxtWriter.SetTextScreenLocation(true, true, true,
                                                    imgMsgPtr->ImageFrame.cols,
                                                    imgMsgPtr->ImageFrame.rows);
            }

            if (VPRptr->ImageProcessControlMsg->TargetImageProcessingEnabled)
            {
                tgtMsgPtr = nullptr;
                try
                {
                    if(VPRptr->PostProcessImages)
                    {
                        double delTS = imgMsgPtr->ImageCaptureTimeStampSec - _startTimeStampSec;
                        unsigned int delImgNo = imgMsgPtr->ImageNumber - _startImageNumber;
                        _imgTxtWriter.WriteImageNoAndTimeStamp(imgMsgPtr->ImageFrame, delImgNo, delTS);

                        //Send a copy of the Image Plus Metadata to the Stream Record Manager.
                        if (VPRptr->StreamRecordEmptyIPMDMsgQueue->GetMessage(tmpMsgPtr))
                        {
                            imgPMetadataMsg2Ptr = static_cast<ImagePlusMetadataMessage *>(tmpMsgPtr);
                            imgPMetadataMsg2Ptr->CopyMessage(imgMsgPtr);
                            //Send the message back to the Stream Manager even if it is bad,
                            //otherwise we will run out of the messages.
                            VPRptr->StreamRecordRxIPMDMsgQueue->AddMessage(imgPMetadataMsg2Ptr);
                        }
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



}