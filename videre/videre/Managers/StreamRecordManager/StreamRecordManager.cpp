/* ****************************************************************
 * Stream Record Manager
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Randy Direen PhD
 * Date: May 2016
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
  *******************************************************************/

#include <image_plus_metadata_message.h>
#include <CompressedImageMessage.h>
#include "StreamRecordManager.h"
#include "RecorderPlayer/ImagePlusMetadataFileHeaders.h"

using namespace std;
using namespace cv;
using namespace Rabit;
using namespace VidereImageprocessing;

namespace videre
{
    //The Stream Record Manager is responsible for compressing and sending images out to the
    //ground station or elsewhere.  The mananager is also responsible for saving/longing to file(s)
    //the images plus metadata.
    //Images and metadata my come from the Image Capture Manager (Pre-Process) or from the
    //Vision Process Manager (Post-Process).  Typically only one of the two methods will be used:
    //Pre or Post Processed images.
    //For Pre-Processing, the Stream Record Manager contains a Pool of empty images messages.
    //The StreamRecordManager sends empty images messages to the Image Capture Manager.  The
    //Image Capture Manager fills in images and metadata and sends the message back to the
    //Stream Record Manager.  The Stream Record Manager then compresses and sends the image out
    //and logs the image plus metadata per the system handling instructions contained in the
    //StreamRecordImageControlMesssage parameter settings.

    StreamRecordManager::StreamRecordManager(std::string name, std::shared_ptr<ConfigData> config)
    : StreamRecordManagerWSRMgr(name), _imagePlusMetadataRecorder(config, true)
    {
        this->SetWakeupTimeDelayMSec(250);
        _config_sptr = config;

        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        //Messages
        _streamRecordControlMsg_sptr = std::make_shared<StreamRecordImageControlMesssage>();
        this->AddPublishSubscribeMessage("StreamRecordImageControlMesssage", _streamRecordControlMsg_sptr);
        _streamRecordControlMsg_sptr->Register_SomethingPublished(boost::bind(&StreamRecordManager::WakeUpManagerEH, this));

        _imageLoggingControlMsg = std::make_shared<ImageLoggingControlMessage>();
        this->AddPublishSubscribeMessage("ImageLoggingControlMessage", _imageLoggingControlMsg);
        _imageLoggingControlMsg->Register_SomethingPublished(boost::bind(&StreamRecordManager::WakeUpManagerEH, this));

        //Queues
        _StreamRecordRxIPMDMsgQueue = std::make_shared<RabitMsgPtrSPSCQueue>(
                ImagePoolSize,
                "StreamRecordRxIPMDMsgQueue");
        this->AddManagerMessageQueue(_StreamRecordRxIPMDMsgQueue->GetMessageQueueName(),
                                     _StreamRecordRxIPMDMsgQueue);

        //Set event that will wakeup the loop when we receive a new Image.
        this->WakeUpManagerOnEnqueue(_StreamRecordRxIPMDMsgQueue);

        _StreamRecordEmptyIPMDMsgQueue = std::make_shared<RabitMsgPtrSPSCQueue>(
                ImagePoolSize,
                "StreamRecordEmptyIPMDMsgQueue");
        this->AddManagerMessageQueue(_StreamRecordEmptyIPMDMsgQueue->GetMessageQueueName(),
                                     _StreamRecordEmptyIPMDMsgQueue);


        _imageStreamMsgQueue_sptr = std::make_shared<RabitMsgPtrSPSCQueue>(
                                                ImagePoolSize,
                                                "ImageStreamMsgQueue");

        this->AddManagerMessageQueue(_imageStreamMsgQueue_sptr->GetMessageQueueName(),
                                     _imageStreamMsgQueue_sptr);

        _imageStreamEmptyMsgQueue_sptr = std::make_shared<RabitMsgPtrSPSCQueue>(
                                    ImagePoolSize,
                                    "ImageStreamEmptyMsgQueue");
        this->AddManagerMessageQueue(_imageStreamEmptyMsgQueue_sptr->GetMessageQueueName(),
                                     _imageStreamEmptyMsgQueue_sptr);


        //Setup the Image Plus Metadata Message Pool.
        //The Stream Record Manager sends empty messages from this message pool to
        //the Image Capture Manager, the Image Capture Manager fills in the image and
        //metadata and sends the message back to this manager for processing.
        //We temporaraly need to create a ImagePlusMetaData Message required to setup the pool.
        //The message is only needed for the setup process and can then be discarded.
        ImagePlusMetadataMessage ipmMsg;
        _imageMessagePool_uptr = unique_ptr<MessagePool>(new MessagePool(&ipmMsg, ImagePoolSize));

        //Setup a Compressed Image Message Pool.
        //Compressed image messages are send to the Comms manager for transport,
        //and the Comms Manager sends the empty messages back for re-use.
        CompressedImageMessage tmpImgMsg;
        _compressedImageMessagePool_uptr = unique_ptr<MessagePool>(new MessagePool(&tmpImgMsg, ImagePoolSize));

        _videoRecorder_uptr = unique_ptr<VisionRecordPlay>(new VisionRecordPlay());

    }

    void StreamRecordManager::Initialize()
    {

        LOGINFO("StreamRecordManager: Initialization Started")
        _streamRecordControlMsg_sptr->Clear();
        _imageLoggingControlMsg->Clear();

        _record_directory = _config_sptr->GetConfigStringValue("video_recording.directory", "recorded_video");
        _record_base_name = _config_sptr->GetConfigStringValue("video_recording.base_name", "ImagePlusMetadata");
        _recordVideo_base_name = _config_sptr->GetConfigStringValue("video_recording.VideoBaseName", "video");
        _record_fps = _config_sptr->GetConfigDoubleValue("video_recording.fps", 15);
        _videoRecordingEnabled = _config_sptr->GetConfigBoolValue("video_recording.VideoRecordingEnabled", false);

        int width = _config_sptr->GetConfigIntValue("opencv.width", 640);
        int height = _config_sptr->GetConfigIntValue("opencv.height", 480);

        bool logCompressedImages = _config_sptr->GetConfigBoolValue("video_recording.CompressImages", true);
        _streamRecordControlMsg_sptr->SetImageCompressionQuality(_config_sptr->GetConfigIntValue("video_stream.quality", 50));
        _streamRecordControlMsg_sptr->SetStreamImageFrameRate( _config_sptr->GetConfigDoubleValue("video_stream.fps", 5) );
        _streamRecordControlMsg_sptr->SetImageScaleDownFactor( _config_sptr->GetConfigDoubleValue("video_stream.ScaleDownFactor", 1.0) );
        _streamRecordControlMsg_sptr->StreamImagesEnabled = _config_sptr->GetConfigBoolValue("video_stream.StreamImagesEnabled", true);
        _streamRecordControlMsg_sptr->RecordImagesEnabled = false;
        _streamRecordControlMsg_sptr->RecordCompressedImages = logCompressedImages;
        _streamRecordControlMsg_sptr->PostMessage();

        if(logCompressedImages)
        {
            _imageLoggingControlMsg->VisionLoggingType = VisionLoggingType_e::LogCompressedImages;
        }
        else
        {
            _imageLoggingControlMsg->VisionLoggingType = VisionLoggingType_e::LogRawImages;
        }
        _imageLoggingControlMsg->PostMessage();

        _imageStreamTxStopwatch.reset();
        _imageStreamTxStopwatch.start();
        _minTimeSecBetweenStreamImages = 1.0 / 30.0;
        if(_streamRecordControlMsg_sptr->StreamImageFrameRate > 0.01)
        {
            _minTimeSecBetweenStreamImages = 1.0 / _streamRecordControlMsg_sptr->StreamImageFrameRate;
            _minTimeSecBetweenStreamImages = 0.90 * _minTimeSecBetweenStreamImages;
        }

        _stream_quality = _streamRecordControlMsg_sptr->ImageCompressionQuality;

        _videoRecorder_uptr->Init(_record_directory,
                                  _recordVideo_base_name,
                           _record_fps,
                           width,
                           height);

        _videoRecordStartTime = 0;

        //_videoRecorder_uptr->SetCodec_MJPG();
        _videoRecorder_uptr->SetVideoCodec(_config_sptr->GetConfigStringValue("video_recording.codec", "MJPEG"));

        _compress_params.push_back(cv::IMWRITE_JPEG_QUALITY);
        _compress_params.push_back(_stream_quality);

        LOGINFO("StreamRecordManager: Initialization Complete")
        std::cout << "StreamRecordManager: Initialization Complete" << std::endl;
    }

    void StreamRecordManager::AddEmptyImageMsgsToQueue()
    {
        if ((_imageLoggingControlMsg->EnableLogging || _streamRecordControlMsg_sptr->StreamImagesEnabled)
            && (_imageMessagePool_uptr->GetNumberOfFreeMessages() > 0))
        {
            //Add a empty message to the queue
            //The message queue was created large enough to handle the size of the
            //Image Message Pool, so we don't need to check to see if the queue is full.
            RabitMessage *emptyMsgPtr = _imageMessagePool_uptr->CheckOutMessage();
            if (emptyMsgPtr != nullptr)
            {
                if (!_StreamRecordEmptyIPMDMsgQueue->AddMessage(emptyMsgPtr))
                {
                    //This should not happen.
                    LOGWARN("StreamRecordManager: EmptyImagePlusMetadataQueue is Full!")
                    _imageMessagePool_uptr->CheckInMessage(emptyMsgPtr);
                }
            }
        }
    }

    //Get any empty Image Messages from the Empty Image Msg Queue and
    //add them back to the Pool.
    void StreamRecordManager::CheckForEmptyImageMessages()
    {
        RabitMessage *emptyImgMsg;
        while( _imageStreamEmptyMsgQueue_sptr->GetMessage(emptyImgMsg))
        {
            _compressedImageMessagePool_uptr->CheckInMessage(emptyImgMsg);
        }
    }

    void StreamRecordManager::ExecuteUnitOfWork()
    {
        Rabit::RabitMessage *tmpMsgPtr = nullptr;
        CompressedImageMessage *cImgMsgPtr = nullptr;
        bool imageCompressedOk = false;
        bool cImgMsgSentOut = false;

        int frameWidth = 0;
        int frameHeight = 0;

        bool ctrlmsgChanged = _imageLoggingControlMsg->FetchMessage();
        ctrlmsgChanged |= _streamRecordControlMsg_sptr->FetchMessage();
        if( ctrlmsgChanged )
        {
            if(_streamRecordControlMsg_sptr->StreamImageFrameRate > 0.01)
            {
                _minTimeSecBetweenStreamImages = 1.0 / _streamRecordControlMsg_sptr->StreamImageFrameRate;
                _minTimeSecBetweenStreamImages = 0.90 * _minTimeSecBetweenStreamImages;
            }

            if(!_streamRecordControlMsg_sptr->RecordImagesEnabled || !_imageLoggingControlMsg->EnableLogging)
            {
                if(_videoRecorder_uptr->IsOpen()) {
                    _videoRecorder_uptr->ClearVideoWriter();
                    _videoRecordStartTime = 0;
                    _videoRecordStartImageNumber = 0;
                }
            }

            _stream_quality = _streamRecordControlMsg_sptr->ImageCompressionQuality;
            _compress_params.clear();
            _compress_params.push_back(cv::IMWRITE_JPEG_QUALITY);
            _compress_params.push_back(_stream_quality);
        }

        AddEmptyImageMsgsToQueue();
        CheckForEmptyImageMessages();

        //Get the message even if it is not going to be processed... it will be
        //effectively thrown away so that when processing is enabled... we will have
        //fresh images.
        if (_StreamRecordRxIPMDMsgQueue->GetMessage(tmpMsgPtr))
        {
            ImagePlusMetadataMessage *imgMsgPtr = static_cast<ImagePlusMetadataMessage *>(tmpMsgPtr);
            if (imgMsgPtr->ImageNumber != 0
                && imgMsgPtr->ImageFrame.data != NULL
                && imgMsgPtr->ImageFrame.cols > 0
                && imgMsgPtr->ImageFrame.rows > 0
                && (_streamRecordControlMsg_sptr->StreamImagesEnabled || _imageLoggingControlMsg->EnableLogging ))
            {
                bool txStreamImage = false;
                if (_streamRecordControlMsg_sptr->StreamImagesEnabled)
                {
                    _imageStreamTxStopwatch.captureTime();
                    double delayTimeSec = _imageStreamTxStopwatch.getTimeElapsed();
                    if (delayTimeSec > _minTimeSecBetweenStreamImages
                            || imgMsgPtr->ForceTxImage )
                    {
                        txStreamImage = true;
                        _imageStreamTxStopwatch.reset();
                        _imageStreamTxStopwatch.start();
                        imgMsgPtr->ForceTxImage = false;
                    }
                }
                //Check to see if we need to compress the image
                imageCompressedOk = false;
                if (txStreamImage
                    || (_imageLoggingControlMsg->EnableLogging
                        && _imageLoggingControlMsg->VisionLoggingType == VisionLoggingType_e::LogCompressedImages))
                {
                    //Get the next available compressed image buffer.
                    cImgMsgPtr = static_cast<CompressedImageMessage *>(_compressedImageMessagePool_uptr->CheckOutMessage());
                    if (cImgMsgPtr != nullptr)
                    {
                        try
                        {
                            cv::Mat *txImage = &imgMsgPtr->ImageFrame;
                            if (_streamRecordControlMsg_sptr->StreamImageScaleDownFactor > 1.1)
                            {
                                double sf = 1.0 / _streamRecordControlMsg_sptr->StreamImageScaleDownFactor;
                                //Check to see if the image has to be scaled down.
                                Size dsize(0, 0);
                                cv::resize(imgMsgPtr->ImageFrame, _resizedImage, dsize, sf, sf);
                                txImage = &_resizedImage;
                            }

                            _imageStreamDataQueueFullMsgSent = false;
                            cImgMsgPtr->ImageFormatType = ImageFormatType_e::ImgFType_JPEG;
                            cImgMsgPtr->ImageNumber = imgMsgPtr->ImageNumber;
                            cImgMsgPtr->GpsTimeStampSec = imgMsgPtr->ImageCaptureTimeStampSec;

                            //Compress the Image before recording or transmitting the image.
                            imageCompressedOk = imencode(".jpg", *txImage,
                                                         cImgMsgPtr->ImageBuffer,
                                                         _compress_params);

                            if(imageCompressedOk)
                            {
                                int imgSize = cImgMsgPtr->ImageBuffer.size();
                                //LOGINFO("Image Size: " << imgSize);
                            }
                        }
                        catch (std::exception &e)
                        {
                            LOGERROR("StreamRecordManager: Exception: " << e.what());
                        }
                    }
                    else
                    {
                        if (!_imageStreamDataQueueFullMsgSent)
                        {
                            LOGWARN("StreamRecordManager: the CompressedImageMessage Pool is Empty!");
                            _imageStreamDataQueueFullMsgSent = true;
                        }
                    }
                }

                //Record Image if required
                if (_imageLoggingControlMsg->EnableLogging)
                {
                    try
                    {
                        if(_videoRecordingEnabled)
                        {
                            if(!_videoRecorder_uptr->IsOpen())
                            {
                                frameWidth = imgMsgPtr->ImageFrame.cols;
                                frameHeight = imgMsgPtr->ImageFrame.rows;
                                _videoRecorder_uptr->SetImageParameters(frameWidth, frameHeight, _record_fps);
                                _videoRecorder_uptr->ResetVideoWriter();
                                _videoRecordStartTime = imgMsgPtr->ImageCaptureTimeStampSec;
                                _videoRecordStartImageNumber = imgMsgPtr->ImageNumber;
                            }

                            _telemetryMessage.LocalTimeSec = imgMsgPtr->ImageCaptureTimeStampSec;
                            _telemetryMessage.gpsTimeStampSec = imgMsgPtr->ImageCaptureTimeStampSec;
                            _telemetryMessage.DeltaTime = _telemetryMessage.gpsTimeStampSec - _videoRecordStartTime;
                            _telemetryMessage.ImageNumber = imgMsgPtr->ImageNumber - _videoRecordStartImageNumber;

                            _videoRecorder_uptr->RecordFrameToVideo((imgMsgPtr->ImageFrame), _telemetryMessage);
                        }
                        else
                        {
                            if (_imageLoggingControlMsg->VisionLoggingType == VisionLoggingType_e::LogCompressedImages
                                && imageCompressedOk &&
                                cImgMsgPtr != nullptr) {
                                //record compressed image.
                                _imagePlusMetadataRecorder.WriteImagePlusMetadata(*imgMsgPtr,
                                                                                  ImageStorageType_e::IST_JPEG,
                                                                                  (char *) cImgMsgPtr->ImageBuffer.data(),
                                                                                  cImgMsgPtr->ImageBuffer.size(),
                                                                                  cImgMsgPtr->ImageFormatType);
                            } else if (_imageLoggingControlMsg->VisionLoggingType ==
                                       VisionLoggingType_e::LogRawImages) {
                                //Record raw image.
                                int size = imgMsgPtr->ImageFrame.rows * imgMsgPtr->ImageFrame.cols *
                                           imgMsgPtr->ImageFrame.channels();
                                _imagePlusMetadataRecorder.WriteImagePlusMetadata(*imgMsgPtr,
                                                                                  ImageStorageType_e::IST_OpenCVMatRGB,
                                                                                  (char *) imgMsgPtr->ImageFrame.data,
                                                                                  size,
                                                                                  ImageFormatType_e::ImgFType_Raw);
                            } else {
                                _imagePlusMetadataRecorder.WriteImagePlusMetadata(*imgMsgPtr,
                                                                                  ImageStorageType_e::IST_None,
                                                                                  (char *) imgMsgPtr->ImageFrame.data,
                                                                                  0,
                                                                                  ImageFormatType_e::ImgFType_Raw);
                            }
                        }
                    }
                    catch (std::exception &e)
                    {
                        LOGERROR("StreamRecordManager Record Image: Exception: " << e.what());
                    }
                }

                if (txStreamImage && imageCompressedOk && cImgMsgPtr != nullptr)
                {
                    if (!_imageStreamMsgQueue_sptr->AddMessage(cImgMsgPtr))
                    {
                        cImgMsgSentOut = false;
                        //We have a limited number of messages in the compressedImageMessagePool
                        //so this should not be possible.
                        if (!_imageStreamDataQueueFullMsgSent)
                        {
                            LOGWARN("StreamRecordManager: ImageStreamData Queue is Full.")
                            _imageStreamDataQueueFullMsgSent = true;
                        }
                    } else
                    {
                        cImgMsgSentOut = true;
                        _imageStreamDataQueueFullMsgSent = false;
                    }
                }

                if (cImgMsgPtr != nullptr && !cImgMsgSentOut)
                {
                    //Return the Compressed Image Message back to the Pool
                    _compressedImageMessagePool_uptr->CheckInMessage(cImgMsgPtr);
                    _imageStreamDataQueueFullMsgSent = false;
                }

            }

            //Now return our ImagePlusMetadataMessage back to the pool to be used again.
            _imageMessagePool_uptr->CheckInMessage(imgMsgPtr);

            //Once again... add an empty message to the queue;
            //Doing this twice in the loop keeps the empties message queue full.
            AddEmptyImageMsgsToQueue();
        }

    }

    void StreamRecordManager::Shutdown()
    {
        _imagePlusMetadataRecorder.closeImageFile();
        _videoRecorder_uptr->ClearVideoWriter();
    }
}
