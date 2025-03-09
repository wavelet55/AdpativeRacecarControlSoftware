/* ****************************************************************
 * Vision Bridge
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Aug. 2016
 * 
 * Developed under contract for:
 * Academy Center for UAS Research 
 * Department of Electrical and Computer Engineering 
 * HQ USAFA/DFEC 
 * 2354 Fairchild Drive 
 * USAF Academy, CO 80840-6236 
 *******************************************************************/

#ifndef VIDERE_DEV_STREAMRECORDIMAGESCONTROLMESSAGE_H
#define VIDERE_DEV_STREAMRECORDIMAGESCONTROLMESSAGE_H

namespace videre
{
    //The Inertial States of the Vehicle typically sent
    //over from the vehicle.
    class StreamRecordImageControlMesssage : public Rabit::RabitMessage
    {
    public:

        /// <summary>
        /// Record Images On/Off
        /// </summary>
        bool RecordImagesEnabled;

        /// <summary>
        /// Stream Images On/Off
        /// </summary>
        bool StreamImagesEnabled;

        /// <summary>
        /// RecordCompressedImages
        /// If true, compressed images are recorded in the Imgage Plus
        /// metadata files, otherwise the raw image format is recorded.
        /// For post image analysis it is better to have the raw images
        /// but they are inherently much larger than compressed images.
        /// </summary>
        bool RecordCompressedImages;

        /// <summary>
        /// This is the max frame rate for streaming images...
        /// The Vision System may be able to capture and process images
        /// at 30 fps or some rate higher than bandwidth will allow to
        /// stream images to the ground.  Setting this number to a lower number
        /// say 5 fps will reduce the image rate sent to the ground.
        /// If zero or less... stream images at max rate.
        /// </summary>
        double StreamImageFrameRate;


        /// <summary>
        /// ImageCompressionQuality
        /// Highest quality is 100, lowest is 1
        /// High Quality means low compression, lower quality
        /// means higher compression.
        /// </summary>
        int ImageCompressionQuality;

        /// <summary>
        /// StreamImageScaleDownFactor
        /// The factor to scale the image down by before
        /// compressing the image and sending it out.
        /// A value of 2.0 will cut the image in half.
        /// The value must be >= 1.0.  1.0 will not change the
        /// image size.
        /// </summary>
        double StreamImageScaleDownFactor;


    public:

        StreamRecordImageControlMesssage() : RabitMessage()
        {
            Clear();
        }

        StreamRecordImageControlMesssage(const StreamRecordImageControlMesssage &msg)
        {
            *this = msg;
        }

        void Clear()
        {
            RecordImagesEnabled = false;
            StreamImagesEnabled = false;
            RecordCompressedImages = false;
            StreamImageFrameRate = 0;
            ImageCompressionQuality = 50;
            StreamImageScaleDownFactor = 1.0;
        }

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<StreamRecordImageControlMesssage>(new StreamRecordImageControlMesssage(*this));
            return std::move(clone);

        }

        void SetImageCompressionQuality(int qf)
        {
            qf = qf < 10 ? 10 : qf > 100 ? 100 : qf;
            ImageCompressionQuality = qf;
        }

        void SetImageScaleDownFactor(int qf)
        {
            qf = qf < 1.0 ? 1.0 : qf > 25.0 ? 25.0 : qf;
            StreamImageScaleDownFactor = qf;
        }

        void SetStreamImageFrameRate(double fps)
        {
            fps = fps < 0.01 ? 0.01 : fps > 100.0 ? 100.0 : fps;
            StreamImageFrameRate = fps;
        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            bool msgCopied = false;
            if (msg->GetTypeIndex() == std::type_index(typeid(StreamRecordImageControlMesssage)))
            {
                //Ensure the Copy process does not loose the mesages'
                //publish subscribe reference.
                std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

                Rabit::RabitMessage::CopyMessage(msg); // call baseclass
                StreamRecordImageControlMesssage *visMsg = static_cast<StreamRecordImageControlMesssage *>(msg);
                *this = *visMsg;

                this->SetGlobalPublishSubscribeMessageRef(psmr);
                msgCopied = true;
            }
            return msgCopied;
        }

    };

}

#endif //VIDERE_DEV_STREAMRECORDIMAGESCONTROLMESSAGE_H
