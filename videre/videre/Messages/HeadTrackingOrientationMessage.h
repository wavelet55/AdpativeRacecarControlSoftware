/* ****************************************************************
 * Message
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Randy Direen PhD
 * Date: May 2016
 *
  *******************************************************************/


#ifndef VIDERE_DEV_HEADTRACKINGORIENTATIONMESSAGE_H
#define VIDERE_DEV_HEADTRACKINGORIENTATIONMESSAGE_H

#include <string>
#include <RabitMessage.h>
#include "CommonImageProcTypesDefs.h"

namespace videre
{

    class TrackHeadOrientationMessage : public Rabit::RabitMessage
    {

    public:
        ImageProcLibsNS::TrackHeadOrientationData_t TrackHeadOrientationData;

        double CovarianceNorm;

        uint32_t ImageNumber;

        double ImageCaptureTimeStampSec;

    public:
        TrackHeadOrientationMessage() : RabitMessage()
        {
            Clear();
        }

        TrackHeadOrientationMessage(const TrackHeadOrientationMessage& msg)
        {
            *this = msg;
        }

        virtual void Clear() final
        {
            TrackHeadOrientationData.Clear();
            ImageNumber = 0;
            ImageCaptureTimeStampSec = 0;
            CovarianceNorm = 0;
        }


        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<TrackHeadOrientationMessage>(new TrackHeadOrientationMessage(*this));
            return std::move(clone);
        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            Rabit::RabitMessage::CopyMessage(msg); // call baseclass
            if (msg->GetTypeIndex() == std::type_index(typeid(TrackHeadOrientationMessage)))
            {
                //Ensure the Copy process does not loose the mesages'
                //publish subscribe reference.
                std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

                TrackHeadOrientationMessage *coMsg = static_cast<TrackHeadOrientationMessage *>(msg);
                *this = *coMsg;

                this->SetGlobalPublishSubscribeMessageRef(psmr);
                return true;
            }
            return false;
        }


    };
}


#endif //VIDERE_DEV_HEADTRACKINGCONTROLMESSAGE_H

