/* ****************************************************************
 * Message
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

#ifndef TELEMETRY_MESSAGE
#define TELEMETRY_MESSAGE

#include <iostream>
#include <string>
#include <RabitMessage.h>

namespace videre
{

    class TelemetryMessage : public Rabit::RabitMessage
    {

    public:
        bool IsFromHOPS = false;        /* Did we get this from HOPS or is it generated */
        unsigned int LocalTimeSec = 0;    /* Time local to the Vision System */
        double DeltaTime = 0;   /* Time between frames */
        bool CoordinatesLatLonOrXY = false;
        double LatitudeRadOrY = 0;
        double LongitudeRadOrX = 0;
        double AltitudeMSL = 0;
        double HeightAGL = 0;
        double VelEastMpS = 0;
        double VelNorthMpS = 0;
        double VelDownMpS = 0;
        double RollRad = 0;
        double PitchRad = 0;
        double YawRad = 0;
        double RollRateRadps = 0;
        double PitchRateRadps = 0;
        double YawRateRadps = 0;
        double gpsTimeStampSec = 0;
        unsigned int ImageNumber = 0;

    public:
        TelemetryMessage() : RabitMessage()
        {
            Clear();
        }

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final
        {
            auto clone = std::unique_ptr<TelemetryMessage>(new TelemetryMessage(*this));
            return std::move(clone);
        }

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final
        {
            Rabit::RabitMessage::CopyMessage(msg); // call baseclass
            if (msg->GetTypeIndex() == std::type_index(typeid(TelemetryMessage)))
            {
                //Ensure the Copy process does not loose the mesages'
                //publish subscribe reference.
                std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

                TelemetryMessage * inpMsg = static_cast<TelemetryMessage *>(msg);
                *this = *inpMsg;

                this->SetGlobalPublishSubscribeMessageRef(psmr);
                return true;
            }
            return false;
        }

        virtual void Clear() final
        {
            IsFromHOPS = false;
            LocalTimeSec = 0;
            DeltaTime = 0;
            CoordinatesLatLonOrXY = false;
            LatitudeRadOrY = 0;
            LongitudeRadOrX = 0;
            AltitudeMSL = 0;
            HeightAGL = 0;
            VelEastMpS = 0;
            VelNorthMpS = 0;
            VelDownMpS = 0;
            RollRad = 0;
            PitchRad = 0;
            YawRad = 0;
            RollRateRadps = 0;
            PitchRateRadps = 0;
            YawRateRadps = 0;
            gpsTimeStampSec = 0;
            ImageNumber = 0;
        }

        virtual std::string ToString() const final
        {
            std::ostringstream os;
            os << "TelemetryMessage: " << IsFromHOPS << ", "
               << ImageNumber << ", "
               << LocalTimeSec << ", "
               << DeltaTime << ", "
               << CoordinatesLatLonOrXY << ", "
               << LatitudeRadOrY << ", "
               << LongitudeRadOrX << ", "
               << AltitudeMSL << ", "
               << HeightAGL << ", "
               << VelEastMpS << ", "
               << VelNorthMpS << ", "
               << VelDownMpS << ", "
               << RollRad << ", "
               << PitchRad << ", "
               << YawRad << ", "
               << RollRateRadps << ", "
               << PitchRateRadps << ", "
               << YawRateRadps << ", "
               << gpsTimeStampSec;
            return os.str();
        }

    };

}
#endif //TELEMETRY_MESSAGE
