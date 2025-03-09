/* ****************************************************************
* Athr(s): Harry Direen PhD, Randy Direen Phd.
* DireenTech Inc.  (www.DireenTech.com)
* Date: March. 2016
*
* Academy Center for UAS Research
* Department of Electrical and Computer Engineering
        * HQ USAFA/DFEC
* 2354 Fairchild Drive
* USAF Academy, CO 80840-6236
*
*******************************************************************/

#include "TargetParametersMessage.h"
#include "GeoCoordinateSystem.h"
#include <math.h>
#include <vector>
#include <boost/math/constants/constants.hpp>

using namespace GeoCoordinateSystemNS;


namespace videre
{

    void TargetParametersMessage::Clear()
    {
        TargetTypeCode = 0;
        IsTaticalTarget = false;
        TargetSizeOrWidth = 0;
        TargetPerimeterOrLenght = 0;
        TargetIRHeatGradient = 0;
        TargetIRSize = 0;
        TargetRGBColorCode = 0;
    }

    void TargetParametersMessage::SetMessage(const vision_messages::TargetParametersPBMsg msgPB)
    {
        TargetTypeCode = msgPB.targettypecode();
        IsTaticalTarget = msgPB.istaticaltarget();
        TargetSizeOrWidth = msgPB.targetsizeorwidth();
        TargetPerimeterOrLenght = msgPB.targetperimeterorlenght();
        TargetIRHeatGradient = msgPB.targetirheatgradient();
        TargetIRSize = msgPB.targetirsize();
        TargetRGBColorCode = msgPB.targetrgbcolorcode();
    }

    std::unique_ptr<Rabit::RabitMessage> TargetParametersMessage::Clone() const
    {
        auto clone = std::unique_ptr<TargetParametersMessage>(new TargetParametersMessage(*this));
        return std::move(clone);
    }

    bool TargetParametersMessage::CopyMessage(Rabit::RabitMessage *msg)
    {
        Rabit::RabitMessage::CopyMessage(msg); // call baseclass
        if (msg->GetTypeIndex() == std::type_index(typeid(TargetParametersMessage)))
        {
            //Ensure the Copy process does not loose the mesages'
            //publish subscribe reference.
            std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

            TargetParametersMessage *coMsg = static_cast<TargetParametersMessage *>(msg);
            *this = *coMsg;

            this->SetGlobalPublishSubscribeMessageRef(psmr);
            return true;
        }
        return false;
    }

}


