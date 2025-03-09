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

#ifndef VIDERE_DEV_TARGETPARAMETERSMESSAGE_H
#define VIDERE_DEV_TARGETPARAMETERSMESSAGE_H

#include <iostream>
#include <string>
#include <RabitMessage.h>
#include "../ProtobufMessages/vision_messages.pb.h"


namespace videre
{
    /// <summary>
    /// This message contains target parameters.
    //  There will be seperate messages for each target type.
    /// </summary>
    class TargetParametersMessage : public Rabit::RabitMessage
    {
    public:

        /// <summary>
        /// Target Type Code
        /// This is an agreeded upont code for the type of target
        /// (Building, car, human...)  Since the codes will change based upon
        /// mission and image processing... an enum is not used.
        /// Image processing will return this code for the type of target it finds.
        /// </summary>
        unsigned int TargetTypeCode = 1;

        /// <summary>
        /// Flag that indicates whether or not the Target Type is
        /// tactical or not.
        /// </summary>
        bool IsTaticalTarget = 2;

        /// <summary>
        /// Estimated Target Size Or Width
        /// How this number is interpreted is based on the Target Type.
        /// </summary>
        double TargetSizeOrWidth = 3;

        /// <summary>
        /// Estimated Target Size Or Width
        /// How this number is interpreted is based on the Target Type.
        /// </summary>
        double TargetPerimeterOrLenght = 4;

        /// <summary>
        /// Estimated Target Infrared Heat Gradient
        /// </summary>
        double TargetIRHeatGradient = 5;

        /// <summary>
        /// Estimated Target Infrared Size
        /// </summary>
        double TargetIRSize = 6;

        /// <summary>
        /// Target EO RGB Color Code
        /// </summary>
        unsigned int TargetRGBColorCode = 7;

    public:

        TargetParametersMessage() : RabitMessage()
        {
            Clear();
        }

        TargetParametersMessage(const TargetParametersMessage& msg)
        {
            *this = msg;
        }

        void Clear();

        void SetMessage(const vision_messages::TargetParametersPBMsg msgPB);

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final;

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final;



    };

}

#endif //VIDERE_DEV_TARGETPARAMETERSMESSAGE_H
