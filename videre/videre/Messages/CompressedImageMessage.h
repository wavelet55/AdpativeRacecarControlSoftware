/* ****************************************************************
* Message
* DireenTech Inc.  (www.direentech.com)
* Athr: Harry Direen PhD
* Date: Aug 2016
*
* Developed under contract for:
* Academy Center for UAS Research
* Department of Electrical and Computer Engineering
       * HQ USAFA/DFEC
* 2354 Fairchild Drive
* USAF Academy, CO 80840-6236
*
*******************************************************************/

#ifndef VIDERE_DEV_COMPRESSEDIMAGEMESSAGE_H
#define VIDERE_DEV_COMPRESSEDIMAGEMESSAGE_H

#include <string>
#include <RabitMessage.h>
#include "global_defines.h"

namespace videre
{
    class CompressedImageMessage : public Rabit::RabitMessage
    {

    public:
        ImageFormatType_e ImageFormatType = ImageFormatType_e::ImgFType_Raw;

        std::vector<unsigned char> ImageBuffer;

        //Every Image has a unique number associated with it... created
        //when the image is captured/
        unsigned int ImageNumber = 0;       //Every Image has a unique

        //The timestamp when the image was captured.
        double GpsTimeStampSec = 0;

    public:
        CompressedImageMessage() : RabitMessage()
        {
            Clear();
        }

        virtual void Clear() final
        {
            ImageFormatType = ImageFormatType_e::ImgFType_Raw;
            ImageNumber = 0;
            GpsTimeStampSec = 0;
            ImageBuffer.clear();
        }

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final;

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final;

        virtual std::string ToString() const final;

        std::string ImageFormatAsString();
    };
}




#endif //VIDERE_DEV_COMPRESSEDIMAGEMESSAGE_H
