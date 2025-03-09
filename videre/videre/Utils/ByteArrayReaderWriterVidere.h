/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Oct 2016
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
 * Desc: ByteArrayReaderWriter
 * Two class ByteArrayReader and ByteArrayWriter are definded.
 * These classes are used for packing or unpacking data into a
 * byte array.  The endianness of the packing order can be defined.

  *******************************************************************/

#ifndef VIDERE_DEV_BYTEARRAYREADERVIDEREWRITER_H
#define VIDERE_DEV_BYTEARRAYREADERVIDEREWRITER_H

#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <string>
#include "LatLonAltStruct.h"
#include "XYZCoord_t.h"
#include "RollPitchYaw_t.h"
#include "AzimuthElevation_t.h"
#include "ByteArrayReaderWriter.h"

using namespace MathLibsNS;
using namespace Rabit;

namespace VidereUtils
{


    class ByteArrayWriterVidere : public Rabit::ByteArrayWriter
    {

    public:
        ByteArrayWriterVidere(byte  *byteArrayPtr, int byteArraySize, EndianOrder_e endianness)
            :Rabit::ByteArrayWriter(byteArrayPtr, byteArraySize, endianness)
        {
        }


        bool writeXYZ(MathLibsNS::XYZCoord_t &value);
        bool writeLatLonAlt(GeoCoordinateSystemNS::LatLonAltCoord_t &value);
        bool writeRollPitchYaw(MathLibsNS::RollPitchYaw_t &value);
        bool writeAzimuthElevation(GeoCoordinateSystemNS::AzimuthElevation_t &value);

    };


    class ByteArrayReaderVidere : public Rabit::ByteArrayReader
    {

    public:
        ByteArrayReaderVidere(unsigned char  *byteArrayPtr, int byteArraySize, EndianOrder_e endianness)
            : ByteArrayReader(byteArrayPtr, byteArraySize, endianness)
        {
        }

        MathLibsNS::XYZCoord_t readXYZ();
        GeoCoordinateSystemNS::LatLonAltCoord_t readLatLonAlt();
        MathLibsNS::RollPitchYaw_t  readRollPitchYaw();
        GeoCoordinateSystemNS::AzimuthElevation_t readAzimuthElevation();

    };

}
#endif //VIDERE_DEV_BYTEARRAYREADERWRITER_H
