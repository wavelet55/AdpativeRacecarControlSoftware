//
// Created by wavelet on 10/7/16.
//

#include "TestByteArrayReaderWriter.h"

#include <vector>
#include <string>

using namespace VidereUtils;
using namespace std;

TEST_F(TestByteArrayReaderWriter, TestReadWriteOfValues)
{
    uint8_t rbyteVal;
    int16_t rint16Val;
    uint16_t ruint16Val;

    int32_t rint32Val;
    uint32_t ruint32Val;

    int64_t rint64Val;
    uint64_t ruint64Val;

    float rfloatVal1;
    float rfloatVal2;
    double rdoubleVal1;
    double rdoubleVal2;


    uint8_t byteVal = 147;
    int16_t int16Val = -13567;
    uint16_t uint16Val = 24793;

    int32_t int32Val = -1345678876;
    uint32_t uint32Val = 1987654321;

    int64_t int64Val = -32112345678876L;
    uint64_t uint64Val = 1231987654321L;

    float floatVal1 = 12.34567;
    float floatVal2 = -4512.345;

    double doubleVal1 = 235.87654;
    double doubleVal2 = -547.986327;

    //EXPECT_EQ(dirExist, true);

    byte byteArray[2048];

    ByteArrayWriter bw(byteArray, 2048, EndianOrder_e::Endian_Big);
    ByteArrayReader br(byteArray, 2048, EndianOrder_e::Endian_Big);

    bw.writeByte(byteVal);
    bw.writeInt16(int16Val);
    bw.writeUInt16(uint16Val);
    bw.writeByte(byteVal);
    bw.writeInt32(int32Val);
    bw.writeUInt32(uint32Val);
    bw.writeByte(byteVal);
    bw.writeInt64(int64Val);
    bw.writeUInt64(uint64Val);
    bw.writeFloat(floatVal1);
    bw.writeDouble(doubleVal1);
    bw.writeByte(byteVal);
    bw.writeFloat(floatVal2);
    bw.writeDouble(doubleVal2);

    rbyteVal = br.readByte();
    rint16Val = br.readInt16();
    ruint16Val = br.readUInt16();
    rbyteVal = br.readByte();
    rint32Val = br.readInt32();
    ruint32Val = br.readUInt32();
    rbyteVal = br.readByte();
    rint64Val = br.readInt64();
    ruint64Val = br.readUInt64();
    rfloatVal1 = br.readFloat();
    rdoubleVal1 = br.readDouble();
    rbyteVal = br.readByte();
    rfloatVal2 = br.readFloat();
    rdoubleVal2 = br.readDouble();

    /****************************
    EXPECT_EQ(rbyteVal, byteVal);
    EXPECT_EQ(rint16Val, int16Val);
    EXPECT_EQ(ruint16Val, uint16Val);
    EXPECT_EQ(rint32Val, int32Val);
    EXPECT_EQ(ruint32Val, uint32Val);
    EXPECT_EQ(rint64Val, int64Val);
    EXPECT_EQ(ruint64Val, uint64Val);
    EXPECT_EQ(rfloatVal1, floatVal1);
    EXPECT_EQ(rfloatVal2, floatVal2);
    EXPECT_EQ(rdoubleVal1, doubleVal1);
    EXPECT_EQ(rdoubleVal2, doubleVal2);
     **************************/
}

TEST_F(TestByteArrayReaderWriter, TestReadWriteOfStrings)
{
    bool error;
    byte byteArray[2048];
    ByteArrayWriter bw(byteArray, 2048, EndianOrder_e::Endian_Big);
    ByteArrayReader br(byteArray, 2048, EndianOrder_e::Endian_Big);

    string str1 = "Hello World";
    string str2 = "I'm here!";
    string rstr1;
    string rstr2;

    bw.writeString(str1);
    bw.writeString(str2);

    error = br.readString(&rstr1);
    //EXPECT_EQ(error, false);
    //EXPECT_EQ(str1, rstr1);

    error = br.readString(&rstr2);
    //EXPECT_EQ(error, false);
    //EXPECT_EQ(str2, rstr2);

    const char *str1Array = str1.data();
    char rstr1Array[64];

    bw.writeString(str1Array, str1.length());
    error = br.readString(rstr1Array, 64);
    //EXPECT_EQ(error, false);
    //EXPECT_EQ(str1, rstr1);

}