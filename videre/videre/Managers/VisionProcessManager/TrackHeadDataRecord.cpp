/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: June 2018
 *
  *******************************************************************/

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include "ByteArrayReaderWriterVidere.h"
#include "TrackHeadDataRecord.h"

namespace videre
{

    TrackHeadDataRecord::TrackHeadDataRecord()
            : DataRecorderAbstractRecord()
    {
        _maxRecordSize = TRACKHEADMAXRECORDSIZE;
        RecordType = DataRecordType_e::DRT_TrackHeadOrientation;
        TrackHeadOrientationMsg = nullptr;
    }

    //Serialize data and put results into the local buffer;
    //DataPtr is a pointer to a structure or object containing the data
    //to be serialized.  If is is null, then the data to be serialized
    //is part of the concrete object.
    //Returns the number of bytes of the serialized data.
    char* TrackHeadDataRecord::serializeDataRecord(uint32_t *recordSizeOut)
    {
        _currentSerializedRecordSize = 0;
        if(TrackHeadOrientationMsg != nullptr)
        {
            VidereUtils::ByteArrayWriterVidere bw((byte *) _recordBuf, TRACKHEADMAXRECORDSIZE,
                                                  Rabit::EndianOrder_e::Endian_Big);

            TimeStampSec = TrackHeadOrientationMsg->GetTimeStamp();
            bw.writeDouble(TimeStampSec);  //system timestamp
            bw.writeDouble(TrackHeadOrientationMsg->ImageCaptureTimeStampSec);
            bw.writeInt32(TrackHeadOrientationMsg->ImageNumber);
            bw.writeByte(TrackHeadOrientationMsg->TrackHeadOrientationData.IsDataValid ? 1 : 0);
            bw.writeDouble(TrackHeadOrientationMsg->TrackHeadOrientationData.HeadOrientationQuaternion.qScale);
            bw.writeDouble(TrackHeadOrientationMsg->TrackHeadOrientationData.HeadOrientationQuaternion.qVec.x);
            bw.writeDouble(TrackHeadOrientationMsg->TrackHeadOrientationData.HeadOrientationQuaternion.qVec.y);
            bw.writeDouble(TrackHeadOrientationMsg->TrackHeadOrientationData.HeadOrientationQuaternion.qVec.z);
            bw.writeDouble(TrackHeadOrientationMsg->TrackHeadOrientationData.HeadTranslationVec.x);
            bw.writeDouble(TrackHeadOrientationMsg->TrackHeadOrientationData.HeadTranslationVec.y);
            bw.writeDouble(TrackHeadOrientationMsg->TrackHeadOrientationData.HeadTranslationVec.z);
            _currentSerializedRecordSize = bw.Idx;
            *recordSizeOut = _currentSerializedRecordSize;
        }
        return _recordBuf;
    }

    //Deserialized the data buffer into a data structure.
    //Returns true if error, false otherwise.
    bool TrackHeadDataRecord::deserialzedDataRecord(char* dataBufPtr)
    {
        bool error = true;
        if(TrackHeadOrientationMsg.get() == nullptr)
        {
            TrackHeadOrientationMsg = std::make_shared<TrackHeadOrientationMessage>();
        }
        if(TrackHeadOrientationMsg != nullptr)
        {
            VidereUtils::ByteArrayReaderVidere br((byte *) dataBufPtr, TRACKHEADMAXRECORDSIZE,
                                            Rabit::EndianOrder_e::Endian_Big);
            TimeStampSec = br.readDouble();
            TrackHeadOrientationMsg->SetTimeStamp(TimeStampSec);
            TrackHeadOrientationMsg->ImageCaptureTimeStampSec = br.readDouble();
            TrackHeadOrientationMsg->ImageNumber = br.readInt32();
            TrackHeadOrientationMsg->TrackHeadOrientationData.IsDataValid = br.readByte() != 0 ? true : false;
            TrackHeadOrientationMsg->TrackHeadOrientationData.HeadOrientationQuaternion.qScale = br.readDouble();
            TrackHeadOrientationMsg->TrackHeadOrientationData.HeadOrientationQuaternion.qVec.x = br.readDouble();
            TrackHeadOrientationMsg->TrackHeadOrientationData.HeadOrientationQuaternion.qVec.y = br.readDouble();
            TrackHeadOrientationMsg->TrackHeadOrientationData.HeadOrientationQuaternion.qVec.z = br.readDouble();
            TrackHeadOrientationMsg->TrackHeadOrientationData.HeadTranslationVec.x = br.readDouble();
            TrackHeadOrientationMsg->TrackHeadOrientationData.HeadTranslationVec.y = br.readDouble();
            TrackHeadOrientationMsg->TrackHeadOrientationData.HeadTranslationVec.z = br.readDouble();
            error = false;
        }
        return error;
    }

    //Read the data record from the file.  It is assumed the file is set to the
    // start of the record and the record header has been read.
    //Returns true if error, false otherwise.
    bool TrackHeadDataRecord::readDataRecordFromFile(std::ifstream &logFile, int recordSize)
    {
        bool error = true;
        try
        {
            logFile.read(_recordBuf, recordSize);
            error = deserialzedDataRecord(_recordBuf);
        }
        catch (std::exception &e)
        {
            error = true;
        }
        return error;
    }




}