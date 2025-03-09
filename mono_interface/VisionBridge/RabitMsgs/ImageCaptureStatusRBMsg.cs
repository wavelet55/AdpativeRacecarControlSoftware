
/* ****************************************************************
  *******************************************************************/

/* Auto-generated from file ImageCaptureStatusRBMsg.msg, do not modify directly */


using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Int8 = System.SByte;
using UInt8 = System.Byte;


namespace VisionBridge.Messages
{
    public class ImageCaptureStatusRBMsg : VSMessage
    {

        public bool ImageCaptureEnabled { get; set;}
        public bool ImageCaptureComplete { get; set;}
        public bool EndOfImages { get; set;}
        public UInt32 TotalNumberOfImagesCaptured { get; set;}
        public UInt32 CurrentNumberOfImagesCaptured { get; set;}
        public double AverageFramesPerSecond { get; set;}
        public UInt8 ImageCaptureSource { get; set;}
        public UInt8 ErrorCode { get; set;}

       
        public ImageCaptureStatusRBMsg()
        {
            Clear();
        }

        public override void Clear()
        {
            base.Clear();
            ImageCaptureEnabled = false;
            ImageCaptureComplete = false;
            EndOfImages = false;
            TotalNumberOfImagesCaptured = 0;
            CurrentNumberOfImagesCaptured = 0;
            AverageFramesPerSecond = 0;
            ImageCaptureSource = 0;
            ErrorCode = 0;

        }


        public override byte[] Serialize(MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteListArrayWriter bw = new ByteListArrayWriter(EndianOrder_e.big);
            SerializeHeader(bw);
            bw.writeBool(ImageCaptureEnabled);
            bw.writeBool(ImageCaptureComplete);
            bw.writeBool(EndOfImages);
            bw.writeUInt32(TotalNumberOfImagesCaptured);
            bw.writeUInt32(CurrentNumberOfImagesCaptured);
            bw.writeDouble(AverageFramesPerSecond);
            bw.writeUInt8(ImageCaptureSource);
            bw.writeUInt8(ErrorCode);

            return bw.ByteArray;
        }

        public override int Deserialize(byte[] b, MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteArrayReader br = new ByteArrayReader(b, EndianOrder_e.big);
            DeserializeHeader(br);
            ImageCaptureEnabled = br.readBool();
            ImageCaptureComplete = br.readBool();
            EndOfImages = br.readBool();
            TotalNumberOfImagesCaptured = br.readUInt32();
            CurrentNumberOfImagesCaptured = br.readUInt32();
            AverageFramesPerSecond = br.readDouble();
            ImageCaptureSource = br.readUInt8();
            ErrorCode = br.readUInt8();

            return br.Idx;
        }
    }
}
