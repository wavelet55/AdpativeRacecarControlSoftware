
/* ****************************************************************
  *******************************************************************/

/* Auto-generated from file ImageCompressionControlRBMsg.msg, do not modify directly */


using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Int8 = System.SByte;
using UInt8 = System.Byte;


namespace VisionBridge.Messages
{
    public class ImageCompressionControlRBMsg : VSMessage
    {

        public bool EnableImageCompression { get; set;}
        public bool TransmitCompressedImages { get; set;}
        public UInt8 ImageComressionType { get; set;}
        public double ImageCompressionRatio { get; set;}
        public double FrameRate { get; set;}

       
        public ImageCompressionControlRBMsg()
        {
            Clear();
        }

        public override void Clear()
        {
            base.Clear();
            EnableImageCompression = false;
            TransmitCompressedImages = false;
            ImageComressionType = 0;
            ImageCompressionRatio = 0;
            FrameRate = 0;

        }


        public override byte[] Serialize(MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteListArrayWriter bw = new ByteListArrayWriter(EndianOrder_e.big);
            SerializeHeader(bw);
            bw.writeBool(EnableImageCompression);
            bw.writeBool(TransmitCompressedImages);
            bw.writeUInt8(ImageComressionType);
            bw.writeDouble(ImageCompressionRatio);
            bw.writeDouble(FrameRate);

            return bw.ByteArray;
        }

        public override int Deserialize(byte[] b, MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteArrayReader br = new ByteArrayReader(b, EndianOrder_e.big);
            DeserializeHeader(br);
            EnableImageCompression = br.readBool();
            TransmitCompressedImages = br.readBool();
            ImageComressionType = br.readUInt8();
            ImageCompressionRatio = br.readDouble();
            FrameRate = br.readDouble();

            return br.Idx;
        }
    }
}
