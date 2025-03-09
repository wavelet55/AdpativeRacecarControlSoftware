
/* ****************************************************************
  *******************************************************************/

/* Auto-generated from file StreamControlRBMsg.msg, do not modify directly */


using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Int8 = System.SByte;
using UInt8 = System.Byte;


namespace VisionBridge.Messages
{
    public class StreamControlRBMsg : VSMessage
    {

        public bool StreamImagesEnabled { get; set;}
        public double StreamImageFrameRate { get; set;}
        public UInt32 ImageCompressionQuality { get; set;}
        public double StreamImageScaleDownFactor { get; set;}

       
        public StreamControlRBMsg()
        {
            Clear();
        }

        public override void Clear()
        {
            base.Clear();
            StreamImagesEnabled = false;
            StreamImageFrameRate = 0;
            ImageCompressionQuality = 0;
            StreamImageScaleDownFactor = 0;

        }


        public override byte[] Serialize(MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteListArrayWriter bw = new ByteListArrayWriter(EndianOrder_e.big);
            SerializeHeader(bw);
            bw.writeBool(StreamImagesEnabled);
            bw.writeDouble(StreamImageFrameRate);
            bw.writeUInt32(ImageCompressionQuality);
            bw.writeDouble(StreamImageScaleDownFactor);

            return bw.ByteArray;
        }

        public override int Deserialize(byte[] b, MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteArrayReader br = new ByteArrayReader(b, EndianOrder_e.big);
            DeserializeHeader(br);
            StreamImagesEnabled = br.readBool();
            StreamImageFrameRate = br.readDouble();
            ImageCompressionQuality = br.readUInt32();
            StreamImageScaleDownFactor = br.readDouble();

            return br.Idx;
        }
    }
}
