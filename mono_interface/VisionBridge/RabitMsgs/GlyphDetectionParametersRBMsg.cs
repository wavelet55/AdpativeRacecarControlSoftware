
/* ****************************************************************
  *******************************************************************/

/* Auto-generated from file GlyphDetectionParametersRBMsg.msg, do not modify directly */


using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Int8 = System.SByte;
using UInt8 = System.Byte;


namespace VisionBridge.Messages
{
    public class GlyphDetectionParametersRBMsg : VSMessage
    {

        public Int32 Canny_low { get; set;}
        public Int32 Canny_high { get; set;}
        public Int32 GlyphAreaPixels_min { get; set;}
        public Int32 GlyphAreaPixels_max { get; set;}
        public Int32 NumberOfIterations { get; set;}
        public double ReprojectionErrorDistance { get; set;}
        public double ConfidencePercent { get; set;}
        public UInt8 GlyphDetectorImageDisplayType { get; set;}
        public UInt8 GlyphModelIndex { get; set;}

       
        public GlyphDetectionParametersRBMsg()
        {
            Clear();
        }

        public override void Clear()
        {
            base.Clear();
            Canny_low = 0;
            Canny_high = 0;
            GlyphAreaPixels_min = 0;
            GlyphAreaPixels_max = 0;
            NumberOfIterations = 0;
            ReprojectionErrorDistance = 0;
            ConfidencePercent = 0;
            GlyphDetectorImageDisplayType = 0;
            GlyphModelIndex = 0;

        }


        public override byte[] Serialize(MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteListArrayWriter bw = new ByteListArrayWriter(EndianOrder_e.big);
            SerializeHeader(bw);
            bw.writeInt32(Canny_low);
            bw.writeInt32(Canny_high);
            bw.writeInt32(GlyphAreaPixels_min);
            bw.writeInt32(GlyphAreaPixels_max);
            bw.writeInt32(NumberOfIterations);
            bw.writeDouble(ReprojectionErrorDistance);
            bw.writeDouble(ConfidencePercent);
            bw.writeUInt8(GlyphDetectorImageDisplayType);
            bw.writeUInt8(GlyphModelIndex);

            return bw.ByteArray;
        }

        public override int Deserialize(byte[] b, MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteArrayReader br = new ByteArrayReader(b, EndianOrder_e.big);
            DeserializeHeader(br);
            Canny_low = br.readInt32();
            Canny_high = br.readInt32();
            GlyphAreaPixels_min = br.readInt32();
            GlyphAreaPixels_max = br.readInt32();
            NumberOfIterations = br.readInt32();
            ReprojectionErrorDistance = br.readDouble();
            ConfidencePercent = br.readDouble();
            GlyphDetectorImageDisplayType = br.readUInt8();
            GlyphModelIndex = br.readUInt8();

            return br.Idx;
        }
    }
}
