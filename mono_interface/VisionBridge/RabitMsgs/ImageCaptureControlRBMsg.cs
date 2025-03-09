
/* ****************************************************************
  *******************************************************************/

/* Auto-generated from file ImageCaptureControlRBMsg.msg, do not modify directly */


using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Int8 = System.SByte;
using UInt8 = System.Byte;


namespace VisionBridge.Messages
{
    public class ImageCaptureControlRBMsg : VSMessage
    {

        public bool ImageCaptureEnabled { get; set;}
        public UInt32 NumberOfImagesToCapture { get; set;}
        public double DesiredFramesPerSecond { get; set;}
        public UInt32 DesiredImageWidth { get; set;}
        public UInt32 DesiredImageHeight { get; set;}
        public UInt8 ImageCaptureSource { get; set;}
        public UInt8 ImageCaptureFormat { get; set;}
        public string ImageCaptureSourceConfigPri { get; set;}
        public string ImageCaptureSourceConfigSec { get; set;}
        public bool ImageSourceLoopAround { get; set;}
        public bool AutoFocusEnable { get; set;}

       
        public ImageCaptureControlRBMsg()
        {
            Clear();
        }

        public override void Clear()
        {
            base.Clear();
            ImageCaptureEnabled = false;
            NumberOfImagesToCapture = 0;
            DesiredFramesPerSecond = 0;
            DesiredImageWidth = 0;
            DesiredImageHeight = 0;
            ImageCaptureSource = 0;
            ImageCaptureFormat = 0;
            ImageCaptureSourceConfigPri = "";
            ImageCaptureSourceConfigSec = "";
            ImageSourceLoopAround = false;
            AutoFocusEnable = false;

        }


        public override byte[] Serialize(MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteListArrayWriter bw = new ByteListArrayWriter(EndianOrder_e.big);
            SerializeHeader(bw);
            bw.writeBool(ImageCaptureEnabled);
            bw.writeUInt32(NumberOfImagesToCapture);
            bw.writeDouble(DesiredFramesPerSecond);
            bw.writeUInt32(DesiredImageWidth);
            bw.writeUInt32(DesiredImageHeight);
            bw.writeUInt8(ImageCaptureSource);
            bw.writeUInt8(ImageCaptureFormat);
            bw.writeString(ImageCaptureSourceConfigPri);
            bw.writeString(ImageCaptureSourceConfigSec);
            bw.writeBool(ImageSourceLoopAround);
            bw.writeBool(AutoFocusEnable);

            return bw.ByteArray;
        }

        public override int Deserialize(byte[] b, MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteArrayReader br = new ByteArrayReader(b, EndianOrder_e.big);
            DeserializeHeader(br);
            ImageCaptureEnabled = br.readBool();
            NumberOfImagesToCapture = br.readUInt32();
            DesiredFramesPerSecond = br.readDouble();
            DesiredImageWidth = br.readUInt32();
            DesiredImageHeight = br.readUInt32();
            ImageCaptureSource = br.readUInt8();
            ImageCaptureFormat = br.readUInt8();
            ImageCaptureSourceConfigPri = br.readString();
            ImageCaptureSourceConfigSec = br.readString();
            ImageSourceLoopAround = br.readBool();
            AutoFocusEnable = br.readBool();

            return br.Idx;
        }
    }
}
