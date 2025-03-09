
/* ****************************************************************
  *******************************************************************/

/* Auto-generated from file VidereRecordControlMsg.msg, do not modify directly */


using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Int8 = System.SByte;
using UInt8 = System.Byte;


namespace VisionBridge.Messages
{
    public class VidereRecordControlMsg : VSMessage
    {

        public UInt8 command { get; set;}
        public Int32 file_index { get; set;}
        public UInt8[] file_guid = new UInt8[36];
        public UInt32 record_version { get; set;}
        public UInt8[] exercise_guid = new UInt8[36];
        public UInt8 exercise_status { get; set;}

       
        public VidereRecordControlMsg()
        {
            Clear();
        }

        public override void Clear()
        {
            base.Clear();
            command = 0;
            file_index = 0;
            Array.Clear( file_guid, 0, 36);
            record_version = 0;
            Array.Clear( exercise_guid, 0, 36);
            exercise_status = 0;

        }


        public override byte[] Serialize(MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteListArrayWriter bw = new ByteListArrayWriter(EndianOrder_e.big);
            SerializeHeader(bw);
            bw.writeUInt8(command);
            bw.writeInt32(file_index);
            for(int n = 0; n < 36; n++){
                bw.writeUInt8(file_guid[n]);
            }
            bw.writeUInt32(record_version);
            for(int n = 0; n < 36; n++){
                bw.writeUInt8(exercise_guid[n]);
            }
            bw.writeUInt8(exercise_status);

            return bw.ByteArray;
        }

        public override int Deserialize(byte[] b, MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteArrayReader br = new ByteArrayReader(b, EndianOrder_e.big);
            DeserializeHeader(br);
            command = br.readUInt8();
            file_index = br.readInt32();
            for(int n = 0; n < 36; n++){
                file_guid[n] = br.readUInt8();
            }
            record_version = br.readUInt32();
            for(int n = 0; n < 36; n++){
                exercise_guid[n] = br.readUInt8();
            }
            exercise_status = br.readUInt8();

            return br.Idx;
        }
    }
}
