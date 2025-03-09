
/* ****************************************************************
  *******************************************************************/

/* Auto-generated from file VidereRecordStatusMsg.msg, do not modify directly */


using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Int8 = System.SByte;
using UInt8 = System.Byte;


namespace VisionBridge.Messages
{
    public class VidereRecordStatusMsg : VSMessage
    {

        public UInt8 record_file_type { get; set;}
        public UInt8 record_file_status { get; set;}
        public UInt8 record_status { get; set;}
        public Int32 file_index { get; set;}
        public UInt8[] record_guid = new UInt8[36];
        public UInt32 record_index { get; set;}
        public UInt32 data_start_index { get; set;}
        public UInt32 data_stop_index { get; set;}

       
        public VidereRecordStatusMsg()
        {
            Clear();
        }

        public override void Clear()
        {
            base.Clear();
            record_file_type = 0;
            record_file_status = 0;
            record_status = 0;
            file_index = 0;
            Array.Clear( record_guid, 0, 36);
            record_index = 0;
            data_start_index = 0;
            data_stop_index = 0;

        }


        public override byte[] Serialize(MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteListArrayWriter bw = new ByteListArrayWriter(EndianOrder_e.big);
            SerializeHeader(bw);
            bw.writeUInt8(record_file_type);
            bw.writeUInt8(record_file_status);
            bw.writeUInt8(record_status);
            bw.writeInt32(file_index);
            for(int n = 0; n < 36; n++){
                bw.writeUInt8(record_guid[n]);
            }
            bw.writeUInt32(record_index);
            bw.writeUInt32(data_start_index);
            bw.writeUInt32(data_stop_index);

            return bw.ByteArray;
        }

        public override int Deserialize(byte[] b, MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteArrayReader br = new ByteArrayReader(b, EndianOrder_e.big);
            DeserializeHeader(br);
            record_file_type = br.readUInt8();
            record_file_status = br.readUInt8();
            record_status = br.readUInt8();
            file_index = br.readInt32();
            for(int n = 0; n < 36; n++){
                record_guid[n] = br.readUInt8();
            }
            record_index = br.readUInt32();
            data_start_index = br.readUInt32();
            data_stop_index = br.readUInt32();

            return br.Idx;
        }
    }
}
