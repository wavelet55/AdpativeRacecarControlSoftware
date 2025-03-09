@###############################################
@#
@# Message generation for C#
@#
@# EmPy template for generating <msg>.cs
@#
@###############################################
@# Start of Template
@###############################################

/* ****************************************************************
  *******************************************************************/

/* Auto-generated from file @file_name_in, do not modify directly */

@{
import genmsg.msgs
from rabit_msg_helper import *

msg_name = spec.short_name
}@

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Int8 = System.SByte;
using UInt8 = System.Byte;

@{
def list_variables(fields):
    for field in fields:
        if (field.is_builtin):
            if (not field.is_array):
                print("        public " + str(type_map_cs[field.type]) + " " + str(field.name) + " { get; set;}")
            else:
                print("        public {}[] {} = new {}[{}];".format(str(type_map_cs[field.base_type]), str(field.name), str(type_map_cs[field.base_type]), str(field.array_len)) )
        else:
            print("not here")

def clear_variables(fields):
    for field in fields:
        if (field.is_builtin):
            if (not field.is_array):
                print("            {} = {};".format(field.name, init_cs[field.base_type]))
            else:
                print("            Array.Clear( {}, 0, {});".format( str(field.name), str(field.array_len)))
        else:
            print("not here")

def serialize_variables(fields):
    for field in fields:
        if (field.is_builtin):
            if (not field.is_array):
                print("            bw.write{}({});".format(str(type_byte_writer_map[field.type]), str(field.name)))
            else:
                print("            for(int n = 0; n < {}; n++){{".format(str(field.array_len)))
                print("                bw.write{}({}[n]);".format(str(type_byte_writer_map[field.base_type]), str(field.name)))
                print("            }")

        else:
                print("not here")

def deserialize_variables(fields):
    for field in fields:
        if (field.is_builtin):
            if (not field.is_array):
                print("            {} = br.read{}();".format(str(field.name), str(type_byte_writer_map[field.type])))
            else:
                print("            for(int n = 0; n < {}; n++){{".format(str(field.array_len)))
                print("                {}[n] = br.read{}();".format(str(field.name), str(type_byte_writer_map[field.base_type])))
                print("            }")
        else:
                print("not here")


}@

namespace @(nm_space)
{
    public class @(msg_name) : VSMessage
    {

@list_variables(spec.parsed_fields())
       
        public @(msg_name)()
        {
            Clear();
        }

        public override void Clear()
        {
            base.Clear();
@clear_variables(spec.parsed_fields())
        }


        public override byte[] Serialize(MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteListArrayWriter bw = new ByteListArrayWriter(EndianOrder_e.big);
            SerializeHeader(bw);
@serialize_variables(spec.parsed_fields())
            return bw.ByteArray;
        }

        public override int Deserialize(byte[] b, MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteArrayReader br = new ByteArrayReader(b, EndianOrder_e.big);
            DeserializeHeader(br);
@deserialize_variables(spec.parsed_fields())
            return br.Idx;
        }
    }
}
