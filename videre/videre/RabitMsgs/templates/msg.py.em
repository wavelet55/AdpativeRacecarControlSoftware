@###############################################
@#
@# Message generation for Python
@#
@# EmPy template for generating <msg>.py
@#
@###############################################
@# Start of Template
@###############################################

# ****************************************************************
# *******************************************************************

# Auto-generated from file @file_name_in, do not modify directly 

@{
import genmsg.msgs
from rabit_msg_helper import *

msg_name = spec.short_name
}@

@{
def format_specifier(fields):
    for field in fields:
        if (field.is_builtin):
            if (not field.is_array):
                print("        self.fmt += '{} '".format(str(type_map_py[field.base_type])) )
            else:
                print("        self.fmt += '{}{} '".format(str(field.array_len), str(type_map_py[field.base_type])))
                
        else:
            print("not here")

def clear_variables(fields):
    for field in fields:
        if (field.is_builtin):
            if (not field.is_array):
                print("        self._msg['{}'] = {{\"type\": \"{}\", \"value\": {} }}".format(str(field.name), str(field.base_type), init_py[field.base_type]) )
            else:
                print("        self._msg['{}'] = {{\"type\": \"{}\", \"length\": {} ,\"value\": [{}]*{} }}".format(str(field.name), str(field.base_type), str(field.array_len), str(init_py[field.base_type]), str(field.array_len)))
        else:
            print("not here")

def getters_and_setter(fields):
    for field in fields:
        if (field.is_builtin):
            if (not field.is_array):
                print("    @property")
                print("    def {}(self):".format(str(field.name)))
                print("        return self._msg[\"{}\"][\"value\"]".format(str(field.name)))
                print("")
                print("    @{}.setter".format(str(field.name)))
                print("    def {}(self, val):".format(str(field.name)))
                print("        self._msg[\"{}\"][\"value\"] = val".format(str(field.name)))
                print("")

            else:
                print("    @property")
                print("    def {}(self):".format(str(field.name)))
                print("        return self._msg[\"{}\"][\"value\"]".format(str(field.name)))
                print("")
                print("    @{}.setter".format(str(field.name)))
                print("    def {}(self, val):".format(str(field.name)))
                print("        if type(val) is (list or tuple):")
                print("            if len(val) == self._msg[\"{}\"][\"length\"]:".format(str(field.name)))
                print("                self._msg[\"{}\"][\"value\"] = val".format(str(field.name)))
                print("            else:")
                print("                raise ValueError(\"{} must be a list of length {}\")".format(str(field.name), str(field.array_len)))
                print("        else:")
                print("            raise ValueError(\"{} must be a list of length {}\")".format(str(field.name), str(field.array_len)))
                print("")
        else:
                print("not here")

def deserialize_variables(fields):
    idx = 0
    for field in fields:
        if (field.is_builtin):
            if (not field.is_array):
                print("        self._msg['{}'][\"value\"] = vals[{}]".format(str(field.name), idx) )
                idx += 1
            else:
                print("        self._msg['{}'][\"value\"] = vals[{}:{}]".format(str(field.name), idx, str(idx + field.array_len)) )
                idx += field.array_len
        else:
                print("not here")
        

}@
import struct

class VSMessage(object):
    """ This is a small enough base class that we are including it in all messages

    Including this in all messages means each python message doesn't have any dependencies
    """

    hfmt = '>iiid'

    def __init__(self):
        self.MessageType = 0
        self.MessageSource = 0
        self.MessageDestination = 0
        self.TimeStampSec = 0
        self.hs = struct.Struct(self.hfmt)

    def getMsgHeaderSizeInBytes(self):
        return 3*4 + 8   # 3 * sizeof(Int32) + sizeof(double);

    def SerializeHeader(self):
        data = (self.MessageType, self.MessageSource, self.MessageDestination, self.TimeStampSec)
        return self.hs.pack(*data)
        
    def Serialize(self, serial_type = "DtiByteArray"):
        return

    def DeserializeHeader(self, byte_array):
        vals = self.hs.unpack(byte_array)
        self.MessageType = vals[0]
        self.MessageSource = vals[1]
        self.MessageDestination = vals[2]
        self.TimeStampSec = vals[3]

    def Deserialize(self, byte_array, serial_type = "DtiByteArray"):
        return



class @(msg_name)(VSMessage):
    """ Auto generated message for @(msg_name)
    """

    def __init__(self):
        super().__init__()

        self.fmt = '>'
@format_specifier(spec.parsed_fields())
        self.s = struct.Struct(self.fmt)

        self._msg = {}
        self.Clear()

    def Clear(self):

@clear_variables(spec.parsed_fields())

@getters_and_setter(spec.parsed_fields())

    def __str__(self):
        msg = [self.MessageType, self.MessageSource, self.MessageDestination, self.TimeStampSec, self._msg]
        return str(msg)

    def ToString(self):
        return str(self)

    def Serialize(self, type = "DtiByteArray"):

        byte_header = self.SerializeHeader()

        values = []
        for key in self._msg:
            if "length" in self._msg[key]:
                for val in self._msg[key]["value"]:
                    values.append(val) 
            else:
                values.append(self._msg[key]["value"])

        byte_body = self.s.pack(*values)

        barray = bytearray(byte_header) + bytearray(byte_body)

        return bytes(barray)

    def Deserialize(self, byte_array, type = "DtiByteArray"):
        
        length = self.getMsgHeaderSizeInBytes()
        header = byte_array[0:length]
        body = byte_array[length:]
        self.DeserializeHeader(header)
        vals = self.s.unpack(body)

@deserialize_variables(spec.parsed_fields())
    