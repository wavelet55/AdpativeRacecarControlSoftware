
# ****************************************************************
# *******************************************************************

# Auto-generated from file VidereRecordStatusMsg.msg, do not modify directly 


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



class VidereRecordStatusMsg(VSMessage):
    """ Auto generated message for VidereRecordStatusMsg
    """

    def __init__(self):
        super().__init__()

        self.fmt = '>'
        self.fmt += 'B '
        self.fmt += 'B '
        self.fmt += 'B '
        self.fmt += 'i '
        self.fmt += '36B '
        self.fmt += 'I '
        self.fmt += 'I '
        self.fmt += 'I '

        self.s = struct.Struct(self.fmt)

        self._msg = {}
        self.Clear()

    def Clear(self):

        self._msg['record_file_type'] = {"type": "uint8", "value": 0 }
        self._msg['record_file_status'] = {"type": "uint8", "value": 0 }
        self._msg['record_status'] = {"type": "uint8", "value": 0 }
        self._msg['file_index'] = {"type": "int32", "value": 0 }
        self._msg['record_guid'] = {"type": "uint8", "length": 36 ,"value": [0]*36 }
        self._msg['record_index'] = {"type": "uint32", "value": 0 }
        self._msg['data_start_index'] = {"type": "uint32", "value": 0 }
        self._msg['data_stop_index'] = {"type": "uint32", "value": 0 }


    @property
    def record_file_type(self):
        return self._msg["record_file_type"]["value"]

    @record_file_type.setter
    def record_file_type(self, val):
        self._msg["record_file_type"]["value"] = val

    @property
    def record_file_status(self):
        return self._msg["record_file_status"]["value"]

    @record_file_status.setter
    def record_file_status(self, val):
        self._msg["record_file_status"]["value"] = val

    @property
    def record_status(self):
        return self._msg["record_status"]["value"]

    @record_status.setter
    def record_status(self, val):
        self._msg["record_status"]["value"] = val

    @property
    def file_index(self):
        return self._msg["file_index"]["value"]

    @file_index.setter
    def file_index(self, val):
        self._msg["file_index"]["value"] = val

    @property
    def record_guid(self):
        return self._msg["record_guid"]["value"]

    @record_guid.setter
    def record_guid(self, val):
        if type(val) is (list or tuple):
            if len(val) == self._msg["record_guid"]["length"]:
                self._msg["record_guid"]["value"] = val
            else:
                raise ValueError("record_guid must be a list of length 36")
        else:
            raise ValueError("record_guid must be a list of length 36")

    @property
    def record_index(self):
        return self._msg["record_index"]["value"]

    @record_index.setter
    def record_index(self, val):
        self._msg["record_index"]["value"] = val

    @property
    def data_start_index(self):
        return self._msg["data_start_index"]["value"]

    @data_start_index.setter
    def data_start_index(self, val):
        self._msg["data_start_index"]["value"] = val

    @property
    def data_stop_index(self):
        return self._msg["data_stop_index"]["value"]

    @data_stop_index.setter
    def data_stop_index(self, val):
        self._msg["data_stop_index"]["value"] = val



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

        self._msg['record_file_type']["value"] = vals[0]
        self._msg['record_file_status']["value"] = vals[1]
        self._msg['record_status']["value"] = vals[2]
        self._msg['file_index']["value"] = vals[3]
        self._msg['record_guid']["value"] = vals[4:40]
        self._msg['record_index']["value"] = vals[40]
        self._msg['data_start_index']["value"] = vals[41]
        self._msg['data_stop_index']["value"] = vals[42]

    