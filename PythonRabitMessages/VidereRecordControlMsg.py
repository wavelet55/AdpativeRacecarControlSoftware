
# ****************************************************************
# *******************************************************************

# Auto-generated from file VidereRecordControlMsg.msg, do not modify directly 


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



class VidereRecordControlMsg(VSMessage):
    """ Auto generated message for VidereRecordControlMsg
    """

    def __init__(self):
        super().__init__()

        self.fmt = '>'
        self.fmt += 'B '
        self.fmt += 'i '
        self.fmt += '36B '
        self.fmt += 'I '
        self.fmt += '36B '
        self.fmt += 'B '

        self.s = struct.Struct(self.fmt)

        self._msg = {}
        self.Clear()

    def Clear(self):

        self._msg['command'] = {"type": "uint8", "value": 0 }
        self._msg['file_index'] = {"type": "int32", "value": 0 }
        self._msg['file_guid'] = {"type": "uint8", "length": 36 ,"value": [0]*36 }
        self._msg['record_version'] = {"type": "uint32", "value": 0 }
        self._msg['exercise_guid'] = {"type": "uint8", "length": 36 ,"value": [0]*36 }
        self._msg['exercise_status'] = {"type": "uint8", "value": 0 }


    @property
    def command(self):
        return self._msg["command"]["value"]

    @command.setter
    def command(self, val):
        self._msg["command"]["value"] = val

    @property
    def file_index(self):
        return self._msg["file_index"]["value"]

    @file_index.setter
    def file_index(self, val):
        self._msg["file_index"]["value"] = val

    @property
    def file_guid(self):
        return self._msg["file_guid"]["value"]

    @file_guid.setter
    def file_guid(self, val):
        if type(val) is (list or tuple):
            if len(val) == self._msg["file_guid"]["length"]:
                self._msg["file_guid"]["value"] = val
            else:
                raise ValueError("file_guid must be a list of length 36")
        else:
            raise ValueError("file_guid must be a list of length 36")

    @property
    def record_version(self):
        return self._msg["record_version"]["value"]

    @record_version.setter
    def record_version(self, val):
        self._msg["record_version"]["value"] = val

    @property
    def exercise_guid(self):
        return self._msg["exercise_guid"]["value"]

    @exercise_guid.setter
    def exercise_guid(self, val):
        if type(val) is (list or tuple):
            if len(val) == self._msg["exercise_guid"]["length"]:
                self._msg["exercise_guid"]["value"] = val
            else:
                raise ValueError("exercise_guid must be a list of length 36")
        else:
            raise ValueError("exercise_guid must be a list of length 36")

    @property
    def exercise_status(self):
        return self._msg["exercise_status"]["value"]

    @exercise_status.setter
    def exercise_status(self, val):
        self._msg["exercise_status"]["value"] = val



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

        self._msg['command']["value"] = vals[0]
        self._msg['file_index']["value"] = vals[1]
        self._msg['file_guid']["value"] = vals[2:38]
        self._msg['record_version']["value"] = vals[38]
        self._msg['exercise_guid']["value"] = vals[39:75]
        self._msg['exercise_status']["value"] = vals[75]

    