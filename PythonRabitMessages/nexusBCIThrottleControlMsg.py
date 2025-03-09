
# ****************************************************************
# *******************************************************************

# Auto-generated from file nexusBCIThrottleControlMsg.msg, do not modify directly 


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



class nexusBCIThrottleControlMsg(VSMessage):
    """ Auto generated message for nexusBCIThrottleControlMsg
    """

    def __init__(self):
        super().__init__()

        self.fmt = '>'
        self.fmt += '? '

        self.s = struct.Struct(self.fmt)

        self._msg = {}
        self.Clear()

    def Clear(self):

        self._msg['ThrottleOn'] = {"type": "bool", "value": False }


    @property
    def ThrottleOn(self):
        return self._msg["ThrottleOn"]["value"]

    @ThrottleOn.setter
    def ThrottleOn(self, val):
        self._msg["ThrottleOn"]["value"] = val



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

        self._msg['ThrottleOn']["value"] = vals[0]

    