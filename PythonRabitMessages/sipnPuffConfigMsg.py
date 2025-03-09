
# ****************************************************************
# *******************************************************************

# Auto-generated from file sipnPuffConfigMsg.msg, do not modify directly 


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



class sipnPuffConfigMsg(VSMessage):
    """ Auto generated message for sipnPuffConfigMsg
    """

    def __init__(self):
        super().__init__()

        self.fmt = '>'
        self.fmt += '? '
        self.fmt += 'd '
        self.fmt += 'd '
        self.fmt += 'd '

        self.s = struct.Struct(self.fmt)

        self._msg = {}
        self.Clear()

    def Clear(self):

        self._msg['EnableSipnPuffIntegration'] = {"type": "bool", "value": False }
        self._msg['SipnPuffBlowGain'] = {"type": "float64", "value": 0 }
        self._msg['SipnPuffSuckGain'] = {"type": "float64", "value": 0 }
        self._msg['SipnPuffDeadBandPercent'] = {"type": "float64", "value": 0 }


    @property
    def EnableSipnPuffIntegration(self):
        return self._msg["EnableSipnPuffIntegration"]["value"]

    @EnableSipnPuffIntegration.setter
    def EnableSipnPuffIntegration(self, val):
        self._msg["EnableSipnPuffIntegration"]["value"] = val

    @property
    def SipnPuffBlowGain(self):
        return self._msg["SipnPuffBlowGain"]["value"]

    @SipnPuffBlowGain.setter
    def SipnPuffBlowGain(self, val):
        self._msg["SipnPuffBlowGain"]["value"] = val

    @property
    def SipnPuffSuckGain(self):
        return self._msg["SipnPuffSuckGain"]["value"]

    @SipnPuffSuckGain.setter
    def SipnPuffSuckGain(self, val):
        self._msg["SipnPuffSuckGain"]["value"] = val

    @property
    def SipnPuffDeadBandPercent(self):
        return self._msg["SipnPuffDeadBandPercent"]["value"]

    @SipnPuffDeadBandPercent.setter
    def SipnPuffDeadBandPercent(self, val):
        self._msg["SipnPuffDeadBandPercent"]["value"] = val



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

        self._msg['EnableSipnPuffIntegration']["value"] = vals[0]
        self._msg['SipnPuffBlowGain']["value"] = vals[1]
        self._msg['SipnPuffSuckGain']["value"] = vals[2]
        self._msg['SipnPuffDeadBandPercent']["value"] = vals[3]

    