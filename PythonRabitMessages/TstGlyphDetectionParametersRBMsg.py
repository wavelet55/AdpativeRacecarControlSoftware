
# ****************************************************************
# *******************************************************************

# Auto-generated from file TstGlyphDetectionParametersRBMsg.msg, do not modify directly 


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



class TstGlyphDetectionParametersRBMsg(VSMessage):
    """ Auto generated message for TstGlyphDetectionParametersRBMsg
    """

    def __init__(self):
        super().__init__()

        self.fmt = '>'
        self.fmt += 'i '
        self.fmt += 'i '
        self.fmt += 'i '
        self.fmt += 'i '
        self.fmt += 'i '
        self.fmt += 'd '
        self.fmt += 'd '
        self.fmt += 'B '
        self.fmt += 'B '

        self.s = struct.Struct(self.fmt)

        self._msg = {}
        self.Clear()

    def Clear(self):

        self._msg['Canny_low'] = {"type": "int32", "value": 0 }
        self._msg['Canny_high'] = {"type": "int32", "value": 0 }
        self._msg['GlyphAreaPixels_min'] = {"type": "int32", "value": 0 }
        self._msg['GlyphAreaPixels_max'] = {"type": "int32", "value": 0 }
        self._msg['NumberOfIterations'] = {"type": "int32", "value": 0 }
        self._msg['ReprojectionErrorDistance'] = {"type": "float64", "value": 0 }
        self._msg['ConfidencePercent'] = {"type": "float64", "value": 0 }
        self._msg['GlyphDetectorImageDisplayType'] = {"type": "uint8", "value": 0 }
        self._msg['GlyphModelIndex'] = {"type": "uint8", "value": 0 }


    @property
    def Canny_low(self):
        return self._msg["Canny_low"]["value"]

    @Canny_low.setter
    def Canny_low(self, val):
        self._msg["Canny_low"]["value"] = val

    @property
    def Canny_high(self):
        return self._msg["Canny_high"]["value"]

    @Canny_high.setter
    def Canny_high(self, val):
        self._msg["Canny_high"]["value"] = val

    @property
    def GlyphAreaPixels_min(self):
        return self._msg["GlyphAreaPixels_min"]["value"]

    @GlyphAreaPixels_min.setter
    def GlyphAreaPixels_min(self, val):
        self._msg["GlyphAreaPixels_min"]["value"] = val

    @property
    def GlyphAreaPixels_max(self):
        return self._msg["GlyphAreaPixels_max"]["value"]

    @GlyphAreaPixels_max.setter
    def GlyphAreaPixels_max(self, val):
        self._msg["GlyphAreaPixels_max"]["value"] = val

    @property
    def NumberOfIterations(self):
        return self._msg["NumberOfIterations"]["value"]

    @NumberOfIterations.setter
    def NumberOfIterations(self, val):
        self._msg["NumberOfIterations"]["value"] = val

    @property
    def ReprojectionErrorDistance(self):
        return self._msg["ReprojectionErrorDistance"]["value"]

    @ReprojectionErrorDistance.setter
    def ReprojectionErrorDistance(self, val):
        self._msg["ReprojectionErrorDistance"]["value"] = val

    @property
    def ConfidencePercent(self):
        return self._msg["ConfidencePercent"]["value"]

    @ConfidencePercent.setter
    def ConfidencePercent(self, val):
        self._msg["ConfidencePercent"]["value"] = val

    @property
    def GlyphDetectorImageDisplayType(self):
        return self._msg["GlyphDetectorImageDisplayType"]["value"]

    @GlyphDetectorImageDisplayType.setter
    def GlyphDetectorImageDisplayType(self, val):
        self._msg["GlyphDetectorImageDisplayType"]["value"] = val

    @property
    def GlyphModelIndex(self):
        return self._msg["GlyphModelIndex"]["value"]

    @GlyphModelIndex.setter
    def GlyphModelIndex(self, val):
        self._msg["GlyphModelIndex"]["value"] = val



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

        self._msg['Canny_low']["value"] = vals[0]
        self._msg['Canny_high']["value"] = vals[1]
        self._msg['GlyphAreaPixels_min']["value"] = vals[2]
        self._msg['GlyphAreaPixels_max']["value"] = vals[3]
        self._msg['NumberOfIterations']["value"] = vals[4]
        self._msg['ReprojectionErrorDistance']["value"] = vals[5]
        self._msg['ConfidencePercent']["value"] = vals[6]
        self._msg['GlyphDetectorImageDisplayType']["value"] = vals[7]
        self._msg['GlyphModelIndex']["value"] = vals[8]

    