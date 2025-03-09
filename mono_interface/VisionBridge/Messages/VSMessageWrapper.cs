using System;
using ProtoBuf;
using System.IO;

namespace VisionBridge.Messages
{

    [ProtoContract]
	/// <summary>
	/// This message is a wrapper for all the messages sent to or coming from
    /// the Vision System.  The message provides a MessageName, a Qualifier
    /// and the message data payload.  The Message Name provided must be unique
    /// to the message type so that the recieving end will know how to process
    /// the message payload.  The qualifier is an optional parameter that can be
    /// used to provide additional information about the message.
    /// The data payload is a byte array that can contain most anything.  The 
    /// data payload could be a protobuf message, a JSON message, an Image, 
    /// or anything else that will can be serialized it the byte array.  
    /// The data payload size gives the number of bytes contained in the data array.
    /// The data payload can be empty... indicated by the data size of zero.
    /// 
    /// Note: The Vision Command Response messages do not use this message wrapper,
    /// see:  VisionCommandPBMsg and VisionResponsePBMsg
	/// </summary>
    public class VSMessageWrapper
    {
        [ProtoMember(1)]
		/// <summary>
		/// A Message Name... case sensitive... It is best to 
        /// use CamelCase names such as:  VehicleInertialStates.
		/// </summary>
		/// <value>The command.</value>
        public string MsgName { get; set; }

        [ProtoMember(2)]
		/// <summary>
		/// A Message Qualifier
		/// This is an optional parameter to the message and will be 
        /// message specific.
		/// </summary>
		/// <value>The cmd qualifier, null if no qualifier</value>
        public string MsgQualifier { get; set; }

		[ProtoMember(3)]
		/// <summary>
		/// Msg Data
		/// Optional data that can be any serialized set of parameters
		/// or data.
		/// </summary>
		/// <value>The cmd data, null if no data.</value>
		public byte[] MsgData { get; set; }

        /// <summary>
        /// The number of bytes in the MsgData.
        /// The size should be set to zero if there is not MsgData
        /// </summary>
        public int MsgDataSize { get; set; }


		public VSMessageWrapper()
		{
			Clear();
		}

		public void Clear()
		{
			MsgName = "Unknown";
			MsgQualifier = null;
			MsgData = null;
            MsgDataSize = 0;
		}

        //----------------------------------------------------------------------
        //----------------------------------------------------------------------
        #region SerializeCommand
        /// <summary>
        /// Serialize Message Wrapper to byte array.
        /// </summary>
        public byte[] Serialize()
        {
            byte[] b = null;
            using (var ms = new MemoryStream())
            {
                Serializer.Serialize<VSMessageWrapper>(ms, this);
                b = new byte[ms.Position];
                var fullB = ms.GetBuffer();
                Array.Copy(fullB, b, b.Length);
            }

            return b;
        }

        /// <summary>
        /// Deserialize the Message Wrapper from a byte array.
        /// </summary>
        /// <param name="b">The blue component.</param>
        public static VSMessageWrapper Deserialize(byte[] b)
        {
            VSMessageWrapper c;
            using (var ms = new MemoryStream(b))
            {
                c = Serializer.Deserialize<VSMessageWrapper>(ms);
            }
            return c;
        }
        #endregion
    }
}

