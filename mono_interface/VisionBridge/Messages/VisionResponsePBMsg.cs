using System;
using ProtoBuf;
using System.IO;

namespace VisionBridge.Messages
{
    [ProtoContract]
	/// <summary>
	/// Vision response PB message.
	/// This message is returned from the Vision System in response to 
	/// a VisionCommandPBMsg.
	/// The message returns a OK or Error flag, and optional Response Message
	/// string, and an optional Response Data.  The Data can be any serializable
	/// data including a serialized proto-buf message.
	/// </summary>
    public class VisionResponsePBMsg 
    {
        public enum ResponseType_e { OK = 0, ERROR = 1 }

        [ProtoMember(1)]
		/// <summary>
		/// The Command Response Type... 
		/// typically this is the only item returned unless there is an error.
		/// </summary>
		/// <value>The type of the cmd response.</value>
        public ResponseType_e CmdResponseType { get; set; }

        [ProtoMember(2)]
		/// <summary>
		/// A Response Message.
		/// It will be null if there is no response message.
		/// </summary>
		/// <value>The response message.</value>
        public string CmdResponseMessage { get; set;}

		[ProtoMember(3)]
		/// <summary>
		/// Optional Cmd Response data.
		/// It can be any serialized information including a protobuf message.
		/// </summary>
		/// <value>The cmd response data.</value>
		public byte[] CmdResponseData { get; set; }

		public VisionResponsePBMsg()
		{
			Clear();
		}

		public void Clear()
		{
			CmdResponseType = ResponseType_e.ERROR;
			CmdResponseMessage = null;
			CmdResponseData = null;
		}

        //----------------------------------------------------------------------
        //----------------------------------------------------------------------
        #region SerializeResponse
        /// <summary>
        /// Serialize VisionResponse to byte array.
        /// </summary>
        public byte[] Serialize()
        {
            byte[] b = null;
            using (var ms = new MemoryStream())
            {
                Serializer.Serialize<VisionResponsePBMsg>(ms, this);
                b = new byte[ms.Position];
                var fullB = ms.GetBuffer();
                Array.Copy(fullB, b, b.Length);
            }

            return b;
        }

        /// <summary>
        /// Deserialize to VisionResponse from byte array.
        /// </summary>
        /// <param name="b">The blue component.</param>
        public static VisionResponsePBMsg Deserialize(byte[] b)
        {
            VisionResponsePBMsg r;
            using (var ms = new MemoryStream(b))
            {
                r = Serializer.Deserialize<VisionResponsePBMsg>(ms);
            }
            return r;
        }
        #endregion
    }
}

