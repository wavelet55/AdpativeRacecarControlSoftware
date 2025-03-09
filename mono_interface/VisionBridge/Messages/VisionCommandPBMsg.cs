using System;
using ProtoBuf;
using System.IO;

namespace VisionBridge.Messages
{

    [ProtoContract]
	/// <summary>
	/// The Vision Command is part of a Command/Response message pair 
	/// to the Vision System.  The main "Command" is contained in the command parameter
	/// which can be any command the Vision system knows how to process. Often all that
	/// is in the command is the command string such as:  
	/// 	"StartVision"
	/// 	"StopVision"
	/// 	"GPUEnable"
	/// 	"GPUDisable"
	/// 
	/// The CmdQualifier [Optional) is a string that can be used to qualify or provide extra
	/// parameters to the the command.
	/// 
	/// The CmdData [Optional] is any serializable byte array... it could contain a serialized
	/// proto-buf message... a JSON message... or most anything.  
	/// 
	/// The Vision system will respond to a Command Message with a VisionResponse Message.
	/// </summary>
    public class VisionCommandPBMsg
    {
        [ProtoMember(1)]
		/// <summary>
		/// A Command in String format.  Commands are case insensitive.
		/// </summary>
		/// <value>The command.</value>
        public string Command { get; set; }

        [ProtoMember(2)]
		/// <summary>
		/// A Command Qualifier
		/// This is an optional parameter to the command.
		/// </summary>
		/// <value>The cmd qualifier, null if no qualifier</value>
        public string CmdQualifier { get; set; }

		[ProtoMember(3)]
		/// <summary>
		/// Command Data
		/// Optional data that can be any serialized set of parameters
		/// or data.
		/// </summary>
		/// <value>The cmd data, null if no data.</value>
		public byte[] CmdData { get; set; }


		public VisionCommandPBMsg()
		{
			Clear();
		}

		public void Clear()
		{
			Command = "Unknown";
			CmdQualifier = null;
			CmdData = null;
		}

        //----------------------------------------------------------------------
        //----------------------------------------------------------------------
        #region SerializeCommand
        /// <summary>
        /// Serialize VisionCommand to byte array.
        /// </summary>
        public byte[] Serialize()
        {
            byte[] b = null;
            using (var ms = new MemoryStream())
            {
                Serializer.Serialize<VisionCommandPBMsg>(ms, this);
                b = new byte[ms.Position];
                var fullB = ms.GetBuffer();
                Array.Copy(fullB, b, b.Length);
            }

            return b;
        }

        /// <summary>
        /// Deserialize to VisionCommand from byte array.
        /// </summary>
        /// <param name="b">The blue component.</param>
        public static VisionCommandPBMsg Deserialize(byte[] b)
        {
            VisionCommandPBMsg c;
            using (var ms = new MemoryStream(b))
            {
                c = Serializer.Deserialize<VisionCommandPBMsg>(ms);
            }
            return c;
        }
        #endregion
    }
}

