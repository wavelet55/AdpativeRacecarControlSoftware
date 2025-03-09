using System;
using ProtoBuf;
using System.IO;

namespace VisionBridge.Messages
{

    [ProtoContract]
    public class VisionBroadcast
    {

        [ProtoMember(1)]
        public string message_string { get; set; }

        [ProtoMember(2)]
        public bool is_exception { get; set; }

		[ProtoMember(3)]
		public byte[] report { get; set; }

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
                Serializer.Serialize<VisionBroadcast>(ms, this);
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
        public static VisionBroadcast Deserialize(byte[] b)
        {
            VisionBroadcast c;
            using (var ms = new MemoryStream(b))
            {
                c = Serializer.Deserialize<VisionBroadcast>(ms);
            }
            return c;
        }
        #endregion
    }
}

