using System;
using ProtoBuf;
using System.IO;
using System.Text;

namespace VisionBridge.Messages
{
	[ProtoContract]
	public class VisionSimpleFilter
	{
		[ProtoMember(1)]
		public string info { get; set; }

		[ProtoMember(2)]
		public byte[] image_jpeg { get; set; }

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
				Serializer.Serialize<VisionSimpleFilter>(ms, this);
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
		public static VisionSimpleFilter Deserialize(byte[] b)
		{
			VisionSimpleFilter c;
			using (var ms = new MemoryStream(b))
			{
				c = Serializer.Deserialize<VisionSimpleFilter>(ms);
			}
			return c;
		}
		#endregion
	}
}

