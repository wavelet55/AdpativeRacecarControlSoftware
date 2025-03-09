/* ****************************************************************
 * Vision Bridge
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: June 2018
 * 
 *******************************************************************/
using System;
using ProtoBuf;
using System.IO;

namespace VisionBridge.Messages
{
	
	[ProtoContract]
	/// <summary>
	/// Steering Control and Monitor Message.
	/// </summary>
	public class HeadOrientationPBMsg
	{
		[ProtoMember(1)]
		public double HeadRollDegrees { get; set; }	
	
		[ProtoMember(2)]
		public double HeadPitchDegrees { get; set; }	

        [ProtoMember(3)]
		public double HeadYawDegrees { get; set; }	


		public HeadOrientationPBMsg()
		{
			Clear();
		}

		public void Clear()
		{
			HeadRollDegrees = 0;
			HeadPitchDegrees = 0;
			HeadYawDegrees = 0;
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
				Serializer.Serialize<HeadOrientationPBMsg>(ms, this);
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
		public static HeadOrientationPBMsg Deserialize(byte[] b)
		{
			HeadOrientationPBMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<HeadOrientationPBMsg>(ms);
			}
			return r;
		}
		#endregion


	}
}

