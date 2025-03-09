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
	public class SipAndPuffPBMsg
	{
		[ProtoMember(1)]
		public double SipAndPuffPercent { get; set; }	
	
		[ProtoMember(2)]
		public double SipAndPuffIntegralPercent { get; set; }	

		public SipAndPuffPBMsg()
		{
			Clear();
		}

		public void Clear()
		{
			SipAndPuffPercent = 0;
			SipAndPuffIntegralPercent = 0;
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
				Serializer.Serialize<SipAndPuffPBMsg>(ms, this);
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
		public static SipAndPuffPBMsg Deserialize(byte[] b)
		{
			SipAndPuffPBMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<SipAndPuffPBMsg>(ms);
			}
			return r;
		}
		#endregion


	}
}


