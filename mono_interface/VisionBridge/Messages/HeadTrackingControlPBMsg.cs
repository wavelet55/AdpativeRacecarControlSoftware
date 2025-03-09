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
	/// Head Tracking Control Message.
	/// </summary>
	public class HeadTrackingControlPBMsg
	{
		[ProtoMember(1)]
		public int Canny_low { get; set; }	

		[ProtoMember(2)]
		public int Canny_high { get; set; }
	
		[ProtoMember(3)]
		public int GlyphAreaPixels_min { get; set; }
	
		[ProtoMember(4)]
		public int GlyphAreaPixels_max { get; set; }	
	
		[ProtoMember(5)]
		public int NumberOfIterations { get; set; }	
	
		[ProtoMember(6)]
		public double ReprojectionErrorDistance { get; set; }	

		[ProtoMember(7)]
		public double ConfidencePercent { get; set; }	

		[ProtoMember(8)]
		public UInt32 HeadTrackingDisplayType { get; set; }	

        [ProtoMember(9)]
		public UInt32 GlyphModelIndex { get; set; }	


		public HeadTrackingControlPBMsg()
		{
			Clear();
		}

		public void Clear()
		{
			Canny_low = 50;
            Canny_high = 150;
            GlyphAreaPixels_min = 1000;
            GlyphAreaPixels_max = 8000;
            NumberOfIterations = 10;
            ReprojectionErrorDistance = 5.0;
			ConfidencePercent = 95.0;
            HeadTrackingDisplayType = 0;
            GlyphModelIndex = 0;
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
				Serializer.Serialize<HeadTrackingControlPBMsg>(ms, this);
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
		public static HeadTrackingControlPBMsg Deserialize(byte[] b)
		{
			HeadTrackingControlPBMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<HeadTrackingControlPBMsg>(ms);
			}
			return r;
		}
		#endregion


	}
}

