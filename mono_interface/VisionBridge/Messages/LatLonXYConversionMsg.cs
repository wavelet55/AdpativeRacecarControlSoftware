/* ****************************************************************
 * Vision Bridge
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Aug. 2015
 * 
 * Developed under contract for:
 * Academy Center for UAS Research 
 * Department of Electrical and Computer Engineering 
 * HQ USAFA/DFEC 
 * 2354 Fairchild Drive 
 * USAF Academy, CO 80840-6236 
 *******************************************************************/
using System;
using ProtoBuf;
using System.IO;

namespace VisionBridge.Messages
{

	[ProtoContract]
	/// <summary>
	/// The outside world can request a coordinate converion from
    /// the vision system.  This message provides both the Lat/Lon
    /// or X-Y info, and the response will fill in the coorisponting 
    /// values.
	/// </summary>
	public class LatLonXYConversionMsg
	{

		[ProtoMember(1)]
		/// <summary>
		/// If true convert from Lat/Lon to X-Y Coordinates
        /// If false convert from X-Y to Lat/Lon
		/// </summary>
		public bool LatLonToXYConversion { get; set; }

		[ProtoMember(2)]
		/// <summary>
		/// Latitude in Degrees [-90, 90]
		/// </summary>
		public double LatitudeDegrees { get; set; }

		[ProtoMember(3)]
		/// <summary>
		/// Longitude in Degrees [-180, 180]
		/// </summary>
		public double LongitudeDegrees { get; set; }

		[ProtoMember(4)]
		/// <summary>
        /// X (East-West) Value Meters
		/// </summary>
		public double X_PosMeters { get; set; }

		[ProtoMember(5)]
		/// <summary>
        /// Y (North-South) Value Meters
		/// </summary>
		public double Y_PosMeters { get; set; }

		public LatLonXYConversionMsg()
		{
			Clear();
		}

		public void Clear()
		{
			LatLonToXYConversion = false;
			LatitudeDegrees = 0;
            LongitudeDegrees = 0;
            X_PosMeters = 0;
            Y_PosMeters = 0;
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
				Serializer.Serialize<LatLonXYConversionMsg>(ms, this);
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
		public static LatLonXYConversionMsg Deserialize(byte[] b)
		{
			LatLonXYConversionMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<LatLonXYConversionMsg>(ms);
			}
			return r;
		}
		#endregion

	}
}


