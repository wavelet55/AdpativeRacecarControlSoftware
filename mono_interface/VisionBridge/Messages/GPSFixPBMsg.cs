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
	public class GPSFixPBMsg
	{

		[ProtoMember(1)]
        public Int32  TrackingSatellites { get; set; }
	
		[ProtoMember(2)]
        public double LatitudeDeg { get; set; }	// degrees N
	
		[ProtoMember(3)]
        public double LongitudeDeg { get; set; }	// degrees E
	
		[ProtoMember(4)]
        public double AltitudeMSL { get; set; }	// meters above mean sea level

	
		[ProtoMember(5)]
        public double Position_X { get; set; }     //Meters
	
		[ProtoMember(6)]
        public double Position_Y { get; set; }
	
		[ProtoMember(7)]
        public double Position_Z { get; set; }

	
		[ProtoMember(8)]
        public double Velocity_X { get; set; }     //Meters / Second
	
		[ProtoMember(9)]
        public double Velocity_Y { get; set; }
	
		[ProtoMember(10)]
        public double Velocity_Z { get; set; }    //Meters above reference point

	
		[ProtoMember(11)]
        public double GPSTimeStampSec { get; set; }  //Videre
	
		[ProtoMember(12)]
        public double VidereTimeStampSec { get; set; }  //Videre


		public GPSFixPBMsg()
		{
			Clear();
		}

		public void Clear()
		{
            TrackingSatellites = 0;
            LatitudeDeg = 0;	// degrees N
            LongitudeDeg = 0;	// degrees E
            AltitudeMSL = 0;	// meters above mean sea level

            Position_X = 0;     //Meters
            Position_Y = 0;
            Position_Z = 0;

            Velocity_X = 0;     //Meters / Second
            Velocity_Y = 0;
            Velocity_Z = 0;    //Meters above reference point

            GPSTimeStampSec = 0;  //Videre
            VidereTimeStampSec = 0;  //Videre
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
				Serializer.Serialize<GPSFixPBMsg>(ms, this);
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
		public static GPSFixPBMsg Deserialize(byte[] b)
		{
			GPSFixPBMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<GPSFixPBMsg>(ms);
			}
			return r;
		}
		#endregion

	}
}


