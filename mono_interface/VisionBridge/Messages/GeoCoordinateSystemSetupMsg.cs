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
	/// Set up the Geo-Coordinate System
    /// The Geo-Coordinate system is responsible for conversions
    /// between Lat/Lon and X-Y coordintes.
    /// Pass in the Center Lat/Lon of the area the mission is being
    /// flown/run in and the type of conversion to use.
	/// </summary>
	public class GeoCoordinateSystemSetupMsg
	{
        public enum GeoCoordinateSystemConversionType_e
        {
            /// <summary>
            /// For small areas < +/2 2/5 km
            /// This is the fastest type of conversion as it only requires
            /// a simple scaling between Lat/Lon and X-Y
            /// </summary>
            Linear = 0,
             
            /// <summary>
            /// Preferred for Larger Areas.
            /// Based upon WGS84 Conversions relative to a Reference location.
            /// X-Y zero (0,0) is at the provide Lat/Lon Reference location
            /// Does not have issues with Map Boundarys or crossing the equator
            /// </summary>
            WGS84_Relative = 1, 
 
            /// <summary>
            /// Provides X-Y Coordinates that are established by the 
            /// WGS-84 Mapping standards.
            /// Warning!!! There are hugh step changes at map boundaries
            /// and at the equator.  Do not used this conversion if there
            /// are any chances of crossing from one WGS-84 map boundary to another.
            /// For this reason... I highly recommend using the WGS84_Relative option.
            /// </summary>
            WGS84_Map = 2
        }

		[ProtoMember(1)]
		/// <summary>
		/// GeoCoordinateSystemConversionType
		/// </summary>
		public GeoCoordinateSystemConversionType_e GeoCoordinateSystemConversionType { get; set; }

		[ProtoMember(2)]
		/// <summary>
		/// Center of operation Latitude in Degrees [-90, 90]
		/// </summary>
		public double CenterLatitudeDegrees { get; set; }

		[ProtoMember(3)]
		/// <summary>
		/// Center of operation Longitude in Degrees [-180, 180]
		/// </summary>
		public double CenterLongitudeDegrees { get; set; }

		[ProtoMember(4)]
		/// <summary>
		/// Ground Altitude in Meters above mean sea level.
		/// </summary>
		public double GroundAltitudeMSL { get; set; }

		[ProtoMember(5)]
		/// <summary>
		/// For Linear Conversion Type the DeltaLatitudeDegrees
        /// is the expected positive range of Latitudes around the 
        /// CenterLatitude the vehicle is expected to operate in.
        /// The full range will be CenterLatitude +/- DeltaLatitude.
		/// </summary>
		public double DeltaLatitudeDegrees { get; set; }

		[ProtoMember(6)]
		/// <summary>
		/// For Linear Conversion Type the DeltaLongitudeDegrees
        /// is the expected positive range of Longitude around the 
        /// CenterLongitude the vehicle is expected to operate in.
        /// The full range will be CenterLongitude +/- DeltaLongitude.
		/// </summary>
		public double DeltaLongitudeDegrees { get; set; }

		public GeoCoordinateSystemSetupMsg()
		{
			Clear();
		}


		public void Clear()
		{
			GeoCoordinateSystemConversionType = GeoCoordinateSystemConversionType_e.Linear;
			CenterLatitudeDegrees = 0;
            CenterLongitudeDegrees = 0;
            GroundAltitudeMSL = 0;
            DeltaLatitudeDegrees = 0;
            DeltaLongitudeDegrees = 0;
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
				Serializer.Serialize<GeoCoordinateSystemSetupMsg>(ms, this);
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
		public static GeoCoordinateSystemSetupMsg Deserialize(byte[] b)
		{
			GeoCoordinateSystemSetupMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<GeoCoordinateSystemSetupMsg>(ms);
			}
			return r;
		}
		#endregion

	}
}

