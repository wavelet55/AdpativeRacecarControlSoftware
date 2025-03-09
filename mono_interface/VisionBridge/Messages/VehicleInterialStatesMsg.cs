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
	/// Vehicle interial states message.
	/// Contains all the Vehical Interial States, position, attitude and rates.
	/// All distances are in meters, and all angles are in radians, unless otherwise
	/// noted.
	/// The X-Y Axis positions are relative positions derived from the GPS Lat/Lon position
	/// based upon an intenal X-Y axix system.
	/// </summary>
	public class VehicleInterialStatesMsg
	{
		[ProtoMember(1)]
		/// <summary>
		/// If CoordinatesLatLon is true, the position coordinates
		/// are in Latitude and Longitude.
		/// If CoordinatesLatLon is false, the position coordinates
		/// are passed as X-Y position data.
		/// </summary>
		public bool CoordinatesLatLonOrXY { get; set; }

		[ProtoMember(2)]
		/// <summary>
		/// Vehicle Latitude in Radians
		/// or Y position in meters
		/// </summary>
		public double LatitudeRadOrY { get; set; }		//In Radians

		[ProtoMember(3)]
		/// <summary>
		/// Vehicle Longitude in Radians
		/// or X position in meters
		/// </summary>
		public double LongitudeRadOrX { get; set; }     //In Radians

		[ProtoMember(4)]
		/// <summary>
		/// Altitude in Meters above Mean Sea Level
		/// </summary>
		public double AltitudeMSL { get; set; }      //In Meters above mean sea level in meters.

		[ProtoMember(5)]
		/// <summary>
		/// Height Above Ground Level in meters
		/// </summary>
		public double HeightAGL { get; set; }        //Heigh above ground level in meters

		[ProtoMember(6)]
		/// <summary>
		/// Velocity along the x-axis in meters per second.
		/// </summary>
		public double VelEastMpS { get; set; }

		[ProtoMember(7)]
		/// <summary>
		/// Velocity along the y-axis in meters per second
		/// </summary>
		public double VelNorthMpS { get; set; }

		[ProtoMember(8)]
		/// <summary>
		/// Velocity in the z-axis... down is a positive 
		/// velocity... this is negative to the z-axis direction
		/// where up is positive.
		/// </summary>
		public double VelDownMpS { get; set; }

		[ProtoMember(9)]
		/// <summary>
		/// Roll angle in radians
		/// At zero degrees/radians the vehicle is aligned with the horizon.
		/// Right-wing down is a postive angle.
		/// </summary>
		public double RollRad { get; set; }

		[ProtoMember(10)]
		/// <summary>
		/// Pitch angle in radians
		/// At zero degrees/radians the vehicle is aligned with the horizon.
		/// Positive angle means the nose of the vehicle is pointed up.
		/// </summary>
		public double PitchRad { get; set; }

		[ProtoMember(11)]
		/// <summary>
		/// Yaw angle in radians.  
		/// Zero degrees/radians is pointing to true North.
		/// </summary>
		public double YawRad { get; set; }

		[ProtoMember(12)]
		/// <summary>
		/// Roll rate in radians per second
		/// </summary>
		public double RollRateRadps { get; set; }

		[ProtoMember(13)]
		/// <summary>
		/// Pitch rate in radians per second
		/// </summary>
		public double PitchRateRadps { get; set; }

		[ProtoMember(14)]
		/// <summary>
		/// Yaw rate in radians per second.
		/// </summary>
		public double YawRateRadps { get; set; }

		[ProtoMember(15)]
		/// <summary>
		/// The Inertial States timestamp is tied to the GPS time coming from the avionics box.
		/// </summary>
		public double gpsTimeStampSec { get; set; }


		public VehicleInterialStatesMsg()
		{
			Clear();
		}

		public void Clear()
		{
			CoordinatesLatLonOrXY = true;   //Lat/Lon is the default
			LatitudeRadOrY = 0;
			LongitudeRadOrX = 0;
			AltitudeMSL = 0;
			HeightAGL = 0;
			VelEastMpS = 0;
			VelNorthMpS = 0;
			VelDownMpS = 0;
			RollRad = 0;
			PitchRad = 0;
			YawRad = 0;
			RollRateRadps = 0;
			PitchRateRadps = 0;
			YawRateRadps = 0;
			gpsTimeStampSec = 0;
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
				Serializer.Serialize<VehicleInterialStatesMsg>(ms, this);
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
		public static VehicleInterialStatesMsg Deserialize(byte[] b)
		{
			VehicleInterialStatesMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<VehicleInterialStatesMsg>(ms);
			}
			return r;
		}
		#endregion
	}
}

