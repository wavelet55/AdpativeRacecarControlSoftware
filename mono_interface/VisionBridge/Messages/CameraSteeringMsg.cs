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
	/// Camera Steering Message
	/// This message provides specific information about how the 
	/// camera sensor is pointed.  This infomation coupled with the 
	/// Vehicle States Message should resolve the specific location
	/// the camera is pointing at.
	/// </summary>
	public class CameraSteeringMsg
	{
		[ProtoMember(1)]
		/// <summary>
		/// SPOI:  Sensor Point of Interest
		/// If CameraSteeringModeSPOI is true, the Camera Sensor is being 
		/// actively pointed to a specific target location.  The Lat/Lon or 
		/// X-Y position inforation is the location (typically on the ground) the
		/// camera sensor is being actively pointed at.  In this case, the vehicle
		/// attitude information (pitch/roll/yaw) and camera Azmuth/Elevation angles,
		/// along with velocity information, should be ignored.  The Lat/Lon or X-Y position
		/// information provides the location the camera is pointed at.
		/// 
		/// If CameraSteeringModeSPOI is false, the camera sensor is pointed (azmuth/elevation)
		/// relative to the vehicle position and attitude.   
		/// </summary>
		public bool CameraSteeringModeSPOI { get; set; }

		[ProtoMember(2)]
		/// <summary>
		/// If CoordinatesLatLon is true, the position coordinates
		/// are in Latitude and Longitude.
		/// If CoordinatesLatLon is false, the position coordinates
		/// are passed as X-Y position data.
		/// </summary>
		public bool CoordinatesLatLonOrXY { get; set; }

		[ProtoMember(3)]
		/// <summary>
		/// Camera Pointing Location Latitude in Radians
		/// Or Y position in meters
		/// </summary>
		public double SpoiLatitudeRadOrY { get; set; }		

		[ProtoMember(4)]
		/// <summary>
		/// Camera Pointing Location Longitude in Radians
		/// or X position in meters
		/// </summary>
		public double SpoiLongitudeRadOrX { get; set; }     

		[ProtoMember(5)]
		/// <summary>
		/// Camera angle in radians with respect to the vehicle.
		/// Azimuth angle is left/right relative to a pilots view.
		/// Zero radians is striaght ahead.
		/// Positive angles are to the right, negative is to the left.
		/// </summary>
		public double CameraAzimuthAngleRad { get; set; }

		[ProtoMember(6)]
		/// <summary>
		/// Camera angle in radians with respect to the vehicle.
		/// Elevation is up/down relative to a pilots view.
		/// Zero radians is straigh ahead.
		/// Positive angles are up, negative angles are down.
		/// -pi/2 is straight down relative to the vehicle.
		/// </summary>
		public double CameraElevationAngleRad { get; set; }

		[ProtoMember(7)]
		/// <summary>
		/// Camera Zoom angle in percent [0 to 100]
		/// 100% is maximum zoom, 0 percent is minimum 
		/// or no zoom.
		/// </summary>
		public double CameraZoomPercent { get; set; }


		public CameraSteeringMsg()
		{
			Clear();
		}

		public void Clear()
		{
			CameraSteeringModeSPOI = false;
			CoordinatesLatLonOrXY = true;
			SpoiLatitudeRadOrY = 0;
			SpoiLongitudeRadOrX = 0;
			CameraAzimuthAngleRad = 0;  //Default is camera pointing straith ahead.
			CameraElevationAngleRad = -0.5 * Math.PI;  //Default is camera pointing down.
			CameraZoomPercent = 0;
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
				Serializer.Serialize<CameraSteeringMsg>(ms, this);
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
		public static CameraSteeringMsg Deserialize(byte[] b)
		{
			CameraSteeringMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<CameraSteeringMsg>(ms);
			}
			return r;
		}
		#endregion


	}
}

