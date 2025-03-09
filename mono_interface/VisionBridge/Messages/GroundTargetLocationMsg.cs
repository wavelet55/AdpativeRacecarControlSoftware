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
using System.Collections.Generic;

namespace VisionBridge.Messages
{
	
	[ProtoContract]
	/// <summary>
	/// Target Location Information for Ground based targets
    /// Provides location on the ground for a target.
	/// </summary>
	public class GroundTargetLocationMsg
	{
		[ProtoMember(1)]
		/// <summary>
        /// Target Type Code
        /// This is an agreeded upon code for the type of target
        /// (Building, car, human...)  Since the codes will change based upon
        /// mission and image processing... an enum is not used.
        /// Image processing will return this code for the type of target it finds.
		/// </summary>
		public UInt32 TargetTypeCode { get; set; }

		[ProtoMember(2)]
		/// <summary>
        /// Target ID
        /// A unique ID number for the target... if known
        /// otherwise set to zero.
		/// </summary>
		public UInt32 TargetID { get; set; }


		[ProtoMember(3)]
		/// <summary>
		/// Target Latitude Location in Radians 
		/// </summary>
		public double TargetLatitudeRadians { get; set; }		

		[ProtoMember(4)]
		/// <summary>
		/// Target Longitude Location in Radians 
		/// </summary>
		public double TargetLongitudeRadians { get; set; }	
	
		[ProtoMember(5)]
		/// <summary>
		/// Target Altitude Location in meters above Mean Sea Level
        /// or above earth standard elipsoid
		/// </summary>
		public double TargetAltitudeMSL { get; set; }		

		[ProtoMember(6)]
		/// <summary>
		/// Target Altitude is set/valid if true,
        /// otherwise the Altitude data was not set and 
        /// is not valid.
		/// </summary>
        public bool TargetAltitudeValid { get; set; }

		[ProtoMember(7)]
        /// <summary>
        /// The Azimuth Angle to the target relative to the UAV
        /// location. 
        /// This value is for compatibility with how target information
        /// is provided to the HOPS Sensor Fusion... it should be obsoleted 
        /// in time and the target lat/lon and covarience used.
        /// </summary>
        public double TargetAzimuthRadians;

		[ProtoMember(8)]
        /// <summary>
        /// The Elevation Angle to the target relative to the UAV
        /// location. 
        /// This value is for compatibility with how target information
        /// is provided to the HOPS Sensor Fusion... it should be obsoleted 
        /// in time and the target lat/lon and covarience used.
        /// </summary>
        public double TargetElevationRadians;

        [ProtoMember(9)]
        /// <summary>
        /// The orientation of the target with respect to true north
        /// in radians with positive angles clockwise (East = pi/2). 
        /// This value is for compatibility with how target information
        /// is provided to the HOPS Sensor Fusion... it should be obsoleted 
        /// in time and the target lat/lon and covarience used.
        /// </summary>
        public double TargetOrientationRadians;

        [ProtoMember(10)]
        /// <summary>
        /// This flag will be true if target Azimuth, Elevation,
        /// and Orientation angle values are set, otherwise the 
        /// flag will be false.
        /// </summary>
        public bool TargetAzimuthElevationOrientationValid;

        //Target Covarience Values... 
        [ProtoMember(11)]
        /// <summary>
        /// A 2x2 Matrix of the Covarience values...
        /// Accesss by:  row, column:  (2*row + column)
        /// If the lenght of TargetCovarianceMatrix is less than 4,
        /// then it should be assumed the covariance values were not 
        /// set and are invalid.
        /// </summary>
        public List<double> TargetCovarianceMatrix;

        [ProtoMember(12)]
        public bool TargetCovarianceValid;

        [ProtoMember(13)]
        public Int32 TargetPixelLocation_x;

        [ProtoMember(14)]
        public Int32 TargetPixelLocation_y;


		public GroundTargetLocationMsg()
		{
            TargetCovarianceMatrix = new List<double>();
			Clear();
		}

		public void Clear()
		{
			TargetTypeCode = 0;
			TargetID = 0;
			TargetLatitudeRadians = 0;
			TargetLongitudeRadians = 0;
			TargetAltitudeMSL = 0;  
			TargetAltitudeValid = false;  
			TargetAzimuthRadians = 0;
            TargetElevationRadians = 0;
            TargetOrientationRadians = 0;
            TargetAzimuthElevationOrientationValid = false;
            TargetCovarianceMatrix.Clear();
            TargetCovarianceValid = false;
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
				Serializer.Serialize<GroundTargetLocationMsg>(ms, this);
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
		public static GroundTargetLocationMsg Deserialize(byte[] b)
		{
			GroundTargetLocationMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<GroundTargetLocationMsg>(ms);
			}
			return r;
		}
		#endregion


	}
}
