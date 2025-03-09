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
	/// Image Location Information 
    /// Provides location on the ground for of the Image.
	/// </summary>
	public class ImageLocationMsg
	{
		[ProtoMember(1)]
		/// <summary>
        /// Image Number
        /// Each Image is tagged with a number when it is captured 
        /// for reference purposes.
		/// </summary>
		public UInt32 ImageNumber { get; set; }

		[ProtoMember(2)]
		/// <summary>
		/// Image Center Latitude Location in Radians 
		/// </summary>
		public double ImageCenterLatitudeRadians { get; set; }		

		[ProtoMember(3)]
		/// <summary>
		/// Image Center Longitude Location in Radians 
		/// </summary>
		public double ImageCenterLongitudeRadians { get; set; }	
	
        [ProtoMember(4)]
        /// <summary>
        /// A 4x2 Matrix of Image Corner Information
        /// Accesss by:  corner number, column:  (2*corner + x-Lat=0 or Long-y=1)
        /// Corner information can be given relative to the image center in x-y 
        /// coordinates where x is the east-west direction and y is in the north-south
        /// direction in meters from the center of the image.
        /// Or, the corners can be given in Lat/Long (radians) 
        /// The corners go counter clockwise from the North-East Corner.
        /// </summary>
        public List<double> TargetCornerLocations;

        [ProtoMember(5)]
        /// <summary>
        /// TargetCornersLatLonOrXY
        /// If true... corner locations are Lat/Long in Radians.
        /// If false... corner locations are X-Y in meters from the center of the 
        /// Image. 
        /// </summary>
        public bool TargetCornersLatLonOrXY;


		public ImageLocationMsg()
		{
            TargetCornerLocations = new List<double>();
			Clear();
		}

		public void Clear()
		{
			ImageNumber = 0;
			ImageCenterLatitudeRadians = 0;
			ImageCenterLongitudeRadians = 0;
            TargetCornerLocations.Clear();
            TargetCornersLatLonOrXY = false;
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
				Serializer.Serialize<ImageLocationMsg>(ms, this);
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
		public static ImageLocationMsg Deserialize(byte[] b)
		{
			ImageLocationMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<ImageLocationMsg>(ms);
			}
			return r;
		}
		#endregion


	}
}


