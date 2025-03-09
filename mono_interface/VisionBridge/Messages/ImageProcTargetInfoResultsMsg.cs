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
	/// Image Processing Target Information 
    /// Provides location the targets found in a given 
    /// image..
	/// </summary>
	public class ImageProcTargetInfoResultsMsg
	{
		[ProtoMember(1)]
		/// <summary>
        /// Image Location Information 
        /// this includes the Image Number.
		/// </summary>
		public ImageLocationMsg ImageLocation { get; set; }

		[ProtoMember(2)]
		/// <summary>
		/// Image Center Latitude Location in Radians 
		/// </summary>
		public VehicleInterialStatesMsg VehicleInertialStates { get; set; }		

        [ProtoMember(3)]
        /// <summary>
        /// List of Target Locations.
        /// If there are no targets in the image, the list will be empty.
        /// </summary>
        public List<GroundTargetLocationMsg> TargetLocations;


		public ImageProcTargetInfoResultsMsg()
		{
            ImageLocation = new ImageLocationMsg();
            VehicleInertialStates = new VehicleInterialStatesMsg();
            TargetLocations = new List<GroundTargetLocationMsg>();
		}

		public void Clear()
		{
            ImageLocation.Clear();
            VehicleInertialStates.Clear();
            TargetLocations.Clear();
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
				Serializer.Serialize<ImageProcTargetInfoResultsMsg>(ms, this);
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
		public static ImageProcTargetInfoResultsMsg Deserialize(byte[] b)
		{
			ImageProcTargetInfoResultsMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<ImageProcTargetInfoResultsMsg>(ms);
			}
			return r;
		}
		#endregion


	}
}


