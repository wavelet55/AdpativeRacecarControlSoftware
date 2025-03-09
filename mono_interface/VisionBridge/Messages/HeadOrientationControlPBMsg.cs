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
    public enum HeadOrientationOutputSelect_e
    {
        NoOutput,
        ImageProcTrackHead,
        HeadOrientation,
        VehicleOrientation
    }

	[ProtoContract]
	/// <summary>
	/// Head Tracking Control Message.
	/// </summary>
	public class HeadOrientationControlPBMsg
	{
		[ProtoMember(1)]
		public UInt32 HeadOrientationSelect { get; set; }	

		[ProtoMember(2)]
		public bool DisableHeadOrientationKalmanFilter { get; set; }
	
		[ProtoMember(3)]
		public bool DisableVehicleInputToHeadOrientation { get; set; }
	
		[ProtoMember(4)]
		public bool DisableVehicleGravityFeedback { get; set; }	
	
		[ProtoMember(5)]
		public double VehicleGravityFeedbackGain { get; set; }	
	
		[ProtoMember(6)]
		public double HeadOrientation_QVar { get; set; }	

		[ProtoMember(7)]
		public double HeadOrientation_RVar { get; set; }	

		public HeadOrientationControlPBMsg()
		{
			Clear();
		}

		public void Clear()
		{
			HeadOrientationSelect = 0;
            DisableHeadOrientationKalmanFilter = false;
            DisableVehicleInputToHeadOrientation = false;
            DisableVehicleGravityFeedback = false;
            VehicleGravityFeedbackGain = 0.0;
            HeadOrientation_QVar = 0.0;
			HeadOrientation_RVar = 0.0;
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
				Serializer.Serialize<HeadOrientationControlPBMsg>(ms, this);
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
		public static HeadOrientationControlPBMsg Deserialize(byte[] b)
		{
			HeadOrientationControlPBMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<HeadOrientationControlPBMsg>(ms);
			}
			return r;
		}
		#endregion


	}
}

