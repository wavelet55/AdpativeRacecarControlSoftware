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
	public class VehicleControlParametersPBMsg
	{
		[ProtoMember(1)]
        public double SipnPuffBlowGain;

		[ProtoMember(2)]
        public double SipnPuffSuckGain;

		[ProtoMember(3)]
        public double SipnPuffDeadBandPercent;
         

		[ProtoMember(4)]
        public bool ReverseSipnPuffThrottleBrake;

		[ProtoMember(5)]
        public double ThrottleSipnPuffGain;

		[ProtoMember(6)]
        public double BrakeSipnPuffGain;
         

		[ProtoMember(7)]
        public bool ThrottleBrakeHeadTiltEnable;

		[ProtoMember(8)]
        public double ThrottleBrakeHeadTiltForwardDeadbandDegrees;

		[ProtoMember(9)]
        public double ThrottleBrakeHeadTiltBackDeadbandDegrees;

		[ProtoMember(10)]
        public double ThrottleHeadTiltGain;

		[ProtoMember(11)]
        public double BrakeHeadTiltGain;
         

		[ProtoMember(12)]
        //Steering Angle or Torque Control
        public bool UseSteeringAngleControl;

		[ProtoMember(13)]
        public double SteeringDeadband;

		[ProtoMember(14)]
        public double SteeringControlGain;

		[ProtoMember(15)]
        public double MaxLRHeadRotationDegrees;

		[ProtoMember(16)]
        public Int32 HeadLeftRighLPFOrder;

		[ProtoMember(17)]
        public double HeadLeftRighLPFCutoffFreqHz;

		[ProtoMember(18)]
        public double SteeringAngleFeedback_Kp;

		[ProtoMember(19)]
        public double SteeringAngleFeedback_Kd;

		[ProtoMember(20)]
        public double SteeringAngleFeedback_Ki;

		[ProtoMember(21)]
        public double SteeringBiasAngleDegrees;

		[ProtoMember(22)]
        public double RCSteeringGain;

		[ProtoMember(23)]
		public double BCIGain;


		public VehicleControlParametersPBMsg()
		{
			Clear();
		}

		public void Clear()
		{
            SipnPuffBlowGain = 1.0;
            SipnPuffSuckGain = 1.0;
            SipnPuffDeadBandPercent = 5.0;
			BCIGain = 1.0;

			ReverseSipnPuffThrottleBrake = false;
            ThrottleSipnPuffGain = 1.0;
            BrakeSipnPuffGain = 1.0;

            ThrottleBrakeHeadTiltEnable = false;
            ThrottleBrakeHeadTiltForwardDeadbandDegrees = 10.0;
            ThrottleBrakeHeadTiltBackDeadbandDegrees = 10.0;
            ThrottleHeadTiltGain = 1.0;
            BrakeHeadTiltGain = 1.0;

            UseSteeringAngleControl = false;
            SteeringDeadband = 2.5;
            SteeringControlGain = 1.0;

            MaxLRHeadRotationDegrees = 60.0;
            HeadLeftRighLPFOrder = 4;
            HeadLeftRighLPFCutoffFreqHz = 5.0;

            SteeringAngleFeedback_Kp = 1.0;
            SteeringAngleFeedback_Kd = 0;
            SteeringAngleFeedback_Ki = 0;
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
				Serializer.Serialize<VehicleControlParametersPBMsg>(ms, this);
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
		public static VehicleControlParametersPBMsg Deserialize(byte[] b)
		{
			VehicleControlParametersPBMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<VehicleControlParametersPBMsg>(ms);
			}
			return r;
		}
		#endregion


	}
}

