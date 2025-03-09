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
	public class KarTechLinearActuatorParamsPBMsg
	{
		[ProtoMember(1)]
		public double MinPositionInches { get; set; }		

        [ProtoMember(2)]
		public double MaxPositionInches { get; set; }		

		[ProtoMember(3)]
		public double MotorMaxCurrentLimitAmps { get; set; }		

		[ProtoMember(4)]
		public double FeedbackCtrl_ErrDeadbandInches { get; set; }		

		[ProtoMember(5)]
		public UInt32 FeedbackCtrl_KP { get; set; }		

		[ProtoMember(6)]
		public UInt32 FeedbackCtrl_KI { get; set; }		

		[ProtoMember(7)]
		public UInt32 FeedbackCtrl_KD { get; set; }		

		[ProtoMember(8)]
		public UInt32 FeedbackCtrl_CLFreq { get; set; }		

        [ProtoMember(9)]
		public UInt32 Motor_MinPWM { get; set; }		

        [ProtoMember(10)]
		public UInt32 Motor_MaxPWM { get; set; }		

        [ProtoMember(11)]
		public UInt32 Motor_pwmFreq { get; set; }		

        [ProtoMember(12)]
		public UInt32 PositionReachedErrorTimeMSec { get; set; }		


		public KarTechLinearActuatorParamsPBMsg()
		{
			Clear();
		}

		public void Clear()
		{
            MinPositionInches = 0;
            MaxPositionInches = 3.0;
            MotorMaxCurrentLimitAmps = 65.0;
            FeedbackCtrl_ErrDeadbandInches = 40;
            FeedbackCtrl_KP = 1000;
            FeedbackCtrl_KI = 1000;
            FeedbackCtrl_KD = 10;
            FeedbackCtrl_CLFreq = 60;
            Motor_MinPWM = 20;
            Motor_MaxPWM = 90;
            Motor_pwmFreq = 2000;
            PositionReachedErrorTimeMSec = 0;
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
				Serializer.Serialize<KarTechLinearActuatorParamsPBMsg>(ms, this);
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
		public static KarTechLinearActuatorParamsPBMsg Deserialize(byte[] b)
		{
			KarTechLinearActuatorParamsPBMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<KarTechLinearActuatorParamsPBMsg>(ms);
			}
			return r;
		}
		#endregion


	}
}

