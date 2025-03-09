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
	public class DceEPASteeringStatusPBMsg
	{
		[ProtoMember(1)]
		public double MotorCurrentAmps { get; set; }		

        [ProtoMember(2)]
		public double PWMDutyCyclePercent { get; set; }		

		[ProtoMember(3)]
		public double MotorTorquePercent { get; set; }		

		[ProtoMember(4)]
		public double SupplyVoltage { get; set; }		

		[ProtoMember(5)]
		public double TempDegC { get; set; }		

		[ProtoMember(6)]
		public double SteeringAngleDeg { get; set; }		

		[ProtoMember(7)]
		public Int32 SteeringTorqueMapSetting { get; set; }		

		[ProtoMember(8)]
		public Int32 SwitchPosition { get; set; }		

        [ProtoMember(9)]
		public Int32 TorqueA { get; set; }		

        [ProtoMember(10)]
		public Int32 TorqueB { get; set; }		

        [ProtoMember(11)]
		public Int32 ErrorCode { get; set; }		

        [ProtoMember(12)]
		public Int32 StatusFlags { get; set; }		

        [ProtoMember(13)]
		public Int32 LimitFlags { get; set; }		

		[ProtoMember(14)]
		public bool ManualExtControl { get; set; }



		public DceEPASteeringStatusPBMsg()
		{
			Clear();
		}

		public void Clear()
		{
            MotorCurrentAmps = 0;
            PWMDutyCyclePercent = 0;
            MotorTorquePercent = 0;
            SupplyVoltage = 0;
            TempDegC = 22;
            SteeringAngleDeg = 0;     
            SteeringTorqueMapSetting = 0;
            SwitchPosition = 0;     
            TorqueA = 0;
            TorqueB = 0;
            ErrorCode = 0;
            StatusFlags = 0;
            LimitFlags = 0;
			ManualExtControl = false;
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
				Serializer.Serialize<DceEPASteeringStatusPBMsg>(ms, this);
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
		public static DceEPASteeringStatusPBMsg Deserialize(byte[] b)
		{
			DceEPASteeringStatusPBMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<DceEPASteeringStatusPBMsg>(ms);
			}
			return r;
		}
		#endregion


	}
}

