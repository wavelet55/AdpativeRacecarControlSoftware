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
	public class LinearActuatorPositionCtrlPBMsg
	{
		[ProtoMember(1)]
		public double PositionPercent { get; set; }		

		[ProtoMember(2)]
		public bool ClutchEnable { get; set; }

		[ProtoMember(3)]
		public bool MotorEnable { get; set; }

		[ProtoMember(4)]
		public bool ManualExtControl { get; set; }

		[ProtoMember(5)]
		public double MotorCurrentAmps { get; set; }		

		[ProtoMember(6)]
		public double TempDegC { get; set; }		

		[ProtoMember(7)]
		public Int32 ErrorFlags { get; set; }

		[ProtoMember(8)]
		public bool ActuatorSetupMode { get; set; }

		public LinearActuatorPositionCtrlPBMsg()
		{
			Clear();
		}

		public void Clear()
		{
			PositionPercent = 0;
			MotorCurrentAmps = 0;
			TempDegC = 22.0;
			ClutchEnable = false;
			MotorEnable = false;
			ManualExtControl = false;
            ActuatorSetupMode = false;
            ErrorFlags = 0;
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
				Serializer.Serialize<LinearActuatorPositionCtrlPBMsg>(ms, this);
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
		public static LinearActuatorPositionCtrlPBMsg Deserialize(byte[] b)
		{
			LinearActuatorPositionCtrlPBMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<LinearActuatorPositionCtrlPBMsg>(ms);
			}
			return r;
		}
		#endregion


	}
}

