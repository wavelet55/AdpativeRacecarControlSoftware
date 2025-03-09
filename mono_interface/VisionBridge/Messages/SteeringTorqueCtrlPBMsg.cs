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
	public class SteeringTorqueCtrlPBMsg
	{
		[ProtoMember(1)]
		public double SteeringTorquePercent { get; set; }	
	
        [ProtoMember(2)]
		public UInt32 SteeringTorqueMap { get; set; }		

		[ProtoMember(3)]
		public bool SteeringControlEnabled { get; set; }

		[ProtoMember(4)]
		public bool ManualExtControl { get; set; }



		public SteeringTorqueCtrlPBMsg()
		{
			Clear();
		}

		public void Clear()
		{
			SteeringTorquePercent = 0;
            SteeringTorqueMap = 0;      //Zero disables the Steering Control
			SteeringControlEnabled = false;
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
				Serializer.Serialize<SteeringTorqueCtrlPBMsg>(ms, this);
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
		public static SteeringTorqueCtrlPBMsg Deserialize(byte[] b)
		{
			SteeringTorqueCtrlPBMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<SteeringTorqueCtrlPBMsg>(ms);
			}
			return r;
		}
		#endregion


	}
}

