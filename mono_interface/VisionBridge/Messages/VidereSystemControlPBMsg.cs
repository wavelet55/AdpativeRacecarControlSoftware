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
	/// Videre System Control and Monitor Message.
	/// </summary>
	public class VidereSystemControlPBMsg
	{
		[ProtoMember(1)]
		public UInt32 SystemState { get; set; }	
	
		[ProtoMember(2)]
		public bool StartProcess { get; set; }	

        [ProtoMember(3)]
		public bool PauseProcess { get; set; }	

        [ProtoMember(4)]
		public bool StopProcess { get; set; }	

		[ProtoMember(5)]
		public UInt32 SystemStatus { get; set; }	

		[ProtoMember(6)]
		public UInt32 StatusCounter { get; set; }	

        [ProtoMember(7)]
		public bool DriverEnableSwitch { get; set; }
	
        [ProtoMember(8)]
		public bool HeadControlEnable { get; set; }

        [ProtoMember(9)]
		public bool ThrottleControlEnable { get; set; }

        [ProtoMember(10)]
		public bool BrakeControlEnable { get; set; }

	[ProtoMember(11)]
		public bool NexusBCIControlEnabled { get; set; }

	[ProtoMember(12)]
		public bool NexusBCIThrottleEnable { get; set; }


		public VidereSystemControlPBMsg()
		{
			Clear();
		}

		public void Clear()
		{
			SystemState = 0;
            StartProcess = false;
            PauseProcess = false;
            StopProcess = false;
			SystemStatus = 0;
			StatusCounter = 0;
            HeadControlEnable = true;
            ThrottleControlEnable = true;
            BrakeControlEnable = true;
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
				Serializer.Serialize<VidereSystemControlPBMsg>(ms, this);
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
		public static VidereSystemControlPBMsg Deserialize(byte[] b)
		{
			VidereSystemControlPBMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<VidereSystemControlPBMsg>(ms);
			}
			return r;
		}
		#endregion


	}
}

