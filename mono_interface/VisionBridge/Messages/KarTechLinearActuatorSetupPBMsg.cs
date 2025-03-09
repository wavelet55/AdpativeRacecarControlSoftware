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
	public class KarTechLinearActuatorSetupPBMsg
	{

		[ProtoMember(1)]
		public bool ResetOutputs { get; set; }

		[ProtoMember(2)]
		public bool ResetHardwareCfgs { get; set; }

		[ProtoMember(3)]
		public bool ResetUserCfgs { get; set; }

		[ProtoMember(4)]
		public bool ResetAll { get; set; }

		[ProtoMember(5)]
		public bool AutoZeroCal { get; set; }

		[ProtoMember(6)]
		public bool SetCanCommandResponseIDs { get; set; }


		public KarTechLinearActuatorSetupPBMsg()
		{
			Clear();
		}

		public void Clear()
		{
            ResetOutputs = false;
            ResetHardwareCfgs = false;
            ResetUserCfgs = false;
            ResetAll = false;
            AutoZeroCal = false;
            SetCanCommandResponseIDs = false;
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
				Serializer.Serialize<KarTechLinearActuatorSetupPBMsg>(ms, this);
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
		public static KarTechLinearActuatorSetupPBMsg Deserialize(byte[] b)
		{
			KarTechLinearActuatorSetupPBMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<KarTechLinearActuatorSetupPBMsg>(ms);
			}
			return r;
		}
		#endregion


	}
}


