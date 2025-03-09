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
	public class IMUCommandResponsePBMsg
	{
        /// <summary>
        /// IMURemoteCtrlEnable
        /// Informs Videre that IMU control commands are coming from 
        /// a remote source such as this moniotor.
        /// </summary>
		[ProtoMember(1)]
		public bool IMURemoteCtrlEnable { get; set; }	

		[ProtoMember(2)]
		public string CmdRspMsg { get; set; }	
	

		public IMUCommandResponsePBMsg()
		{
			Clear();
		}

		public void Clear()
		{
            IMURemoteCtrlEnable = true;
			CmdRspMsg = "";
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
				Serializer.Serialize<IMUCommandResponsePBMsg>(ms, this);
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
		public static IMUCommandResponsePBMsg Deserialize(byte[] b)
		{
			IMUCommandResponsePBMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<IMUCommandResponsePBMsg>(ms);
			}
			return r;
		}
		#endregion


	}
}

