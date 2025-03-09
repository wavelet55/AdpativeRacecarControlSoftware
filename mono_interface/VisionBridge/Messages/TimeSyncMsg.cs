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

namespace VisionBridge.Messages
{
	[ProtoContract]
	/// <summary>
	/// Time sync message.
	/// Provides a time sync offset in order to synchronize the CPU clock
	/// with GPS time.
	/// </summary>
	public class TimeSyncMsg
	{

		[ProtoMember(1)]
		/// <summary>
		/// Time Sync Offset in Seconds
		/// Add this offset time in seconds to synchronize the system/computer
		/// clock with the GPS time.
		/// </summary>
		public double TimeSyncOffsetSeconds { get; set; }


		public TimeSyncMsg()
		{
			TimeSyncOffsetSeconds = 0;
		}

		public TimeSyncMsg(double timeOffsetSec)
		{
			TimeSyncOffsetSeconds = timeOffsetSec;
		}

		public void Clear()
		{
			TimeSyncOffsetSeconds = 0;
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
				Serializer.Serialize<TimeSyncMsg>(ms, this);
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
		public static TimeSyncMsg Deserialize(byte[] b)
		{
			TimeSyncMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<TimeSyncMsg>(ms);
			}
			return r;
		}
		#endregion

	}
}

