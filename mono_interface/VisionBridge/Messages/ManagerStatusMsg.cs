/* ****************************************************************
 * Vision Bridge
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Aug. 2016
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
	/// Manager Status message.
	/// Contains operational status of a Vision system manager.
	/// </summary>
	public class ManagerStatsMsg
	{
		[ProtoMember(1)]
		/// <summary>
		/// The Manager's Name
		/// </summary>
		public string ManagerName { get; set; }

		[ProtoMember(2)]
		/// <summary>
		/// The Manager's State of operation:
		/// 	0 = Startup
		/// 	1 = Running
		/// 	2 = Shutdown
		/// </summary>
		public Int32 RunningState { get; set; }

		[ProtoMember(3)]
		/// <summary>
		/// The Manager Error 
		/// 	true = Error
		/// 	false = Ok
		/// </summary>
		public bool ErrorCondition { get; set; }

		[ProtoMember(4)]
		/// <summary>
		/// The Manager's Error Code:
		/// </summary>
		public Int32 ErrorCode { get; set; }

		[ProtoMember(5)]		
		/// <summary>
		/// Gets or sets the totol number of execute cycles.
		/// </summary>
		/// <value>The totol number of execute cycles.</value>
		public Int32 TotalNumberOfExecuteCycles { get; set; }

		[ProtoMember(6)]
		/// <summary>
		/// Gets or sets the number of execute cycles since
		/// last Stats Reset.
		/// </summary>
		/// <value>The number of execute cycles.</value>
		public Int32 NumberOfExecuteCycles { get; set; }

		[ProtoMember(7)]
		/// <summary>
		/// TimeSinceLastStatsReset
		/// </summary>
		public double TimeSinceLastStatsReset_Sec { get; set; }	

		[ProtoMember(8)]
		public double MinExecuteUnitOfWorkTime_Sec { get; set; }

		[ProtoMember(9)]
		public double MaxExecuteUnitOfWorkTime_Sec { get; set; }

		[ProtoMember(10)]
		public double AveExecuteUnitOfWorkTime_Sec { get; set; }

		[ProtoMember(11)]
		public double TotalExecuteUnitOfWorkTime_Sec { get; set; }		

		[ProtoMember(12)]
		public double MinSleepTime_Sec { get; set; }

		[ProtoMember(13)]
		public double MaxSleepTime_Sec { get; set; }

		[ProtoMember(14)]
		public double AveSleepTime_Sec { get; set; }

		[ProtoMember(15)]
		public double TotalSleepTime_Sec { get; set; }		

		[ProtoMember(16)]
		public Int32 NumberOfWakeupCallsWhileAsleep { get; set; }

		[ProtoMember(17)]
		public Int32 NumberOfWakeupCallsWhileAwake { get; set; }

		public ManagerStatsMsg()
		{
			Clear();
		}

		public void Clear()
		{
			ManagerName = "Unknown";
			RunningState = 0;
			ErrorCondition = false;
			ErrorCode = 0;
			TotalNumberOfExecuteCycles = 0;
			NumberOfExecuteCycles = 0;
			TimeSinceLastStatsReset_Sec = 0;

			MinExecuteUnitOfWorkTime_Sec = 0;
			MaxExecuteUnitOfWorkTime_Sec = 0;
			AveExecuteUnitOfWorkTime_Sec = 0;
			TotalExecuteUnitOfWorkTime_Sec = 0;

			MinSleepTime_Sec = 0;
			MaxSleepTime_Sec = 0;
			AveSleepTime_Sec = 0;
			TotalSleepTime_Sec = 0;

			NumberOfWakeupCallsWhileAsleep = 0;
			NumberOfWakeupCallsWhileAwake = 0;
		}

		//----------------------------------------------------------------------
		//----------------------------------------------------------------------
		#region SerializeResponse
		/// <summary>
		/// Serialize ManagerStatsMsg to byte array.
		/// </summary>
		public byte[] Serialize()
		{
			byte[] b = null;
			using (var ms = new MemoryStream())
			{
				Serializer.Serialize<ManagerStatsMsg>(ms, this);
				b = new byte[ms.Position];
				var fullB = ms.GetBuffer();
				Array.Copy(fullB, b, b.Length);
			}

			return b;
		}

		/// <summary>
		/// Deserialize to ManagerStatsMsg from byte array.
		/// </summary>
		/// <param name="b">The blue component.</param>
		public static ManagerStatsMsg Deserialize(byte[] b)
		{
			ManagerStatsMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<ManagerStatsMsg>(ms);
			}
			return r;
		}
		#endregion

	}
}
