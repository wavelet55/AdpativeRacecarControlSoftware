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
	/// Manager Control message.
	/// Contains control data for a Manager.
	/// </summary>
	public class ManagerControlMsg
	{

		[ProtoMember(1)]
		/// <summary>
		/// The Manager's Name 
		/// or "All" if the control message applies to all 
		/// managers.  Manager Names are case-insensitive.
		/// </summary>
		public string ManagerName { get; set; }

		[ProtoMember(2)]
		/// <summary>
		/// Use to shutdown an individual manager or all managers
		/// if the Manager name is "All".
		/// A manager cannot be re-started once shutdown... so becareful
		/// with this flag.
		/// </summary>
		public bool ShutdownManager { get; set; }

		[ProtoMember(3)]
		/// <summary>
		/// Toggle this flag to cause the Manager's Stats to be reset/cleared.
		/// If the Manager name is "All" then all managers stats will be reset.
		/// </summary>
		public bool ResetManagerStatsToggle { get; set; }

		[ProtoMember(4)]
		/// <summary>
		/// This controls how often the managers stats are published.
		/// If zero or less, the manager stats are turned off and 
		/// not published.
		/// </summary>
		/// <value>The publish mgr stats time sec.</value>
		public double PublishMgrStatsTime_Sec { get; set; }

		public ManagerControlMsg()
		{
			Clear();
		}

		public void Clear()
		{
			ManagerName = "Unknown";
			ShutdownManager = false;
			ResetManagerStatsToggle = false;
			PublishMgrStatsTime_Sec = 10.0;
		}

		//----------------------------------------------------------------------
		//----------------------------------------------------------------------
		#region SerializeResponse
		/// <summary>
		/// Serialize ManagerControlMsg to byte array.
		/// </summary>
		public byte[] Serialize()
		{
			byte[] b = null;
			using (var ms = new MemoryStream())
			{
				Serializer.Serialize<ManagerControlMsg>(ms, this);
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
		public static ManagerControlMsg Deserialize(byte[] b)
		{
			ManagerControlMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<ManagerControlMsg>(ms);
			}
			return r;
		}
		#endregion

	}
}

