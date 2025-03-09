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
	/// List of Manager Names and number of Manager
    /// used by Videre.
	/// </summary>
	public class ListOfManagerNamesPBMsg
	{

		[ProtoMember(1)]
		/// <summary>
		/// The Number of Acrive Managers used by Videre.
		/// </summary>
		public int NumberOfManagers { get; set; }

		[ProtoMember(2)]
		/// <summary>
		/// List of Active Manager Names
		/// </summary>
		public string[] ListOfManagerNames { get; set; }


		public ListOfManagerNamesPBMsg()
		{
			Clear();
		}

		public void Clear()
		{
			NumberOfManagers = 0;
            ListOfManagerNames = null;
		}

		//----------------------------------------------------------------------
		//----------------------------------------------------------------------
		#region SerializeResponse
		/// <summary>
		/// Serialize ListOfManagerNamesPBMsg to byte array.
		/// </summary>
		public byte[] Serialize()
		{
			byte[] b = null;
			using (var ms = new MemoryStream())
			{
				Serializer.Serialize<ListOfManagerNamesPBMsg>(ms, this);
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
		public static ListOfManagerNamesPBMsg Deserialize(byte[] b)
		{
			ListOfManagerNamesPBMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<ListOfManagerNamesPBMsg>(ms);
			}
			return r;
		}
		#endregion

	}
}

