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
	/// Camera Calibration Status message.
	/// </summary>
	public class CameraCalStatusMsg
	{
        public enum CameraCalState_e
        {
            Reset = 0,
            WaitForStart = 1,
            StreamImages = 2,
            ImageValidate = 3,
            ImageCapturedWait = 4,
            CalProcess = 5,
            CalComplete = 6,
            CalError = 7,
        }

        [ProtoMember(1)]
		/// <summary>
		/// The Camera Calibration State of Operation 
		/// </summary>
        public CameraCalState_e CameraCalState { get; set; }

        [ProtoMember(2)]
		/// <summary>
		/// The number of images capture for processing. 
		/// </summary>
        public int NumberOfImagesCaptured { get; set; }

		[ProtoMember(3)]
		/// <summary>
		/// Message from the Camera Calibration Process
		/// </summary>
		public string CameraCalMsg { get; set; }

        [ProtoMember(4)]
		/// <summary>
		/// Flag indicating the image is ok... found chess board. 
		/// </summary>
        public bool ImageOk { get; set; }

		public CameraCalStatusMsg()
		{
			Clear();
		}

		public void Clear()
		{
            CameraCalState = CameraCalState_e.Reset;
            NumberOfImagesCaptured = 0;
            CameraCalMsg = "";
            ImageOk = false;
		}

		//----------------------------------------------------------------------
		//----------------------------------------------------------------------
		#region SerializeResponse
		/// <summary>
		/// Serialize CameraCalStatusMsg to byte array.
		/// </summary>
		public byte[] Serialize()
		{
			byte[] b = null;
			using (var ms = new MemoryStream())
			{
				Serializer.Serialize<CameraCalStatusMsg>(ms, this);
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
		public static CameraCalStatusMsg Deserialize(byte[] b)
		{
			CameraCalStatusMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<CameraCalStatusMsg>(ms);
			}
			return r;
		}
		#endregion

	}
}

