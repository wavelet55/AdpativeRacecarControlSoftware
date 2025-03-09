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
	/// Camera Calibration Control message.
	/// </summary>
	public class CameraCalControlMsg
	{
        public enum CameraCalCmd_e
        {
            NullCmd = 0,        //Remain in current state
            Reset,              //Go To reset State
            ClearImageSet,      //Go to Reset after clearing directory of Images
            StreamImages,       //Stream Images and wait for Capture Image
            CaptureImage,       //Capture and verify image
            SetImageOk,         //Causes image to be stored... goto StreamImages
            RejectImage,        //Reject image and goto StreamImages
            RunCalProcess       //Run Calibration against Image Set.
        }

        [ProtoMember(1)]
		/// <summary>
		/// The Camera Calibration Type... 
		/// </summary>
        public CameraCalibrationType_e CameraCalibrationType { get; set; }

        [ProtoMember(2)]
		/// <summary>
		/// The Camera Calibration Type... 
		/// </summary>
        public CameraCalCmd_e CameraCalCmd { get; set; }

		[ProtoMember(3)]
		/// <summary>
		/// The base file name of the Camera Calibration Data 
		/// </summary>
		public string CameraCalBaseFilename { get; set; }

		[ProtoMember(4)]
		/// <summary>
		/// Number of Checker Board (or similar patter) rows 
		/// </summary>
		public int NumberOfRows { get; set; }

		[ProtoMember(5)]
		/// <summary>
		/// Number of Checker Board (or similar patter) columns 
		/// </summary>
		public int NumberOfCols { get; set; }

		[ProtoMember(6)]
		/// <summary>
		/// Number of Checker Board (or similar patter) columns 
		/// </summary>
		public double SquareSizeMilliMeters { get; set; }

		[ProtoMember(7)]
		/// <summary>
		/// Camera Mounting Yaw Correction 
		/// </summary>
		public double YawCorrectionDegrees { get; set; }

		[ProtoMember(8)]
		/// <summary>
		/// Camera Mounting Pitch Correction 
		/// </summary>
		public double PitchCorrectionDegrees { get; set; }

		[ProtoMember(9)]
		/// <summary>
		/// Camera Mounting Roll Correction 
		/// </summary>
		public double RollCorrectionDegrees { get; set; }

		[ProtoMember(10)]
		/// <summary>
		/// Camera Mounting Del-X Correction 
		/// </summary>
		public double DelXCorrectionCentiMeters { get; set; }

		[ProtoMember(11)]
		/// <summary>
		/// Camera Mounting Del-Y Correction 
		/// </summary>
		public double DelYCorrectionCentiMeters { get; set; }

		[ProtoMember(12)]
		/// <summary>
		/// Camera Mounting Del-Z Correction 
		/// </summary>
		public double DelZCorrectionCentiMeters { get; set; }

		public CameraCalControlMsg()
		{
			Clear();
		}

		public void Clear()
		{
			CameraCalibrationType = CameraCalibrationType_e.CCT_2DPlaneCheckerBoard;
			CameraCalCmd = CameraCalCmd_e.NullCmd;
			CameraCalBaseFilename = "CameraCalData";
            NumberOfCols = 6;
            NumberOfRows = 7;
            SquareSizeMilliMeters = 25.4;
            YawCorrectionDegrees = 0;
            PitchCorrectionDegrees = 0;
            RollCorrectionDegrees = 0;
            DelXCorrectionCentiMeters = 0;
            DelYCorrectionCentiMeters = 0;
            DelZCorrectionCentiMeters = 0;
		}

		//----------------------------------------------------------------------
		//----------------------------------------------------------------------
		#region SerializeResponse
		/// <summary>
		/// Serialize CameraCalControlMsg to byte array.
		/// </summary>
		public byte[] Serialize()
		{
			byte[] b = null;
			using (var ms = new MemoryStream())
			{
				Serializer.Serialize<CameraCalControlMsg>(ms, this);
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
		public static CameraCalControlMsg Deserialize(byte[] b)
		{
			CameraCalControlMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<CameraCalControlMsg>(ms);
			}
			return r;
		}
		#endregion

	}
}

