/* ****************************************************************
 * Vision Bridge
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Nov. 2017
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
	/// CameraParameters Setup Protobuf Message
	/// This message provides specific camera parameters
    /// and control items.
    /// Values of zero or negative numbers typically mean 
    /// use default value or no value passed.
	/// </summary>
	public class CameraParametersSetupPBMsg
	{

		[ProtoMember(1)]
		/// <summary>
		/// ImageCaptureFormat
		/// </summary>
		public CPImageCaptureFormat_e ImageCaptureFormat { get; set; }

		[ProtoMember(2)]
		/// <summary>
		/// Mode
		/// </summary>
		public UInt32 Mode { get; set; }

		[ProtoMember(3)]
		/// <summary>
		/// FrameWidth
		/// </summary>
		public UInt32 FrameWidth { get; set; }

		[ProtoMember(4)]
		/// <summary>
		/// FrameWidth
		/// </summary>
		public UInt32 FrameHeight { get; set; }

		[ProtoMember(5)]
		/// <summary>
		/// FrameRateFPS
		/// </summary>
		public double FrameRateFPS { get; set; }		

		[ProtoMember(6)]
		/// <summary>
		/// Autofocus on --> true
		/// </summary>
		public bool Autofocus { get; set; }

		[ProtoMember(7)]
		/// <summary>
		/// Focus
		/// </summary>
		public double FocusValue { get; set; }		

		[ProtoMember(8)]
		/// <summary>
		/// Brightness
		/// </summary>
		public double Brightness { get; set; }     

		[ProtoMember(9)]
		/// <summary>
		/// Contrast
		/// </summary>
		public double Contrast { get; set; }

		[ProtoMember(10)]
		/// <summary>
		/// Saturation
		/// </summary>
		public double Saturation { get; set; }

		[ProtoMember(11)]
		/// <summary>
		/// Hue
		/// </summary>
		public double Hue { get; set; }

		[ProtoMember(12)]
		/// <summary>
		/// Gain
		/// </summary>
		public double Gain { get; set; }

		[ProtoMember(13)]
		/// <summary>
		/// Exposure
		/// </summary>
		public double Exposure { get; set; }

		[ProtoMember(14)]
		/// <summary>
		/// Exposure
		/// </summary>
		public double WhiteBallanceBlue { get; set; }

		[ProtoMember(15)]
		/// <summary>
		/// Exposure
		/// </summary>
		public double WhiteBallanceRed { get; set; }

		[ProtoMember(16)]
		/// <summary>
		/// ExternalTrigger on --> true
		/// </summary>
		public bool ExternalTrigger { get; set; }


		public CameraParametersSetupPBMsg()
		{
			Clear();
		}

		public void Clear()
		{
			ImageCaptureFormat = CPImageCaptureFormat_e.RGB24;
			Mode = 0;
			FrameWidth = 640;
			FrameHeight = 480;
			FrameRateFPS = 30;
            Autofocus = true;
		    FocusValue = 0;		
		    Brightness = 0;     
		    Contrast = 0;
		    Saturation = 0;
		    Hue = 0;
		    Gain = 0;
		    Exposure = 0;
		    WhiteBallanceBlue = 0;
		    WhiteBallanceRed = 0;
		    ExternalTrigger = false;		
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
				Serializer.Serialize<CameraParametersSetupPBMsg>(ms, this);
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
		public static CameraParametersSetupPBMsg Deserialize(byte[] b)
		{
			CameraParametersSetupPBMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<CameraParametersSetupPBMsg>(ms);
			}
			return r;
		}
		#endregion


	}
}

