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
	/// Vision Processing Control
    /// Sets many of the primary Vision/Image Processing Control
    /// Flags and Parameters.
	/// </summary>
	public class VisionProcessingControlMsg
	{

		[ProtoMember(1)]
		/// <summary>
		/// The Image Capture On flag must be true
        /// to capture images and do any of the processing 
        /// of images.  Setting this flag to false will disable 
        /// all other image processing.
		/// </summary>
		public bool ImageCaptureEnabled { get; set; }

        [ProtoMember(2)]
		/// <summary>
		/// If NumberOfImagesToCapture is greater than zero,
        /// the Vision System will capture the set number of images,
        /// process the images, and then disable ImageCapture
        /// If NumberOfImagesToCapture is zero or less... the Vision
        /// System will continue capturing and processing images until
        /// ImageCaptureEnabled is disabled by the user.
		/// </summary>
        public int NumberOfImagesToCapture;

        [ProtoMember(3)]
		/// <summary>
		/// Desired Frames per second.
        /// In general the frame rate will be controlled by the time
        /// to do the image processing.  If image processing is disabled
        /// or very quick, this value can be used to slow down the image 
        /// capture and processing rate.  Set to a higher value to get the 
        /// max frame rate that is supported by the image processing time.
		/// </summary>
        public double DesiredFramesPerSecond;


		[ProtoMember(4)]
		/// <summary>
		/// GPU Processing On
        /// Turns on/off all GPU Image Processing.   
        /// Finer level control of the Image processing is given below.
		/// </summary>
		public bool GPUProcessingEnabled { get; set; }

		[ProtoMember(5)]
		/// <summary>
        /// Target Image Processing On/Off
		/// </summary>
		public bool TargetImageProcessingEnabled { get; set; }

        [ProtoMember(6)]
        /// <summary>
        /// Vision Processing Mode
        /// This is the High-Level Vision / Image Processing mode of operation.
        /// </summary>
        public VisionProcessingMode_e VisionProcessingMode;

        /*****************************************
         * See def at:  CommonDefs:  TargetProcessingMode_e
        public enum TargetProcessingModeEnum
        {
            TgtProcMode_None = 0,
            TgtProcMode_Std = 1,        //Blob Target Detection using the OpenCV Blob Detector
            TgtProcMode_Blob = 2,       //Blob Target Detection using the Huntsman/JHART Blob Detector
        }
         * ****************************************/

        [ProtoMember(7)]
        /// <summary>
        /// Target Image Processing Mode
        /// Various types of Target Processing could be supported,
        /// this enum selects the active Target Processing Mode.
        /// </summary>
        public TargetProcessingMode_e TargetProcessingMode;

		[ProtoMember(8)]
		/// <summary>
        /// Target Image Processing On/Off
		/// </summary>
		public bool GPSDeniedProcessingEnabled { get; set; }

        public enum GPSDeniedProcessingModeEnum
        {
            GpsDeniedMode_None = 0,
            GpsDeniedMode_Std = 1,
        }

        [ProtoMember(9)]
        /// <summary>
        /// GPS Denied Processing Mode
        /// Various types of Target Processing could be supported,
        /// this enum selects the active Target Processing Mode.
        /// </summary>
        public GPSDeniedProcessingModeEnum GPSDeniedProcessingMode;

		[ProtoMember(10)]
		/// <summary>
        /// Record Images On/Off
		/// </summary>
		public bool RecordImagesEnabled { get; set; }
        
		[ProtoMember(11)]
		/// <summary>
        /// Stream Images On/Off
		/// </summary>
		public bool StreamImagesEnabled { get; set; }
 
		public VisionProcessingControlMsg()
		{
			Clear();
		}

		public void Clear()
		{
            ImageCaptureEnabled = false;
            NumberOfImagesToCapture = 0;
            DesiredFramesPerSecond = 10.0;
            GPUProcessingEnabled = false;
            TargetImageProcessingEnabled = false;
            TargetProcessingMode = TargetProcessingMode_e.TgtProcMode_Std;
            GPSDeniedProcessingEnabled = false;
            GPSDeniedProcessingMode = GPSDeniedProcessingModeEnum.GpsDeniedMode_Std;
            RecordImagesEnabled = false;
            StreamImagesEnabled = false;
		}

		//----------------------------------------------------------------------
		//----------------------------------------------------------------------
		#region SerializeResponse
		/// <summary>
		/// Serialize VisionProcessingControlMsg to byte array.
		/// </summary>
		public byte[] Serialize()
		{
			byte[] b = null;
			using (var ms = new MemoryStream())
			{
				Serializer.Serialize<VisionProcessingControlMsg>(ms, this);
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
		public static VisionProcessingControlMsg Deserialize(byte[] b)
		{
			VisionProcessingControlMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<VisionProcessingControlMsg>(ms);
			}
			return r;
		}
		#endregion

	}
}

