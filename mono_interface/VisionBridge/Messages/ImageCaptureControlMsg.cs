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
    /// Image Capture Control
    /// Controls Image Capture and provides configuration
    /// parameters for Image Capture.
    /// Note:  The configureation parameters are only read
    /// and updated/set by Videre when the Image Capture is
    /// disabled.
	/// </summary>
	public class ImageCaptureControlMsg
	{
	    [ProtoMember(1)]
        /// <summary>
        /// The Image Capture On flag must be true
        /// to capture images and do any of the processing
        /// of images.  Setting this flag to false will disable
        /// all other image processing.
        /// Note:  The configureation parameters are only read
        /// and updated/set by Videre when the Image Capture is
        /// disabled.
        /// </summary>
        public bool ImageCaptureEnabled { get; set; }

	    [ProtoMember(2)]
        /// <summary>
        /// If NumberOfImagesToCapture is greater than zero,
        /// the Vision System will capture the set number of images,
        /// process the images, and then disable ImageCapture.
        /// If NumberOfImagesToCapture is zero or less... the Vision
        /// System will continue capturing and processing images until
        /// ImageCaptureEnabled is disabled by the user.
        /// To Restart Image Capture after the NumberOfImagesToCapture
        /// has been reached... Disable and then re-enable Image
        /// capture with the ImageCaptureEnabled flag.
        /// </summary>
        public UInt32 NumberOfImagesToCapture { get; set; }

	    [ProtoMember(3)]
        /// <summary>
        /// Desired Frames per second.
        /// In general the frame rate will be controlled by the time
        /// to do the image processing.  If image processing is disabled
        /// or very quick, this value can be used to slow down the image
        /// capture and processing rate.  Set to a higher value to get the
        /// max frame rate that is supported by the image processing time.
        /// </summary>
        public double DesiredFramesPerSecond { get; set; }

	    [ProtoMember(4)]
        /// <summary>
        /// Desired Image Width and Height
        /// Where Image Width and Height can be controlled, use
        /// these parameters.  If set to zero, the Videre Config
        /// info will be used.
        /// </summary>
        public UInt32  DesiredImageWidth { get; set; }

	    [ProtoMember(5)]
        public UInt32  DesiredImageHeight { get; set; }

	    [ProtoMember(6)]
        /// <summary>
        /// Source of Images for Image Capture
        /// </summary>
        public ImageCaptureSource_e ImageCaptureSource { get; set; }


	    [ProtoMember(7)]
        /// <summary>
        /// Image Capture Format
        /// These are WebCam dependent in terms of the webcam's 
        /// capabilities.
        /// </summary>
        public CPImageCaptureFormat_e ImageCaptureFormat { get; set; }

	    [ProtoMember(8)]
        /// <summary>
        /// Primary Configuration String for the ImageCaptureSource.
        /// This could be the Device number for the WebCam,
        ///  or it could be the Directory of Image Files.
        /// If this is empty the Videre Config info will be used.
        /// </summary>
        public string ImageCaptureSourceConfigPri { get; set; }

	    [ProtoMember(9)]
        /// <summary>
        /// Secondary Configuration String for the ImageCaptureSource.
        /// This could be the Device number for the WebCam,
        ///  or it could be the Directory of Image Files.
        /// If this is empty the Videre Config info will be used.
        /// </summary>
        public string ImageCaptureSourceConfigSec { get; set; }

	    [ProtoMember(10)]
        /// <summary>
        /// When images are being captured from a finite
        /// source such as a directory of image files,
        /// if this flag is true, Image capture will restart
        /// capture from the start of the source after reaching
        /// the end.
        /// </summary>
        public bool ImageSourceLoopAround { get; set; }

	    [ProtoMember(11)]
        /// <summary>
        /// Auto Focus Enable
        /// Used to enable or disable auto focus on a camera
        /// not all cameras support auto focus.
        /// </summary>
        public bool AutoFocusEnable { get; set; }


		public ImageCaptureControlMsg()
		{
			Clear();
		}

		public void Clear()
		{
			ImageCaptureEnabled = false;
			NumberOfImagesToCapture = 0;
			DesiredFramesPerSecond = 100;
			DesiredImageWidth = 0;
			DesiredImageHeight = 0;
            ImageCaptureSource = ImageCaptureSource_e.NoChange;
            ImageCaptureFormat = CPImageCaptureFormat_e.Unknown;
			ImageCaptureSourceConfigPri = "";
            ImageCaptureSourceConfigSec = "";
            ImageSourceLoopAround = false;
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
				Serializer.Serialize<ImageCaptureControlMsg>(ms, this);
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
		public static ImageCaptureControlMsg Deserialize(byte[] b)
		{
			ImageCaptureControlMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<ImageCaptureControlMsg>(ms);
			}
			return r;
		}
		#endregion


	}

}
