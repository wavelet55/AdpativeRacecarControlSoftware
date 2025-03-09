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
    /// Image Capture Status
    /// Status of the Image Capture Manager.
	/// </summary>
	public class ImageCaptureStatusMsg
	{
	    [ProtoMember(1)]
        /// <summary>
        /// The Image Capture Enabled/Disabled Status:
        /// If there is a error or NumberOfImagesToCapture has
        /// been reached, or the images have been exhausted,
        /// this will be false.
        /// </summary>
        public bool ImageCaptureEnabled { get; set; }

	    [ProtoMember(2)]
        /// <summary>
        /// Set to true when the number of images capture
        /// equals the NumberOfImagesToCapture (assuming
        /// NumberOfImagesToCapture > 0);
        /// </summary>
        public bool ImageCaptureComplete { get; set; }

	    [ProtoMember(3)]
        /// <summary>
        /// Set to true if the source of images is exhausted.
        /// which can occur if images are being pulled from a
        /// directory of images.
        /// </summary>
        public bool EndOfImages { get; set; }

	    [ProtoMember(4)]
        /// <summary>
        /// Total Number of Images Captured Since Start of Videre
        /// </summary>
        public UInt32 TotalNumberOfImagesCaptured { get; set; }

	    [ProtoMember(5)]
        /// <summary>
        /// Total Number of Images Captured Since Image Capture
        /// Enabled... Gets reset to zero when image capture is
        /// disabled.
        /// </summary>
        public UInt32 CurrentNumberOfImagesCaptured { get; set; }


	    [ProtoMember(6)]
        /// <summary>
        /// Average Frames per second base on
        /// CurrentNumberOfImagesCaptured / Time since Last Enabled.
        /// </summary>
        public double AverageFramesPerSecond { get; set; }

	    [ProtoMember(7)]
        /// <summary>
        /// Source of Images for Image Capture
        /// </summary>
        public ImageCaptureSource_e ImageCaptureSource { get; set; }

	    [ProtoMember(8)]
        /// <summary>
        /// The Error Number will be non-zero if there is an
        /// error in the image capture process.  The error
        /// number may be used to indicate what the error is.
        /// </summary>
        public ImageCaptureError_e  ErrorCode { get; set; }


		public ImageCaptureStatusMsg()
		{
			Clear();
		}

		public void Clear()
		{
			ImageCaptureEnabled = false;
			ImageCaptureComplete = false;
			EndOfImages = false;
			TotalNumberOfImagesCaptured = 0;
			CurrentNumberOfImagesCaptured = 0;
            AverageFramesPerSecond = 0;
            ImageCaptureSource = ImageCaptureSource_e.NoChange;
            ErrorCode = 0;
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
				Serializer.Serialize<ImageCaptureStatusMsg>(ms, this);
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
		public static ImageCaptureStatusMsg Deserialize(byte[] b)
		{
			ImageCaptureStatusMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<ImageCaptureStatusMsg>(ms);
			}
			return r;
		}
		#endregion

	}

}


