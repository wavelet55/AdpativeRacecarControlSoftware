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
	/// Image Compression Control Message.
	/// Establishes Image compression parameters
	/// and controls whether or not image compression is used.
	/// </summary>
	public class ImageCompressionControlMsg
	{

		/// <summary>
		/// Supported Image Compression Types.
		/// </summary>
		public enum VisionImageCompressionType_e
		{
			jpeg = 0,
			mpeg = 1,
			gif = 2
		}

		[ProtoMember(1)]
		/// <summary>
		/// Enable or Disable Image Compression
		/// </summary>
		public bool EnableImageCompression { get; set; }

		[ProtoMember(2)]
		/// <summary>
		/// Enable or Disable Transmission of Compressed images.
		/// Note:  Image Compression could be enabled just to log
		/// the compressed images... not to transmit the images.
		/// </summary>
		public bool TransmitCompressedImages { get; set; }


		[ProtoMember(3)]
		/// <summary>
		/// Set the Type of Image Compression.
		/// </summary>
		public VisionImageCompressionType_e ImageComressionType { get; set; }

		[ProtoMember(4)]
		/// <summary>
		/// Set the approximate image compression ratio.
		/// </summary>
		/// <value>The image compression ratio.</value>
		public double ImageCompressionRatio { get; set; }

		[ProtoMember(5)]
		/// <summary>
		/// The Desired framerate through the image compression 
		/// process.  The actual framerate may be less if compression 
		/// time or other factors limit the framerate.
		/// </summary>
		/// <value>The image compression ratio.</value>
		public double FrameRate { get; set; }



		public ImageCompressionControlMsg()
		{
			Clear();
		}


		public void Clear()
		{
			EnableImageCompression = false;
			TransmitCompressedImages = false;
			ImageComressionType = VisionImageCompressionType_e.jpeg;
			ImageCompressionRatio = 0.75;
			FrameRate = 1.0;
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
				Serializer.Serialize<ImageCompressionControlMsg>(ms, this);
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
		public static ImageCompressionControlMsg Deserialize(byte[] b)
		{
			ImageCompressionControlMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<ImageCompressionControlMsg>(ms);
			}
			return r;
		}
		#endregion

	}
}


