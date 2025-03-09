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
	/// Stream Control Proto-buf Message
	/// </summary>
	public class StreamControlPBMsg
	{

        [ProtoMember(1)]
        /// <summary>
        /// Stream Images On/Off
        /// </summary>
        public bool StreamImagesEnabled;

		[ProtoMember(2)]
		/// <summary>
		/// This is the max frame rate for streaming images...
        /// The Vision System may be able to capture and process images
        /// at 30 fps or some rate higher than bandwidth will allow to 
        /// stream images to the ground.  Setting this number to a lower number
        /// say 5 fps will reduce the image rate sent to the ground.       
        /// If zero or less... stream images at max rate.
		/// </summary>
		public double StreamImageFrameRate { get; set; }		

		[ProtoMember(3)]
        /// <summary>
        /// ImageCompressionQuality
        /// Highest quality is 100, lowest is 1 
        /// High Quality means low compression, lower quality
        /// means higher compression.
        /// </summary>
        public UInt32 ImageCompressionQuality { get; set; }

		[ProtoMember(4)]
		/// <summary>
		/// The factor to scale the image down by before
        /// compressing the image and sending it out.
        /// A value of 2.0 will cut the image in half.
        /// The value must be >= 1.0.  1.0 will not change the
        /// image size.
		/// </summary>
		public double StreamImageScaleDownFactor { get; set; }		

		public StreamControlPBMsg()
		{
			Clear();
		}

		public void Clear()
		{
			StreamImagesEnabled = false;
			StreamImageFrameRate = 0;
			ImageCompressionQuality = 50;
            StreamImageScaleDownFactor = 4.0;  
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
				Serializer.Serialize<StreamControlPBMsg>(ms, this);
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
		public static StreamControlPBMsg Deserialize(byte[] b)
		{
			StreamControlPBMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<StreamControlPBMsg>(ms);
			}
			return r;
		}
		#endregion


	}
}

