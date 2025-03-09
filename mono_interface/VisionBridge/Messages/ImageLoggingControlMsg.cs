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
	/// Logging control message.
	/// Establishes Logging parameters and 
	/// controls whether or not logging is enabled.
	/// It may be desired to disable image logging when
	/// the UAV/Vehicle is not activily persuing a target
	/// to reduce the size the log files. 
	/// </summary>
	public class ImageLoggingControlMsg
	{
		/// <summary>
		/// Vision logging type.
		/// The specific types of logging Falcon Vision Handles.
		/// </summary>
		public enum VisionLoggingType_e
		{
			LogMetaDataOnly = 0,
			LogRawImages = 1,
			LogCompressedImages = 2
		}


		[ProtoMember(1)]
		/// <summary>
		/// Enable or Disable Logging
		/// </summary>
		public bool EnableLogging { get; set; }

		[ProtoMember(2)]
		/// <summary>
		/// Set the Type of Logging... Raw Images, Compressed
		/// images, or only metadata for images.
		/// </summary>
		public VisionLoggingType_e LoggingType { get; set; }

		public ImageLoggingControlMsg()
		{
			Clear();
		}

		public ImageLoggingControlMsg(bool enable, VisionLoggingType_e logType)
		{
			EnableLogging = enable;
			LoggingType = logType;
		}

		public void Clear()
		{
			EnableLogging = false;
			LoggingType = VisionLoggingType_e.LogRawImages;
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
				Serializer.Serialize<ImageLoggingControlMsg>(ms, this);
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
		public static ImageLoggingControlMsg Deserialize(byte[] b)
		{
			ImageLoggingControlMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<ImageLoggingControlMsg>(ms);
			}
			return r;
		}
		#endregion

	}
}

