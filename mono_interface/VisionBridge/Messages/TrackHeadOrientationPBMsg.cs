/* ****************************************************************
 * Vision Bridge
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: June 2018
 * 
 *******************************************************************/
using System;
using ProtoBuf;
using System.IO;

namespace VisionBridge.Messages
{
	
	[ProtoContract]
	/// <summary>
	/// Head Tracking Orientation Message.
	/// </summary>
	public class TrackHeadOrientationPBMsg
	{
		[ProtoMember(1)]
		public double HeadOrientationQuaternion_W { get; set; }	

		[ProtoMember(2)]
		public double HeadOrientationQuaternion_X { get; set; }	

		[ProtoMember(3)]
		public double HeadOrientationQuaternion_Y { get; set; }	
	
		[ProtoMember(4)]
		public double HeadOrientationQuaternion_Z { get; set; }	
	
		[ProtoMember(5)]
		public double HeadTranslationRVec_X { get; set; }	
	
		[ProtoMember(6)]
		public double HeadTranslationRVec_Y { get; set; }	
	
		[ProtoMember(7)]
		public double HeadTranslationRVec_Z { get; set; }	

		[ProtoMember(8)]
		public bool IsDataValid { get; set; }	

		[ProtoMember(9)]
		public UInt32 ImageNumber { get; set; }	

		[ProtoMember(10)]
		public double ImageCaptureTimeStampSec { get; set; }	

		[ProtoMember(11)]
		public double VidereTimeStampSec { get; set; }	

		public TrackHeadOrientationPBMsg()
		{
			Clear();
		}

		public void Clear()
		{
            HeadOrientationQuaternion_W = 0;		
            HeadOrientationQuaternion_X = 0;		
            HeadOrientationQuaternion_Y = 0;
            HeadOrientationQuaternion_Z = 0;
            HeadTranslationRVec_X = 0;
            HeadTranslationRVec_Y = 0;
            HeadTranslationRVec_Z = 0;
            IsDataValid = false;
            ImageNumber = 0;
            ImageCaptureTimeStampSec = 0;
            VidereTimeStampSec = 0;
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
				Serializer.Serialize<TrackHeadOrientationPBMsg>(ms, this);
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
		public static TrackHeadOrientationPBMsg Deserialize(byte[] b)
		{
			TrackHeadOrientationPBMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<TrackHeadOrientationPBMsg>(ms);
			}
			return r;
		}
		#endregion


	}
}

