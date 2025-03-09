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
	/// Target Parmeters Message
    /// Provides Parameters for the given Target used by Image Processing
    /// to identify the given target type.
	/// </summary>
	public class TargetParametersMsg
	{
		[ProtoMember(1)]
		/// <summary>
        /// Target Type Code
        /// This is an agreeded upont code for the type of target
        /// (Building, car, human...)  Since the codes will change based upon
        /// mission and image processing... an enum is not used.
        /// Image processing will return this code for the type of target it finds.
		/// </summary>
		public UInt32 TargetTypeCode { get; set; }
 
        [ProtoMember(2)]
		/// <summary>
        /// Flag that indicates whether or not the Target Type is 
        /// tactical or not.
		/// </summary>
        public bool IsTaticalTarget { get; set; }

		[ProtoMember(3)]
		/// <summary>
		/// Estimated Target Size Or Width
        /// How this number is interpreted is based on the Target Type.
		/// </summary>
		public double TargetSizeOrWidth { get; set; }		

		[ProtoMember(4)]
		/// <summary>
		/// Estimated Target Size Or Width
        /// How this number is interpreted is based on the Target Type.
		/// </summary>
		public double TargetPerimeterOrLenght { get; set; }		

		[ProtoMember(5)]
		/// <summary>
		/// Estimated Target Infrared Heat Gradient
		/// </summary>
		public double TargetIRHeatGradient { get; set; }		

		[ProtoMember(6)]
		/// <summary>
		/// Estimated Target Infrared Size
		/// </summary>
		public double TargetIRSize { get; set; }		

		[ProtoMember(7)]
		/// <summary>
		/// Target EO RGB Color Code
		/// </summary>
		public UInt32 TargetRGBColorCode { get; set; }		

		public TargetParametersMsg()
		{
			Clear();
		}

		public void Clear()
		{
			TargetTypeCode = 0;
			IsTaticalTarget = true;
			TargetSizeOrWidth = 0;
			TargetPerimeterOrLenght = 0;  
			TargetIRHeatGradient = 0;  
			TargetIRSize = 0;
            TargetRGBColorCode = 0;
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
				Serializer.Serialize<TargetParametersMsg>(ms, this);
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
		public static TargetParametersMsg Deserialize(byte[] b)
		{
			TargetParametersMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<TargetParametersMsg>(ms);
			}
			return r;
		}
		#endregion


	}
}

