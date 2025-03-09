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


    public enum FeatureMatchingState_e
    {
        Reset,
        WaitForStart,
        StreamImages,
        ImageValidate,
        ImageCapturedWait,
        FMProcess,
        FMComplete,
        FMError
    };



	[ProtoContract]
	/// <summary>
	/// Feature Match Processor Control message.
	/// </summary>
	public class FeatureMatchProcStatusPBMsg
	{

        [ProtoMember(1)]
        public FeatureMatchingState_e FeatureMatchingState { get; set; }

        [ProtoMember(2)]
        public FeatureExtractionTypeRoutine_e FeatureExtractionTypeRoutine { get; set; }

        [ProtoMember(3)]
        public FeatureMatchTypeRoutine_e FeatureMatchTypeRoutine { get; set; }

        [ProtoMember(4)]
        public string StatusMessage { get; set; }

        [ProtoMember(5)]
        public int NumberOfImagesCaptured  { get; set; }

        [ProtoMember(6)]
        public double ProcessTimer_1   { get; set; }

        [ProtoMember(7)]
        public double ProcessTimer_2   { get; set; }


        //Generic Parameters for the processing routines
        [ProtoMember(10)]
        public int StatusValI_1  { get; set; }

        [ProtoMember(11)]
        public int StatusValI_2  { get; set; }

        [ProtoMember(12)]
        public int StatusValI_3  { get; set; }

        [ProtoMember(13)]
        public int StatusValI_4  { get; set; }

        [ProtoMember(14)]
        public int StatusValI_5  { get; set; }

        [ProtoMember(15)]
        public int StatusValI_6  { get; set; }

        [ProtoMember(16)]
        public int StatusValI_7  { get; set; }

        [ProtoMember(17)]
        public int StatusValI_8  { get; set; }

        [ProtoMember(18)]
        public int StatusValI_9  { get; set; }

        [ProtoMember(20)]
        public double StatusValF_10  { get; set; }

        [ProtoMember(21)]
        public double StatusValF_11  { get; set; }

        [ProtoMember(22)]
        public double StatusValF_12  { get; set; }

        [ProtoMember(23)]
        public double StatusValF_13  { get; set; }

        [ProtoMember(24)]
        public double StatusValF_14  { get; set; }

        [ProtoMember(25)]
        public double StatusValF_15  { get; set; }

        [ProtoMember(26)]
        public double StatusValF_16  { get; set; }

        [ProtoMember(27)]
        public double StatusValF_17  { get; set; }

        [ProtoMember(28)]
        public double StatusValF_18  { get; set; }

        [ProtoMember(29)]
        public double StatusValF_19  { get; set; }



		public FeatureMatchProcStatusPBMsg()
		{
			Clear();
		}

		public void Clear()
		{
            FeatureMatchingState = FeatureMatchingState_e.Reset;
            FeatureExtractionTypeRoutine = FeatureExtractionTypeRoutine_e.ORB;
            FeatureMatchTypeRoutine = FeatureMatchTypeRoutine_e.BruteForce;
            StatusMessage = "";
            NumberOfImagesCaptured = 0;
            ProcessTimer_1 = 0;
            ProcessTimer_2 = 0;

            StatusValI_1 = 0;
            StatusValI_2 = 0;
            StatusValI_3 = 0;
            StatusValI_4 = 0;
            StatusValI_5 = 0;
            StatusValI_6 = 0;
            StatusValI_7 = 0;
            StatusValI_8 = 0;
            StatusValI_9 = 0;

            StatusValF_10 = 0;
            StatusValF_11 = 0;
            StatusValF_12 = 0;
            StatusValF_13 = 0;
            StatusValF_14 = 0;
            StatusValF_15 = 0;
            StatusValF_16 = 0;
            StatusValF_17 = 0;
            StatusValF_18 = 0;
            StatusValF_19 = 0;
		}

		//----------------------------------------------------------------------
		//----------------------------------------------------------------------
		#region SerializeResponse
		/// <summary>
		/// Serialize FeatureMatchProcStatusPBMsg to byte array.
		/// </summary>
		public byte[] Serialize()
		{
			byte[] b = null;
			using (var ms = new MemoryStream())
			{
				Serializer.Serialize<FeatureMatchProcStatusPBMsg>(ms, this);
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
		public static FeatureMatchProcStatusPBMsg Deserialize(byte[] b)
		{
			FeatureMatchProcStatusPBMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<FeatureMatchProcStatusPBMsg>(ms);
			}
			return r;
		}
		#endregion

	}
}

