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

    public enum FeatureMatchingProcCmd_e
    {
        NullCmd = 0,            //Remain in current state
        Reset = 1,              //Go To reset State
        ClearImageSet = 2,      //Go to Reset after clearing directory of Images
        StreamImages = 3,       //Stream Images and wait for Capture Image
        CaptureImage = 4,       //Capture and verify image
        SetImageOk = 5,         //Causes image to be stored... goto StreamImages
        RejectImage = 6,        //Reject image and goto StreamImages
        RunImageProcess = 7,    //Run Image Prcess against Feature Set.
    };

    //A range of different feature extraction type routings can be
    //selected.
    public enum FeatureExtractionTypeRoutine_e
    {
        ORB,
        SIFT,
        SURF,
    };

    //A range of different feature extraction type routings can be
    //selExtractionected.
    public enum FeatureMatchTypeRoutine_e
    {
        BruteForce,
        FLANN,
    };

    public enum FMImagePostProcessMethod_e
    {
        None,
        GenFeatureMap,    //Generate an image showing mapping between Obj Image and Test image
        MarkObjectFoundRect,
        MarkObjectFoundCircle,
        GenFeatureMapAndMarkObjRect,
        GenFeatureMapAndMarkObjCircle,
    };



	[ProtoContract]
	/// <summary>
	/// Feature Match Processor Control message.
	/// </summary>
	public class FeatureMatchProcCtrlPBMsg
	{

        [ProtoMember(1)]
        public FeatureMatchingProcCmd_e FeatureMatchingProcCmd { get; set; }

        [ProtoMember(2)]
        public FeatureExtractionTypeRoutine_e FeatureExtractionTypeRoutine { get; set; }

        [ProtoMember(3)]
        public FeatureMatchTypeRoutine_e FeatureMatchTypeRoutine { get; set; }

        [ProtoMember(4)]
        public FMImagePostProcessMethod_e FMImagePostProcessMethod  { get; set; }

       //Use NVidia GPU/Cuda processing or standard processing.
        //Helps with measuring perfomance differences between the two.
        [ProtoMember(5)]
        public bool UseGPUProcessing;

        //Generic Parameters for the processing routines
        [ProtoMember(10)]
        public int ParamI_1  { get; set; }

        [ProtoMember(11)]
        public int ParamI_2  { get; set; }

        [ProtoMember(12)]
        public int ParamI_3  { get; set; }

        [ProtoMember(13)]
        public int ParamI_4  { get; set; }

        [ProtoMember(14)]
        public int ParamI_5  { get; set; }

        [ProtoMember(15)]
        public int ParamI_6  { get; set; }

        [ProtoMember(16)]
        public int ParamI_7  { get; set; }

        [ProtoMember(17)]
        public int ParamI_8  { get; set; }

        [ProtoMember(18)]
        public int ParamI_9  { get; set; }

        [ProtoMember(20)]
        public double ParamF_10  { get; set; }

        [ProtoMember(21)]
        public double ParamF_11  { get; set; }

        [ProtoMember(22)]
        public double ParamF_12  { get; set; }

        [ProtoMember(23)]
        public double ParamF_13  { get; set; }

        [ProtoMember(24)]
        public double ParamF_14  { get; set; }

        [ProtoMember(25)]
        public double ParamF_15  { get; set; }

        [ProtoMember(26)]
        public double ParamF_16  { get; set; }

        [ProtoMember(27)]
        public double ParamF_17  { get; set; }

        [ProtoMember(28)]
        public double ParamF_18  { get; set; }

        [ProtoMember(29)]
        public double ParamF_19  { get; set; }


		public FeatureMatchProcCtrlPBMsg()
		{
			Clear();
		}

		public void Clear()
		{
            FeatureMatchingProcCmd = FeatureMatchingProcCmd_e.NullCmd;
            FeatureExtractionTypeRoutine = FeatureExtractionTypeRoutine_e.ORB;
            FeatureMatchTypeRoutine = FeatureMatchTypeRoutine_e.BruteForce;
            FMImagePostProcessMethod = FMImagePostProcessMethod_e.None;
            UseGPUProcessing = false;

            ParamI_1 = 0;
            ParamI_2 = 0;
            ParamI_3 = 0;
            ParamI_4 = 0;
            ParamI_5 = 0;
            ParamI_6 = 0;
            ParamI_7 = 0;
            ParamI_8 = 0;
            ParamI_9 = 0;

            ParamF_10 = 0;
            ParamF_11 = 0;
            ParamF_12 = 0;
            ParamF_13 = 0;
            ParamF_14 = 0;
            ParamF_15 = 0;
            ParamF_16 = 0;
            ParamF_17 = 0;
            ParamF_18 = 0;
            ParamF_19 = 0;
		}

		//----------------------------------------------------------------------
		//----------------------------------------------------------------------
		#region SerializeResponse
		/// <summary>
		/// Serialize FeatureMatchProcCtrlPBMsg to byte array.
		/// </summary>
		public byte[] Serialize()
		{
			byte[] b = null;
			using (var ms = new MemoryStream())
			{
				Serializer.Serialize<FeatureMatchProcCtrlPBMsg>(ms, this);
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
		public static FeatureMatchProcCtrlPBMsg Deserialize(byte[] b)
		{
			FeatureMatchProcCtrlPBMsg r;
			using (var ms = new MemoryStream(b))
			{
				r = Serializer.Deserialize<FeatureMatchProcCtrlPBMsg>(ms);
			}
			return r;
		}
		#endregion

	}
}

