
// This file has been generated by the GUI designer. Do not modify.

public partial class MainWindow
{
	private global::Gtk.VBox VertBox;
	
	private global::Gtk.HBox hbox4;
	
	private global::Gtk.VBox vbox1;
	
	private global::Gtk.HBox hbox2;
	
	private global::Gtk.Label label3;
	
	private global::Gtk.Label label1;
	
	private global::Gtk.Label label2;
	
	private global::Gtk.HBox hbox6;
	
	private global::Gtk.Button btnZMQConnect;
	
	private global::Gtk.TextView tbTcpAddrInput;
	
	private global::Gtk.TextView tbThisCompTcpAddrInput;
	
	private global::Gtk.HBox hbox1;
	
	private global::Gtk.Button StartVideoButton;
	
	private global::Gtk.Button StopVideoButton;
	
	private global::Gtk.Button GPUStart;
	
	private global::Gtk.Button GPUStop;
	
	private global::Gtk.HBox hbox7;
	
	private global::Gtk.Button startRecording;
	
	private global::Gtk.Button stopRecording;
	
	private global::Gtk.Button StartStreamButton;
	
	private global::Gtk.Button StopStreamButton;
	
	private global::Gtk.HBox hbox8;
	
	private global::Gtk.Button InfoButton;
	
	private global::Gtk.Button System;
	
	private global::Gtk.Button KillButton;
	
	private global::VisionSystemGtkViewer.ManagerStats mgrStatsComm;
	
	private global::Gtk.HBox hbox11;
	
	private global::Gtk.VBox vbox7;
	
	private global::Gtk.TextView textOutputWindow;
	
	private global::Gtk.TextView textVisionOutput;
	
	private global::VisionSystemGtkViewer.ManagerStats mgrStatsImageCapture;
	
	private global::Gtk.HBox hbox21;
	
	private global::Gtk.ScrolledWindow scrolledwindow1;
	
	private global::Gtk.Image image1;
	
	private global::Gtk.VBox vbox6;
	
	private global::VisionSystemGtkViewer.ManagerStats mgrStatsImageProc;
	
	private global::Gtk.HBox hbox22;
	
	private global::Gtk.ScrolledWindow scrolledwindow2;
	
	private global::Gtk.Image image2;
	
	private global::Gtk.VBox vbox8;
	
	private global::VisionSystemGtkViewer.ManagerStats mgrStatsSteamRecord;

	protected virtual void Build ()
	{
		global::Stetic.Gui.Initialize (this);
		// Widget MainWindow
		this.HeightRequest = 50;
		this.Name = "MainWindow";
		this.Title = global::Mono.Unix.Catalog.GetString ("MainWindow");
		this.WindowPosition = ((global::Gtk.WindowPosition)(4));
		// Container child MainWindow.Gtk.Container+ContainerChild
		this.VertBox = new global::Gtk.VBox ();
		this.VertBox.Name = "VertBox";
		// Container child VertBox.Gtk.Box+BoxChild
		this.hbox4 = new global::Gtk.HBox ();
		this.hbox4.Name = "hbox4";
		this.hbox4.Spacing = 6;
		// Container child hbox4.Gtk.Box+BoxChild
		this.vbox1 = new global::Gtk.VBox ();
		this.vbox1.Name = "vbox1";
		this.vbox1.Spacing = 6;
		// Container child vbox1.Gtk.Box+BoxChild
		this.hbox2 = new global::Gtk.HBox ();
		this.hbox2.Name = "hbox2";
		this.hbox2.Spacing = 6;
		// Container child hbox2.Gtk.Box+BoxChild
		this.label3 = new global::Gtk.Label ();
		this.label3.Name = "label3";
		this.label3.LabelProp = global::Mono.Unix.Catalog.GetString (" ");
		this.hbox2.Add (this.label3);
		global::Gtk.Box.BoxChild w1 = ((global::Gtk.Box.BoxChild)(this.hbox2 [this.label3]));
		w1.Position = 0;
		// Container child hbox2.Gtk.Box+BoxChild
		this.label1 = new global::Gtk.Label ();
		this.label1.Name = "label1";
		this.label1.Xpad = 10;
		this.label1.Xalign = 1F;
		this.label1.LabelProp = global::Mono.Unix.Catalog.GetString ("Vision System IP Addrss");
		this.hbox2.Add (this.label1);
		global::Gtk.Box.BoxChild w2 = ((global::Gtk.Box.BoxChild)(this.hbox2 [this.label1]));
		w2.Position = 1;
		w2.Expand = false;
		w2.Fill = false;
		w2.Padding = ((uint)(25));
		// Container child hbox2.Gtk.Box+BoxChild
		this.label2 = new global::Gtk.Label ();
		this.label2.Name = "label2";
		this.label2.Xpad = 10;
		this.label2.LabelProp = global::Mono.Unix.Catalog.GetString ("This Computer IP Addr");
		this.hbox2.Add (this.label2);
		global::Gtk.Box.BoxChild w3 = ((global::Gtk.Box.BoxChild)(this.hbox2 [this.label2]));
		w3.Position = 2;
		w3.Expand = false;
		w3.Fill = false;
		w3.Padding = ((uint)(100));
		this.vbox1.Add (this.hbox2);
		global::Gtk.Box.BoxChild w4 = ((global::Gtk.Box.BoxChild)(this.vbox1 [this.hbox2]));
		w4.Position = 0;
		w4.Expand = false;
		w4.Fill = false;
		// Container child vbox1.Gtk.Box+BoxChild
		this.hbox6 = new global::Gtk.HBox ();
		this.hbox6.Name = "hbox6";
		this.hbox6.Spacing = 6;
		// Container child hbox6.Gtk.Box+BoxChild
		this.btnZMQConnect = new global::Gtk.Button ();
		this.btnZMQConnect.CanFocus = true;
		this.btnZMQConnect.Name = "btnZMQConnect";
		this.btnZMQConnect.UseUnderline = true;
		this.btnZMQConnect.Label = global::Mono.Unix.Catalog.GetString ("Connect");
		this.hbox6.Add (this.btnZMQConnect);
		global::Gtk.Box.BoxChild w5 = ((global::Gtk.Box.BoxChild)(this.hbox6 [this.btnZMQConnect]));
		w5.Position = 0;
		w5.Expand = false;
		w5.Fill = false;
		// Container child hbox6.Gtk.Box+BoxChild
		this.tbTcpAddrInput = new global::Gtk.TextView ();
		this.tbTcpAddrInput.CanFocus = true;
		this.tbTcpAddrInput.Name = "tbTcpAddrInput";
		this.tbTcpAddrInput.CursorVisible = false;
		this.tbTcpAddrInput.AcceptsTab = false;
		this.hbox6.Add (this.tbTcpAddrInput);
		global::Gtk.Box.BoxChild w6 = ((global::Gtk.Box.BoxChild)(this.hbox6 [this.tbTcpAddrInput]));
		w6.Position = 1;
		// Container child hbox6.Gtk.Box+BoxChild
		this.tbThisCompTcpAddrInput = new global::Gtk.TextView ();
		this.tbThisCompTcpAddrInput.CanFocus = true;
		this.tbThisCompTcpAddrInput.Name = "tbThisCompTcpAddrInput";
		this.tbThisCompTcpAddrInput.CursorVisible = false;
		this.tbThisCompTcpAddrInput.AcceptsTab = false;
		this.hbox6.Add (this.tbThisCompTcpAddrInput);
		global::Gtk.Box.BoxChild w7 = ((global::Gtk.Box.BoxChild)(this.hbox6 [this.tbThisCompTcpAddrInput]));
		w7.Position = 2;
		this.vbox1.Add (this.hbox6);
		global::Gtk.Box.BoxChild w8 = ((global::Gtk.Box.BoxChild)(this.vbox1 [this.hbox6]));
		w8.Position = 1;
		w8.Expand = false;
		w8.Fill = false;
		// Container child vbox1.Gtk.Box+BoxChild
		this.hbox1 = new global::Gtk.HBox ();
		this.hbox1.Name = "hbox1";
		this.hbox1.Spacing = 6;
		// Container child hbox1.Gtk.Box+BoxChild
		this.StartVideoButton = new global::Gtk.Button ();
		this.StartVideoButton.CanFocus = true;
		this.StartVideoButton.Name = "StartVideoButton";
		this.StartVideoButton.UseUnderline = true;
		this.StartVideoButton.Label = global::Mono.Unix.Catalog.GetString ("Start Video Capture");
		this.hbox1.Add (this.StartVideoButton);
		global::Gtk.Box.BoxChild w9 = ((global::Gtk.Box.BoxChild)(this.hbox1 [this.StartVideoButton]));
		w9.Position = 0;
		w9.Expand = false;
		w9.Fill = false;
		// Container child hbox1.Gtk.Box+BoxChild
		this.StopVideoButton = new global::Gtk.Button ();
		this.StopVideoButton.CanFocus = true;
		this.StopVideoButton.Name = "StopVideoButton";
		this.StopVideoButton.UseUnderline = true;
		this.StopVideoButton.Label = global::Mono.Unix.Catalog.GetString ("Stop Video Capture");
		this.hbox1.Add (this.StopVideoButton);
		global::Gtk.Box.BoxChild w10 = ((global::Gtk.Box.BoxChild)(this.hbox1 [this.StopVideoButton]));
		w10.Position = 1;
		w10.Expand = false;
		w10.Fill = false;
		// Container child hbox1.Gtk.Box+BoxChild
		this.GPUStart = new global::Gtk.Button ();
		this.GPUStart.CanFocus = true;
		this.GPUStart.Name = "GPUStart";
		this.GPUStart.UseUnderline = true;
		this.GPUStart.Label = global::Mono.Unix.Catalog.GetString ("GPU Start");
		this.hbox1.Add (this.GPUStart);
		global::Gtk.Box.BoxChild w11 = ((global::Gtk.Box.BoxChild)(this.hbox1 [this.GPUStart]));
		w11.Position = 2;
		w11.Expand = false;
		w11.Fill = false;
		// Container child hbox1.Gtk.Box+BoxChild
		this.GPUStop = new global::Gtk.Button ();
		this.GPUStop.CanFocus = true;
		this.GPUStop.Name = "GPUStop";
		this.GPUStop.UseUnderline = true;
		this.GPUStop.Label = global::Mono.Unix.Catalog.GetString ("GPU Stop");
		this.hbox1.Add (this.GPUStop);
		global::Gtk.Box.BoxChild w12 = ((global::Gtk.Box.BoxChild)(this.hbox1 [this.GPUStop]));
		w12.Position = 3;
		w12.Expand = false;
		w12.Fill = false;
		this.vbox1.Add (this.hbox1);
		global::Gtk.Box.BoxChild w13 = ((global::Gtk.Box.BoxChild)(this.vbox1 [this.hbox1]));
		w13.Position = 2;
		w13.Expand = false;
		w13.Fill = false;
		// Container child vbox1.Gtk.Box+BoxChild
		this.hbox7 = new global::Gtk.HBox ();
		this.hbox7.Name = "hbox7";
		this.hbox7.Spacing = 6;
		// Container child hbox7.Gtk.Box+BoxChild
		this.startRecording = new global::Gtk.Button ();
		this.startRecording.CanFocus = true;
		this.startRecording.Name = "startRecording";
		this.startRecording.UseUnderline = true;
		this.startRecording.Label = global::Mono.Unix.Catalog.GetString ("Start Recording");
		this.hbox7.Add (this.startRecording);
		global::Gtk.Box.BoxChild w14 = ((global::Gtk.Box.BoxChild)(this.hbox7 [this.startRecording]));
		w14.Position = 0;
		w14.Expand = false;
		w14.Fill = false;
		// Container child hbox7.Gtk.Box+BoxChild
		this.stopRecording = new global::Gtk.Button ();
		this.stopRecording.CanFocus = true;
		this.stopRecording.Name = "stopRecording";
		this.stopRecording.UseUnderline = true;
		this.stopRecording.Label = global::Mono.Unix.Catalog.GetString ("Stop Recording");
		this.hbox7.Add (this.stopRecording);
		global::Gtk.Box.BoxChild w15 = ((global::Gtk.Box.BoxChild)(this.hbox7 [this.stopRecording]));
		w15.Position = 1;
		w15.Expand = false;
		w15.Fill = false;
		// Container child hbox7.Gtk.Box+BoxChild
		this.StartStreamButton = new global::Gtk.Button ();
		this.StartStreamButton.CanFocus = true;
		this.StartStreamButton.Name = "StartStreamButton";
		this.StartStreamButton.UseUnderline = true;
		this.StartStreamButton.Label = global::Mono.Unix.Catalog.GetString ("Start Image Stream");
		this.hbox7.Add (this.StartStreamButton);
		global::Gtk.Box.BoxChild w16 = ((global::Gtk.Box.BoxChild)(this.hbox7 [this.StartStreamButton]));
		w16.Position = 2;
		w16.Expand = false;
		w16.Fill = false;
		// Container child hbox7.Gtk.Box+BoxChild
		this.StopStreamButton = new global::Gtk.Button ();
		this.StopStreamButton.CanFocus = true;
		this.StopStreamButton.Name = "StopStreamButton";
		this.StopStreamButton.UseUnderline = true;
		this.StopStreamButton.Label = global::Mono.Unix.Catalog.GetString ("Stop Image Stream");
		this.hbox7.Add (this.StopStreamButton);
		global::Gtk.Box.BoxChild w17 = ((global::Gtk.Box.BoxChild)(this.hbox7 [this.StopStreamButton]));
		w17.PackType = ((global::Gtk.PackType)(1));
		w17.Position = 4;
		w17.Expand = false;
		w17.Fill = false;
		this.vbox1.Add (this.hbox7);
		global::Gtk.Box.BoxChild w18 = ((global::Gtk.Box.BoxChild)(this.vbox1 [this.hbox7]));
		w18.Position = 3;
		w18.Expand = false;
		w18.Fill = false;
		// Container child vbox1.Gtk.Box+BoxChild
		this.hbox8 = new global::Gtk.HBox ();
		this.hbox8.Name = "hbox8";
		this.hbox8.Spacing = 6;
		// Container child hbox8.Gtk.Box+BoxChild
		this.InfoButton = new global::Gtk.Button ();
		this.InfoButton.WidthRequest = 59;
		this.InfoButton.CanFocus = true;
		this.InfoButton.Name = "InfoButton";
		this.InfoButton.UseUnderline = true;
		this.InfoButton.Label = global::Mono.Unix.Catalog.GetString ("Info");
		this.hbox8.Add (this.InfoButton);
		global::Gtk.Box.BoxChild w19 = ((global::Gtk.Box.BoxChild)(this.hbox8 [this.InfoButton]));
		w19.Position = 0;
		w19.Expand = false;
		w19.Fill = false;
		// Container child hbox8.Gtk.Box+BoxChild
		this.System = new global::Gtk.Button ();
		this.System.CanFocus = true;
		this.System.Name = "System";
		this.System.UseUnderline = true;
		this.System.Label = global::Mono.Unix.Catalog.GetString ("System Info");
		this.hbox8.Add (this.System);
		global::Gtk.Box.BoxChild w20 = ((global::Gtk.Box.BoxChild)(this.hbox8 [this.System]));
		w20.Position = 1;
		w20.Expand = false;
		w20.Fill = false;
		// Container child hbox8.Gtk.Box+BoxChild
		this.KillButton = new global::Gtk.Button ();
		this.KillButton.TooltipMarkup = "Kills the computer vision process";
		this.KillButton.WidthRequest = 60;
		this.KillButton.CanFocus = true;
		this.KillButton.Name = "KillButton";
		this.KillButton.UseUnderline = true;
		this.KillButton.Label = global::Mono.Unix.Catalog.GetString ("Kill");
		this.hbox8.Add (this.KillButton);
		global::Gtk.Box.BoxChild w21 = ((global::Gtk.Box.BoxChild)(this.hbox8 [this.KillButton]));
		w21.PackType = ((global::Gtk.PackType)(1));
		w21.Position = 3;
		w21.Expand = false;
		w21.Fill = false;
		this.vbox1.Add (this.hbox8);
		global::Gtk.Box.BoxChild w22 = ((global::Gtk.Box.BoxChild)(this.vbox1 [this.hbox8]));
		w22.Position = 4;
		w22.Expand = false;
		w22.Fill = false;
		this.hbox4.Add (this.vbox1);
		global::Gtk.Box.BoxChild w23 = ((global::Gtk.Box.BoxChild)(this.hbox4 [this.vbox1]));
		w23.Position = 0;
		w23.Expand = false;
		w23.Fill = false;
		// Container child hbox4.Gtk.Box+BoxChild
		this.mgrStatsComm = new global::VisionSystemGtkViewer.ManagerStats ();
		this.mgrStatsComm.Events = ((global::Gdk.EventMask)(256));
		this.mgrStatsComm.Name = "mgrStatsComm";
		this.hbox4.Add (this.mgrStatsComm);
		global::Gtk.Box.BoxChild w24 = ((global::Gtk.Box.BoxChild)(this.hbox4 [this.mgrStatsComm]));
		w24.Position = 1;
		w24.Expand = false;
		this.VertBox.Add (this.hbox4);
		global::Gtk.Box.BoxChild w25 = ((global::Gtk.Box.BoxChild)(this.VertBox [this.hbox4]));
		w25.Position = 0;
		w25.Expand = false;
		w25.Fill = false;
		// Container child VertBox.Gtk.Box+BoxChild
		this.hbox11 = new global::Gtk.HBox ();
		this.hbox11.Name = "hbox11";
		this.hbox11.Spacing = 6;
		// Container child hbox11.Gtk.Box+BoxChild
		this.vbox7 = new global::Gtk.VBox ();
		this.vbox7.Name = "vbox7";
		this.vbox7.Spacing = 6;
		// Container child vbox7.Gtk.Box+BoxChild
		this.textOutputWindow = new global::Gtk.TextView ();
		this.textOutputWindow.HeightRequest = 20;
		this.textOutputWindow.CanDefault = true;
		this.textOutputWindow.CanFocus = true;
		this.textOutputWindow.Name = "textOutputWindow";
		this.textOutputWindow.Editable = false;
		this.textOutputWindow.CursorVisible = false;
		this.textOutputWindow.AcceptsTab = false;
		this.vbox7.Add (this.textOutputWindow);
		global::Gtk.Box.BoxChild w26 = ((global::Gtk.Box.BoxChild)(this.vbox7 [this.textOutputWindow]));
		w26.Position = 0;
		// Container child vbox7.Gtk.Box+BoxChild
		this.textVisionOutput = new global::Gtk.TextView ();
		this.textVisionOutput.HeightRequest = 20;
		this.textVisionOutput.CanDefault = true;
		this.textVisionOutput.CanFocus = true;
		this.textVisionOutput.Name = "textVisionOutput";
		this.textVisionOutput.Editable = false;
		this.textVisionOutput.CursorVisible = false;
		this.textVisionOutput.AcceptsTab = false;
		this.vbox7.Add (this.textVisionOutput);
		global::Gtk.Box.BoxChild w27 = ((global::Gtk.Box.BoxChild)(this.vbox7 [this.textVisionOutput]));
		w27.Position = 1;
		this.hbox11.Add (this.vbox7);
		global::Gtk.Box.BoxChild w28 = ((global::Gtk.Box.BoxChild)(this.hbox11 [this.vbox7]));
		w28.Position = 0;
		// Container child hbox11.Gtk.Box+BoxChild
		this.mgrStatsImageCapture = new global::VisionSystemGtkViewer.ManagerStats ();
		this.mgrStatsImageCapture.Events = ((global::Gdk.EventMask)(256));
		this.mgrStatsImageCapture.Name = "mgrStatsImageCapture";
		this.hbox11.Add (this.mgrStatsImageCapture);
		global::Gtk.Box.BoxChild w29 = ((global::Gtk.Box.BoxChild)(this.hbox11 [this.mgrStatsImageCapture]));
		w29.Position = 1;
		w29.Expand = false;
		this.VertBox.Add (this.hbox11);
		global::Gtk.Box.BoxChild w30 = ((global::Gtk.Box.BoxChild)(this.VertBox [this.hbox11]));
		w30.Position = 1;
		w30.Expand = false;
		w30.Fill = false;
		// Container child VertBox.Gtk.Box+BoxChild
		this.hbox21 = new global::Gtk.HBox ();
		this.hbox21.Name = "hbox21";
		this.hbox21.Spacing = 6;
		// Container child hbox21.Gtk.Box+BoxChild
		this.scrolledwindow1 = new global::Gtk.ScrolledWindow ();
		this.scrolledwindow1.CanFocus = true;
		this.scrolledwindow1.Name = "scrolledwindow1";
		this.scrolledwindow1.ShadowType = ((global::Gtk.ShadowType)(1));
		// Container child scrolledwindow1.Gtk.Container+ContainerChild
		global::Gtk.Viewport w31 = new global::Gtk.Viewport ();
		w31.ShadowType = ((global::Gtk.ShadowType)(0));
		// Container child GtkViewport1.Gtk.Container+ContainerChild
		this.image1 = new global::Gtk.Image ();
		this.image1.Name = "image1";
		w31.Add (this.image1);
		this.scrolledwindow1.Add (w31);
		this.hbox21.Add (this.scrolledwindow1);
		global::Gtk.Box.BoxChild w34 = ((global::Gtk.Box.BoxChild)(this.hbox21 [this.scrolledwindow1]));
		w34.Position = 0;
		// Container child hbox21.Gtk.Box+BoxChild
		this.vbox6 = new global::Gtk.VBox ();
		this.vbox6.Name = "vbox6";
		this.vbox6.Spacing = 6;
		// Container child vbox6.Gtk.Box+BoxChild
		this.mgrStatsImageProc = new global::VisionSystemGtkViewer.ManagerStats ();
		this.mgrStatsImageProc.Events = ((global::Gdk.EventMask)(256));
		this.mgrStatsImageProc.Name = "mgrStatsImageProc";
		this.vbox6.Add (this.mgrStatsImageProc);
		global::Gtk.Box.BoxChild w35 = ((global::Gtk.Box.BoxChild)(this.vbox6 [this.mgrStatsImageProc]));
		w35.Position = 0;
		w35.Expand = false;
		w35.Fill = false;
		this.hbox21.Add (this.vbox6);
		global::Gtk.Box.BoxChild w36 = ((global::Gtk.Box.BoxChild)(this.hbox21 [this.vbox6]));
		w36.Position = 1;
		w36.Expand = false;
		w36.Fill = false;
		this.VertBox.Add (this.hbox21);
		global::Gtk.Box.BoxChild w37 = ((global::Gtk.Box.BoxChild)(this.VertBox [this.hbox21]));
		w37.Position = 2;
		// Container child VertBox.Gtk.Box+BoxChild
		this.hbox22 = new global::Gtk.HBox ();
		this.hbox22.Name = "hbox22";
		this.hbox22.Spacing = 6;
		// Container child hbox22.Gtk.Box+BoxChild
		this.scrolledwindow2 = new global::Gtk.ScrolledWindow ();
		this.scrolledwindow2.CanFocus = true;
		this.scrolledwindow2.Name = "scrolledwindow2";
		this.scrolledwindow2.ShadowType = ((global::Gtk.ShadowType)(1));
		// Container child scrolledwindow2.Gtk.Container+ContainerChild
		global::Gtk.Viewport w38 = new global::Gtk.Viewport ();
		w38.ShadowType = ((global::Gtk.ShadowType)(0));
		// Container child GtkViewport2.Gtk.Container+ContainerChild
		this.image2 = new global::Gtk.Image ();
		this.image2.Name = "image2";
		w38.Add (this.image2);
		this.scrolledwindow2.Add (w38);
		this.hbox22.Add (this.scrolledwindow2);
		global::Gtk.Box.BoxChild w41 = ((global::Gtk.Box.BoxChild)(this.hbox22 [this.scrolledwindow2]));
		w41.Position = 0;
		// Container child hbox22.Gtk.Box+BoxChild
		this.vbox8 = new global::Gtk.VBox ();
		this.vbox8.Name = "vbox8";
		this.vbox8.Spacing = 6;
		// Container child vbox8.Gtk.Box+BoxChild
		this.mgrStatsSteamRecord = new global::VisionSystemGtkViewer.ManagerStats ();
		this.mgrStatsSteamRecord.Events = ((global::Gdk.EventMask)(256));
		this.mgrStatsSteamRecord.Name = "mgrStatsSteamRecord";
		this.vbox8.Add (this.mgrStatsSteamRecord);
		global::Gtk.Box.BoxChild w42 = ((global::Gtk.Box.BoxChild)(this.vbox8 [this.mgrStatsSteamRecord]));
		w42.Position = 0;
		w42.Expand = false;
		w42.Fill = false;
		this.hbox22.Add (this.vbox8);
		global::Gtk.Box.BoxChild w43 = ((global::Gtk.Box.BoxChild)(this.hbox22 [this.vbox8]));
		w43.Position = 1;
		w43.Expand = false;
		w43.Fill = false;
		this.VertBox.Add (this.hbox22);
		global::Gtk.Box.BoxChild w44 = ((global::Gtk.Box.BoxChild)(this.VertBox [this.hbox22]));
		w44.Position = 3;
		this.Add (this.VertBox);
		if ((this.Child != null)) {
			this.Child.ShowAll ();
		}
		this.DefaultWidth = 1025;
		this.DefaultHeight = 731;
		this.textOutputWindow.HasDefault = true;
		this.textVisionOutput.HasDefault = true;
		this.Show ();
		this.DeleteEvent += new global::Gtk.DeleteEventHandler (this.OnDeleteEvent);
		this.btnZMQConnect.Clicked += new global::System.EventHandler (this.OnBtnZMQConnectClicked);
		this.StartVideoButton.Clicked += new global::System.EventHandler (this.OnStartVideoButtonClicked);
		this.StopVideoButton.Clicked += new global::System.EventHandler (this.OnStopVideoButtonClicked);
		this.GPUStart.Clicked += new global::System.EventHandler (this.OnGPUStartClicked);
		this.GPUStop.Clicked += new global::System.EventHandler (this.OnGPUStopClicked);
		this.startRecording.Clicked += new global::System.EventHandler (this.OnStartRecordingClicked);
		this.stopRecording.Clicked += new global::System.EventHandler (this.OnStopRecordingClicked);
		this.StartStreamButton.Clicked += new global::System.EventHandler (this.OnStartStreamButtonClicked);
		this.StopStreamButton.Clicked += new global::System.EventHandler (this.OnStopStreamButtonClicked);
		this.InfoButton.Clicked += new global::System.EventHandler (this.OnInfoButtonClicked);
		this.System.Clicked += new global::System.EventHandler (this.OnSystemClicked);
		this.KillButton.Clicked += new global::System.EventHandler (this.OnKillButtonClicked);
	}
}
