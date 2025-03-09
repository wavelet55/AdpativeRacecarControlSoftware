/*
 * 2253 Device Interface header
 * Copyright (c) 2010, Sensoray Co., Inc.
 */

#ifndef _S2253_HEADER
#define _S2253_HEADER

#ifndef __WIN32
#include "asm/types.h"
#else
typedef UINT8 __u8;
typedef UINT64 __u64;
typedef UINT32 __u32;
#endif

/* All fields are little-endian unless otherwise specified */

struct s2253_osd {
    int osdOn; //OSD on if != 0, 1=8x14 font, 2=16x16 font
    int osdChan; // OSD channel (0=capture, 1=preview, 2=display, 3=capture&preview)
    int osdBmp; // reserved, set to 0
    int transparent; //transparent OSD if !=0, 1=100%, 2=50%, 3=25%
    int positionTop; //OSD string on top of the screen if !=0 
    int ddmm; //date format 0=mm-dd 1=dd-mm 2=mmm-dd, 3=dd-mmm- 4=mmm dd, 5=ddmmm
    int year2; //year value in date is truncated to 2 digits if != 0
    int fraction; //number of digits showing fraction of a second: 0, 1, 2, 3
    int xOffset, yOffset; // x offset from left side, y offset from top or bottom
    char line[160]; //caption text
};

#ifdef __WIN32
#pragma pack(1)
#endif
struct s2253_payload_header {
    __u8 length;  /* Length of this header (not including payload) */
    /* Frame id; end-of-frame; PTS present; SCR present; reserved; still image; error; end-of-header 
     fid: toggles for each frame.
     eof: signifies that this packet is the last in the current frame.
          For output, this signifies that it is ok to start decoding up to this packet,
                      otherwise a certain amount is buffered before decoding.
     res: indicates 0=video packet  1=audio packet
     sti: indicates that the image is transferred by fields: even lines are sent first, then odd lines. lines are 0 to 479 (575 for pal)
     err: not used, except for lastpacket
     eoh: not used, except for lastpacket
    */
#ifndef __WIN32
#if defined(__LITTLE_ENDIAN) || defined(__LITTLE_ENDIAN__) || defined(__ARMEL__)
    unsigned fid:1, eof:1, pts:1, scr:1, res:1, sti:1, err:1, eoh:1;
#elif defined(__BIG_ENDIAN) || defined(__BIG_ENDIAN__)
    unsigned eoh:1, err:1, sti:1, res:1, scr:1, pts:1, eof:1, fid:1;
#else
#error Unable to determine cpu endianness
#endif
#else
	// must define as char in WINDDK or structure size will be too large
	// even with correct packing
    unsigned char fid:1, eof:1, pts:1, scr:1, res:1, sti:1, err:1, eoh:1;
#endif
    __u8 PTS[4];  /* Presentation timestamp, when the frame was captured */
    __u8 SCR[6];  /* System clock reference, when the frame was sent over USB */
    __u64 seq;/* sequence count, little-endian */
#ifndef __WIN32  // array of 0 not allowed by Windows driver compiler
    __u8 payload[0];
#endif

}
#ifdef __WIN32
;
#pragma pack()
#else
__attribute__ ((packed));
#endif

#define S2253_LASTPACKET_INITIALIZER   {       \
    .length = sizeof(struct s2253_payload_header),      \
    .eof = 1,   \
    .pts = 0,   \
    .scr = 0,   \
    .err = 1,   \
    .eoh = 1,   \
}

struct s2253_clock {
    __u64 sec;
    __u32 usec;
};

struct s2253_first_scr {
    unsigned long long record; /* first audio scr */
    unsigned long long preview; /* first preview scr */
    unsigned long long capture; /* first capture scr */
};

struct s2253_mfg_info {
    __u32 model_no;
    __u32 reserved1;
    __u32 features;
    __u32 serial_no;
};

#define S2253_FEATURE_MPEG2 1

struct s2253_overlay_image {
    unsigned char id; // unique id representing this overlay (0-255 on display, 0-15 on others)
    unsigned char transparent; // 0=transparent .. 7=opaque  8=use PNG or BMP transparency info(display only)
    unsigned char update; // bit 0: update the display, bit 1: update the x&y offset, bit 2: update the transparency (see defines below)
    unsigned char channel; // channel 0=display 1=stream1(capture) 2=stream2(preview) 3=resizer(capture & preview)
    int xOffset, yOffset; // x offset from left side, y offset from top
    unsigned int length; // length of image data (or 0 to use existing data, and update xOffset and yOffset only)
#ifndef __WIN32  // array of 0 not allowed by Windows driver compiler
    unsigned char data[0]; // data must contain headers that describe image dimensions
#endif
};

#define OVERLAY_UPDATE_DISPLAY 1
#define OVERLAY_UPDATE_POSITION 2
#define OVERLAY_UPDATE_TRANSPARENT 4

#define OVERLAY_CHANNEL_DISPLAY 0
#define OVERLAY_CHANNEL_CAPTURE 1
#define OVERLAY_CHANNEL_PREVIEW 2
#define OVERLAY_CHANNEL_RESIZER 3
/* overlay channels 1,2,3 will only accept RLE image data - nothing else! */

struct s2253_line {
    int x1;
    int y1;
    int x2;
    int y2;
    int width;
    int argb;
};

#define OVERLAY_FORMAT_LINE 0x656e696c
#define OVERLAY_FORMAT_TEXT 0x74786574
#define OVERLAY_FORMAT_RLE  0x20454c52

/* if possible, keep this structure backward compatible with
 * older firmware by adding new entries at the end.
 */ 
enum s2253_parameter {
    S2253_FIRMWARE_VERSION,     /* R: version # */
/* 1-3 */
    S2253_GPIO_INPUT_ENABLE,    /* R/W: 0=disable, 1=enable */
    S2253_GPIO_INPUT_VALUE,     /* R: value */
    S2253_GPIO_OUTPUT_VALUE,    /* R/W: value 0=low 1=high 2=VSYNC */
/* 4 */
    S2253_CLOCK,                /* R/W: for setting the clock, used for on-screen timestamps */
/* 5 */
    S2253_INPUT_VIDEO_STANDARD, /* R/W: use enum s2253_standard below */
/* 6-13 */
    /* note: CAPTURE params must be grouped */
    /* CAPTURE is stream A. */
    S2253_CAPTURE_STATE,        /* R/W: 0=idle, 1=streaming, others=error */
    S2253_CAPTURE_WIDTH,        /* R/W: 360,640,720, etc */
    S2253_CAPTURE_HEIGHT,       /* R/W: 240,480,576, etc */
    S2253_CAPTURE_FORMAT,       /* R/W: MPEG4,H264,JPEG, see format enum */
    S2253_CAPTURE_BITRATE,      /* R/W: in kbps (changes happen immediately) */
    S2253_CAPTURE_BITRATE_MODE, /* R/W: use enum s2253_bitrate_mode below */
    S2253_CAPTURE_JPEG_QUALITY, /* R/W: 0-100 */
    S2253_CAPTURE_DECIMATION,   /* R/W: N=1+ (encode 1 frame every N frames) */
/* 14-21 */
    /* note: PREVIEW params must be grouped, and in the same order as CAPTURE */
    /* PREVIEW is stream B. */
    S2253_PREVIEW_STATE,        /* R/W: 0=idle, 1=streaming, others=error */
    S2253_PREVIEW_WIDTH,        /* R/W: 360,640,720, etc */
    S2253_PREVIEW_HEIGHT,       /* R/W: 240,480,576, etc */
    S2253_PREVIEW_FORMAT,       /* R/W: MPEG4,H264,JPEG, see format enum */
    S2253_PREVIEW_BITRATE,      /* R/W: in kbps (changes happen immediately) */
    S2253_PREVIEW_BITRATE_MODE, /* R/W: use enum s2253_bitrate_mode below */
    S2253_PREVIEW_JPEG_QUALITY, /* R/W: 0-100 */
    S2253_PREVIEW_DECIMATION,   /* R/W: N=1+ (encode 1 frame every N frames) */
/* 22-26 */
    /* RECORD_* is the audio stream, aside from the stream audio parameters */
    S2253_RECORD_STATE,         /* R/W: audio state: 0=idle, 1=streaming, others=error */
    S2253_RECORD_FORMAT,        /* R/W: PCM, g.711 or AAC */
    S2253_RECORD_BITRATE,       /* R/W: AAC audio bitrate in kbps (overridden by S2253_CAPTURE_ABITRATE or S2253_PREVIEW_ABITRATE, if active) */
    S2253_RECORD_SAMPLERATE,    /* R/W: samples per sec, 8000 or 48000 */
    S2253_RECORD_INPUT,         /* R/W: 0=mic 1=line in */
    // S2253_RECORD_ADTS : defined later (AAC only)
/* 27-29 */
    /* PLAYBACK is the output audio stream, NotYetImplemented */
    S2253_PLAYBACK_STATE,       /* R/W: 0=idle 1=streaming others=error */
    S2253_PLAYBACK_FORMAT,      /* R/W: g.711 or AAC */
    S2253_PLAYBACK_SAMPLERATE,  /* R/W: samples per sec */
/* 30-32 */
    S2253_DEBUG_MESSAGE,        /* R: debug message string, from ERR or DBG macro */
    S2253_LOG_MESSAGE,          /* R: log message string, automatically moves to next line when read */
    S2253_KLOG_MESSAGE,         /* R: kernel message string, automatically moves to next line when read */
/* 33-38 */
    S2253_SET_OSD,              /* see s2253_osd above*/
    S2253_BRIGHTNESS,           /* R/W: 0..255 */
    S2253_CONTRAST,             /* R/W: 0..255 */
    S2253_SATURATION,           /* R/W: 0..255 */
    S2253_HUE,                  /* R/W: -128..127 */
    S2253_INTERPOLATE,          /* R/W: 0=off 1=interpolate */
/* 39-44 */
    S2253_ECHO_TIMESTAMP,       /* W: prints the elapsed time since the given timestamp */
    S2253_FIRST_SCR,            /* R: first SCRs see s2253_first_scr */
    S2253_TIMESTAMP,            /* R/W: the 90khz counter used for generating timestamps */
    S2253_RECORD_ADTS,          /* R/W: turns on/off AAC ADTS headers. default off */
    S2253_RECORD_CHANNELS,      /* R/W: number of audio channels: 1=mono(mixed) 2=stereo, default stereo */
    S2253_MFG_INFO,             /* R: 16 bytes Manufacturing information, including model no., serial no., etc. */
/* 45-52 */
    /* keep the spacing even between these CAPTURE/PREVIEW groups as the above groups */
    S2253_CAPTURE_CROP_LEFT,    /* R/W: OBSOLETE */
    S2253_CAPTURE_CROP_RIGHT,   /* R/W: OBSOLETE */
    S2253_CAPTURE_FLIP,         /* R/W: flip vertically: bit 0, flip horizontally: bit 1, flip diagonally: bit 2 */
    S2253_CAPTURE_ABITRATE,     /* R/W: AAC audio bitrate in kbps */
    S2253_CAPTURE_ADTS,         /* R/W: turns on/off AAC ADTS headers. default off */
    S2253_CAPTURE_GOPSIZE,      /* R/W: GOP size. (use 0 for encoder default) */
    S2253_CAPTURE_IDRFRAMES,    /* R/W: IDR frame interval. H264 only. 0=IDR first frame only, otherwise IDR every Nth GOP */
    S2253_CAPTURE_ACHANNELS,    /* R/W: number of audio channels: 1=mono(mixed) 2=stereo 3=mono(L only) 4=mono(R only), default stereo */
/* 53-60 */
    S2253_PREVIEW_CROP_LEFT,    /* R/W: OBSOLETE */
    S2253_PREVIEW_CROP_RIGHT,   /* R/W: OBSOLETE */
    S2253_PREVIEW_FLIP,         /* R/W: flip vertically: bit 0, flip horizontally: bit 1, flip diagonally: bit 2 */
    S2253_PREVIEW_ABITRATE,     /* R/W: AAC audio bitrate in kbps (overridden by S2253_CAPTURE_ABITRATE, if active) */
    S2253_PREVIEW_ADTS,         /* R/W: turns on/off AAC ADTS headers. default off */
    S2253_PREVIEW_GOPSIZE,      /* R/W: GOP size. (use 0 for encoder default) */
    S2253_PREVIEW_IDRFRAMES,    /* R/W: IDR frame interval. H264 only. 0=IDR first frame only, otherwise IDR every Nth GOP */
    S2253_PREVIEW_ACHANNELS,    /* R/W: number of audio channels: 1=mono(mixed) 2=stereo 3=mono(L only) 4=mono(R only), default stereo */
/* 61-65 */
    S2253_RECORD_VOLUME,        /* R/W: volume control: left channel: lower 8 bits, right channel: upper 8 bits, see S2253_RECORD_VOLUME_MIN/MAX 
                                    NOTE: volume control has no effect when AGC is on. */
    S2253_RECORD_AGC,           /* R/W: automatic gain control: left channel: bit 0, right channel: bit 1, off=0, on=1, default=on */
                                     /*    also muting control: left channel: bit 2, right channel: bit 3, unmuted=0, muted=1, default=0 */
    S2253_PLAYBACK_CHANNELS,    /* R/W: number of audio channels: 1=mono 2=stereo */
    S2253_PLAYBACK_VOLUME,      /* R/W: volume control: left channel: lower 8 bits, right channel: upper 8 bits, see S2253_RECORD_VOLUME_MIN/MAX */
    S2253_SYSTEM_REBOOT,        /* W: set to anything to reboot */
/* 66-70 */
    S2253_OUTPUT_MODE,          /* R/W: select from: stream, capture, colorbars  */
    S2253_OUTPUT_VIDEO_STANDARD, /* R/W: use enum s2253_standard below */
    S2253_OUTPUT_STATE,         /* R/W: 0=idle, 1=streaming, others=error */
    S2253_OUTPUT_FORMAT,        /* R/W: only: JPEG, H264, MPEG4 or MP4 or'd with audio format 
                                        A/V sync will occur only if audio format is present.
                                        MP4 will use video and audio stream format from headers with S2253_FORMAT_MP4|S2253_FORMAT_NOVIDEO 
                                        (Autodetect format will ocurr if stream headers are sent as data)
                                        RAW playback use: S2253_FORMAT_UYVY, S2253_FORMAT_YUV420PSEMI, or S2253_FORMAT_Y8
                                        and set parameters S2253_OUTPUT_WIDTH and S2253_OUTPUT_HEIGHT.
                                        */
    S2253_OUTPUT_DECIMATION,    /* R/W: undecimation, rather. each decoded frame is output N times.  NYI */

/* 71-78 */
    /* keep the spacing even between these CAPTURE/PREVIEW groups as the above groups */
    S2253_CAPTURE_USER_DATA,    /* R/W: user data to be inserted in the stream - MPEG2 MPEG4 and H264 only, 1024 bytes max */
    S2253_CAPTURE_USER_INTERVAL,/* R/W: 0=once per USER_DATA update, otherwise every N frames */
    S2253_CAPTURE_ASPECT_RATIO, /* R/W: display aspect ratio: width: bits 15-8, height: bits 7-0 ... 0x0000=square pixels 0x0403=4:3 0x1009=16:9*/
    S2253_CAPTURE_IFRAME_HDR,   /* R/W: 0=MPEG4 VOL or MPEG2 seq header at start only, 1=header at each Iframe */
    S2253_CAPTURE_H264_PROFILE, /* R/W: 66=baseline 77=main 100=high(default)*/
    S2253_CAPTURE_H264_LEVEL,   /* R/W: 10=1.0 9=1B 11=1.1 12=1.2 13=1.3 20=2.0 21=2.1 22=2.2 30=3.0 31=3.1 32=3.2 40=4.0(default) 41=4.1 42=4.2 50=5.0 */
    S2253_CAPTURE_AUDIO_DELAY,  /* R/W: Audio delay adjustment in ms (signed 16-bit value) */
    S2253_CAPTURE_CC_ENABLE,    /* R/W: 1=closed-caption in H.264 data (cannot be used with userdata) 2=serial port data */
/* 79-86 */
    S2253_PREVIEW_USER_DATA,    /* R/W: user data to be inserted in the stream - MPEG2 MPEG4 and H264 only, 1024 bytes max */
    S2253_PREVIEW_USER_INTERVAL,/* R/W: 0=once per USER_DATA update, otherwise every N frames */
    S2253_PREVIEW_ASPECT_RATIO, /* R/W: display aspect ratio: width: bits 15-8, height: bits 7-0 ... 0x0000=square pixels 0x0403=4:3 0x1009=16:9 */
    S2253_PREVIEW_IFRAME_HDR,   /* R/W: 0=MPEG4 VOL or MPEG2 seq header at start only, 1=header at each Iframe */
    S2253_PREVIEW_H264_PROFILE, /* R/W: 66=baseline 77=main 100=high(default) */
    S2253_PREVIEW_H264_LEVEL,   /* R/W: 10=1.0 9=1B 11=1.1 12=1.2 13=1.3 20=2.0 21=2.1 22=2.2 30=3.0 31=3.1 32=3.2 40=4.0(default) 41=4.1 42=4.2 50=5.0 */
    S2253_PREVIEW_AUDIO_DELAY,  /* R/W: Audio delay adjustment in ms (signed 16-bit value) */
    S2253_PREVIEW_CC_ENABLE,    /* R/W: 1=closed-caption in H.264 data (cannot be used with userdata) 2=serial port data*/
/* 87-89 */
    S2253_OUTPUT_WIDTH,         /* R/W: output width in pixels, only for raw formats */
    S2253_OUTPUT_HEIGHT,        /* R/W: output height in pixels, only for raw formats */
    S2253_OUTPUT_LATENCY_MODE,  /* R/W: latency reduction mode  0=off (file mode)  1=on (drop audio and video based on latency) */
/* 90-98 */
    S2253_INPUT_HVLOCK,         /* R: 0=no lock, 1=locked */
    S2253_INPUT_SEL,            /* R/W: 0=composite 1=svideo */
    S2253_XLED_MODE,            /* R/W: 0=blink when no video lock, 1=off when no video lock, 2=always on, 3=always off */
    S2253_FRAME_COUNTER,        /* R/W: global frame counter, wraps at 10k  (4-byte little endian value) */
    S2253_TV_VCR_MODE,          /* R/W: 0=automatic, 1=reserved, 2=VCR(nonstandard video) 3=TV(standard video) */
    S2253_CAPTURE_TIMEOUTS,     /* R/W: count of capture timeouts  (4-byte little endian value) */
    S2253_INPUT_H_ADJUST,       /* R/W: video decoder horizontal adjustment, signed 16-bit value -256..255 */
    S2253_INPUT_V_ADJUST,       /* R/W: video decoder vertical adjustment, signed 16-bit value -128..127 */

    S2253_INPUT_FIELD_MODE,     /* R/W: video decoder interlaced field reconstruction
                                        NTSC: 0=bottom-field-first 1=top-field-first
                                        PAL:  0=top-field-first 1=bottom-field-first */
/* 99-128 */
    S2253TB_ENCODER_RESET,	/* W: bitmask of channels to reset to zero */
    S2253TB_ENCODER_READ,	/* R/W: W: bitmask of channels to read, read returns the value, 0xffffffff=not ready yet  (4-byte read chan0 only, 8-byte read both channels) */
    S2253TB_GPS_DATA,		/* R/W: gps data, n bytes */
    S2253TB_GPS_READSTATUS,	/* R/W: u8 gps status, W:send cmd, R:read result, 0xff=not ready yet */
    S2253TB_GPIO_CONFIG,	/* W: upper 8 bits: mask of gpios to config, lower 8 bits: mask of direction 0=in 1=out */
    S2253TB_GPIO_WRITE,		/* W: upper 8 bits: mask of gpios to write, lower 8 bits: mask of values 0=low 1=high  */
    S2253TB_GPIO_READ,		/* R/W: W: lower 8 bits: mask of gpios to read, R: upper 8 bits: mask 0=ready 1=not ready, lower 8 bits: mask of values 0=low 1=high */
    S2253TB_ECHO,		/* W: echo cmd */
    S2253TB_GPS_ENABLE,		/* R/W: W: gps enable 0..1, read 0=not enabled, 1=enabled, 0xff=not ready*/
    S2253TB_VERSION_READ,	/* R/W: W: send cmd, R: read 32-bits result, 0xffffffff=not ready yet */
    S2253TB_COMSTAT_READ,	/* R/W: W: send cmd, R: read 12-bytes result, 12x 0xff=not ready yet */
    S2253TB_XIO_ENABLE,		/* W: upper byte: mask of xio to change, bits 3..0: A0 B0 A1 B1  0=disable 1=enabled */
    S2253TB_XIO_READ,           /* R/W: W: send cmd, R: bits 3..0: A0 B0 A1 B1 0=low 1=high or 0xff if not ready */
    S2253TB_XIO_PAUSE,		/* R/W: bits7-6: stream 0=capture 1=preview 2=output
					bits5-4: XIO port 0..3
					bits3-0: mode 0=disabled 1=rising edge trig, 2=falling edge, 3=level high, 4=level low */
    S2253TB_GPS_LATITUDE,	/* R: 10 chars   ddmm.mmmmp p=N/S  */
    S2253TB_GPS_LONGITUDE,	/* R: 11 chars  dddmm.mmmme e=E/W  */
    S2253TB_GPS_SPEED,		/* R:  4 chars  d.dd  knots  */
    S2253TB_GPS_COURSE,		/* R:  6 chars  ddddd.dd   degrees  */
    S2253TB_GPS_UTC_TIME,	/* R: 10 chars   hhmmss.sss */
    S2253TB_GPS_UTC_DATE,	/* R:  6 chars   ddmmyy */
    S2253TB_GPS_SATELLITES,     /* R: 0-12 satellites */
    S2253TB_GPS_LOCK,		/* R: 0=no lock 1=locked */
    S2253TB_GPS_GGA,		/* R: last GGA message */
    S2253TB_GPS_GSA,		/* R: last GSA message */
    S2253TB_GPS_GSV,		/* R: last GSV message */
    S2253TB_GPS_RMC,		/* R: last RMC message */
    S2253TB_ENC_ASYNCEN,	/* W: upper byte: bitmask of channels to set, lower byte: bitmask 0=disabled 1=enabled */
    S2253TB_ONLINE,		/* R/W: W: sends online cmd, R: returns 1 if TB responded to online cmd, otherwise 0 */
    S2253TB_ENCODER_LOAD,	/* W: 5 bytes: 1 byte chan id, 4 bytes encoder count */
    S2253TB_SUSPEND,		/* W: 0: keep TB on during USB suspend (default), 1: hold TB in reset on suspend */

/* 129-136 */
    /* keep the spacing even between these CAPTURE/PREVIEW groups as the above groups */
    S2253_CAPTURE_INTRAFRAMEQP, /* R/W: Fixed QP value 0-51 for intra frames, use -1 for auto rate control */
    S2253_CAPTURE_INTERPFRAMEQP,/* R/W: Fixed QP value 0-51 for inter p-frames, use -1 for auto rate control */
    S2253_CAPTURE_TRIGGER,      /* R/W: GPI trigger mode 0=disabled 1=rising edge trig, 2=falling edge, 3=level high, 4=level low */
    S2253_CAPTURE_RESERVED4,    /* R/W: */
    S2253_CAPTURE_RESERVED5,    /* R/W: */
    S2253_CAPTURE_RESERVED6,    /* R/W: */
    S2253_CAPTURE_RESERVED7,    /* R/W: */
    S2253_CAPTURE_RESERVED8,    /* R/W: */
/* 137-144 */
    S2253_PREVIEW_INTRAFRAMEQP, /* R/W: Fixed QP value 0-51 for intra frames, use -1 for auto rate control */
    S2253_PREVIEW_INTERPFRAMEQP,/* R/W: Fixed QP value 0-51 for inter p-frames, use -1 for auto rate control */
    S2253_PREVIEW_TRIGGER,      /* R/W: GPI trigger mode 0=disabled 1=rising edge trig, 2=falling edge, 3=level high, 4=level low */
    S2253_PREVIEW_RESERVED4,    /* R/W:  */
    S2253_PREVIEW_RESERVED5,    /* R/W:  */
    S2253_PREVIEW_RESERVED6,    /* R/W:  */
    S2253_PREVIEW_RESERVED7,    /* R/W:  */
    S2253_PREVIEW_RESERVED8,    /* R/W:  */
/* 145-148 */
    S2253_INPUT_CROP_LEFT,	/* R/W: default: 8 */
    S2253_INPUT_CROP_TOP,	/* R/W: default: 0 */
    S2253_INPUT_CROP_WIDTH,	/* R/W: default: 704 */
    S2253_INPUT_CROP_HEIGHT,	/* R/W: default: 576 (will be limited by vid std) */
    S2253_EXT_CLKDIV,		/* R/W: resizer clk divider for experimentation */
    S2253_OUTPUT_TRIGGER,       /* R/W: GPI trigger mode 0=disabled 1=rising edge trig, 2=falling edge, 3=level high, 4=level low */
    S2253_INPUT_FREEZE,		/* R/W: freeze frame, (zoom/mirror/overlays can still be changed) affects both streams */
    S2253_FREEZE_TRIGGER,	/* R/W: GPI trigger mode 0=disabled 1=rising edge trig, 2=falling edge, 3=level high, 4=level low */

    S2253_READ_EEPROM,		/* R: read eeprom */
    S2253_BOOTLOADER_VERSION,	/* R: bootloader version, likely 0 on older boards, 0x130 on rev 1.30 etc */
    S2253_READ_FILE,            /* W: set filename, R: read from it */
    S2253_OUTPUT_VGA,		/* R/W: enable/disable 640x480 output mode */
    S2253_USB_RESET,            /* W: write 1 to reset usb, and reenumerate device without full CPU reset */

    S2253TB_GPS_ALTITUDE,	/* R: 10 chars  nnn.nM  in meters */
    S2253_BLACK_LEVEL,		/* R/W: YCbCr output code range
					0 = ITU-R BT.601 coding range (Y ranges from 16 to 235. U and V range from 16 to 240)
					1 = Extended coding range (Y, U, and V range from 1 to 254) (default) */
    S2253_CPU_LOAD,		/* R: read current CPU load */
    S2253TB_ENCODER_SCALE_FACTOR,/* R/W: 4 or 8 bytes, float32 for each encoder */
    S2253TB_ENCODER_READ_SCALED,/* R/W: 4 or 8 bytes, float32 for each encoder */
    S2253_VIDREG_WRITE,		/* W: bits 15:8 register address 7:0 data */

    S2253_LAST_PARAM,           /* terminating entry */
};

enum s2253_standard {
    S2253_STANDARD_NONE,
    S2253_STANDARD_NTSC,
    S2253_STANDARD_PAL,
};

enum s2253_state {
    S2253_STATE_IDLE,           /* streaming stopped */
    S2253_STATE_STREAMING,      /* streaming in progress */
    S2253_ERROR_NO_INPUT,       /* no input detected */
    S2253_ERROR_INVALID_SIZE,   /* invalid size */
    S2253_ERROR_INVALID_FORMAT, /* invalid format */
    S2253_ERROR_INVALID_FRAMERATE,/* invalid framerate */
    S2253_ERROR_INVALID_BITRATE,/* invalid bitrate */
    S2253_ERROR_VIDEO_CAPTURE,  /* capture device could not be opened */
    S2253_ERROR_VIDEO_ENCODER,  /* video encoder could not be created */
    S2253_ERROR_IMAGE_ENCODER,  /* image encoder could not be created */
    S2253_ERROR_MEMORY,         /* out of memory on device */
    S2253_ERROR_FIFO,           /* fifo error */
    S2253_ERROR_FRAMECOPY_CREATE,
    S2253_ERROR_FRAMECOPY_CONFIG,
    S2253_ERROR_FRAMECOPY_EXEC,
    S2253_ERROR_DISPLAY,
    S2253_ERROR_USB_TIMEOUT,
    S2253_ERROR_WRITER,         /* writer device (usb endpoint) could not be opened */
    S2253_STATE_PAUSED,         /* streaming paused.  if state changes to stopped while paused, the remaining data is not encoded/decoded */
};

enum s2253_audio_input {
    S2253_AUDIO_INPUT_MIC,      /* mic */
    S2253_AUDIO_INPUT_LINE,     /* line in */
};

enum s2253_format {
    S2253_FORMAT_MPEG4,        /* MPEG-4 SP elementary stream */
    S2253_FORMAT_H264,         /* h.264 elementary stream */
    S2253_FORMAT_JPEG,         /* JPEG images */
    S2253_FORMAT_UYVY,         /* YUV 422 packed */
    S2253_FORMAT_YUV420PSEMI,  /* YUV 420 semi-planar (Y in one plane, CrCb in another) */
    S2253_FORMAT_Y8,           /* Y8 grayscale */
    S2253_FORMAT_MPEG2,        /* MPEG-2 main profile elementary stream */
    S2253_FORMAT_RGB24,        /* RGB 24-bpp (actually BGR) */
    S2253_FORMAT_RGB565,       /* RGB 16-bpp */
    S2253_FORMAT_NOVIDEO,      /* no video, just audio over STREAM A or B endpoint.  not yet implemented */
    S2253_FORMAT_PCM = 0x10,   /* PCM audio encoding */
    S2253_FORMAT_AAC = 0x20,   /* AAC audio encoding */
    S2253_FORMAT_G711_ALAW = 0x30, /* g.711 A-law audio encoding */
    S2253_FORMAT_G711_ULAW = 0x40, /* g.711 u-law audio encoding */
    S2253_FORMAT_MP2 = 0x50,   /* MPEG-1 Layer2 audio encoding */
    S2253_FORMAT_MP3 = 0x60,   /* MPEG-1 Layer3 audio encoding */
    S2253_FORMAT_MP4 = 0x100,  /* MP4 fragmented muxed stream */
    S2253_FORMAT_MPEG_TS = 0x200, /* MPEG Transport stream */
    S2253_FORMAT_MPEG_PS = 0x300, /* MPEG Program stream */
};
#define S2253_VIDEO_MASK        (0xf)
#define S2253_AUDIO_MASK        (0xf0)
#define S2253_MUX_MASK          (0xf00)
/* muxer is created by or'ing muxtype,video,audio together:
   S2253_FORMAT_MP4 | S2253_FORMAT_H264 | S2253_FORMAT_AAC */
/* to mux on the host, with audio and video on the same endpoint:
   S2253_FORMAT_H264 | S2253_FORMAT_AAC */


// see http://processors.wiki.ti.com/index.php/DM36x_Rate_Control_Modes
enum s2253_bitrate_mode {
    S2253_BITRATE_MODE_CBR, // constrained/constant (drops frames)
    S2253_BITRATE_MODE_VBR,
    /* the following are H.264 only */
    S2253_BITRATE_MODE_FIXEDQP, // use with INTRAFRAMEQP/INTERPFRAMEQP
    S2253_BITRATE_MODE_CVBR,
    S2253_BITRATE_MODE_FIXEDFRAMESIZE,
    S2253_BITRATE_MODE_CBR1,
    S2253_BITRATE_MODE_VBR1, // default
};

#define S2253_MAX_IMAGE_WIDTH  768
#define S2253_MAX_IMAGE_HEIGHT 768
#define S2253_MIN_IMAGE_WIDTH  128
#define S2253_MIN_IMAGE_HEIGHT  96
/* NOTE: image size cannot exceed this size even though max width and height need to allow larger range for image rotation */
/* image max size = 720x576x3 */
#define S2253_MAX_VIDEO_BUFFER_SIZE   1244160

#define USB_REQ_S2253_PARAM     0x53

/* USB endpoint numbers */
#define S2253_ENDPOINT_STREAM_A         0x81
#define S2253_ENDPOINT_STREAM_B         0x82
#define S2253_ENDPOINT_RECORD           0x83
#define S2253_ENDPOINT_PLAYBACK         0x03
#define S2253_ENDPOINT_GPIO             0x84
#define S2253_ENDPOINT_OUTPUT           0x01
#define S2253_ENDPOINT_OVERLAY          0x02
#define S2253_USB_PACKET_SIZE           (32768)

/* video timestamps use this frequency */
#define S2253_TS_HZ             90000

/* min, max and default JPEG Q values */
#define S2253_MIN_JPEG_Q  10
#define S2253_MAX_JPEG_Q  90
#define S2253_DEF_JPEG_Q  75

/* min, max and default MPEG bitrate values, in kbps */
#define S2253_MIN_MPEG_BITRATE   100
#define S2253_MAX_MPEG_BITRATE 20000
#define S2253_DEF_MPEG_BITRATE  2000

/* min, max and default AAC bitrate values */
#define S2253_MIN_AAC_BITRATE   32
#define S2253_MAX_AAC_BITRATE  512
#define S2253_DEF_AAC_BITRATE  192

/* volume range */
/* note: record volume range 0 (0dB) to 127 (59.5dB) */
#define S2253_RECORD_VOLUME_MIN 0
#define S2253_RECORD_VOLUME_MAX 127
/* note: playback volume range (0dB) to 127 (-63.5dB) */
#define S2253_PLAYBACK_VOLUME_MIN 0
#define S2253_PLAYBACK_VOLUME_MAX 127

/* used for S2253_OUTPUT_MODE parameter */
enum s2253_output_mode {
    S2253_OUTPUT_IDLE, // black screen
    S2253_OUTPUT_CAPTURE, // pass-thru
    S2253_OUTPUT_COLORBARS,
    S2253_OUTPUT_FLASH, // display 1 white frame, then switch to IDLE
    S2253_OUTPUT_STREAM, // firmware use only: use S2253_OUTPUT_FORMAT instead
    S2253_OUTPUT_RAW, // firmware use only: low-latency UYVY or YUV420PSEMI data on OUTPUT endpoint
    S2253_OUTPUT_CAPTURE_GPIO_180, // pass-thru, and rotate 180 when gpio is set (not working yet)
    S2253_OUTPUT_MODE_LAST,
};

enum s2253_xled_mode {
    S2253_XLED_MODE_HVLOCK_BLINK,
    S2253_XLED_MODE_HVLOCK_SOLID,
    S2253_XLED_MODE_ON,
    S2253_XLED_MODE_OFF,
};

#endif /* _S2253_HEADER */

