"""Isolated preview conversion worker (no model imports).

Two strategies:

* ``remux``: H.264 that browsers already decode (8-bit 4:2:0) is copied into
  MP4 without re-encoding. Surveillance DVRs export PS/TS/AVI containers that
  no browser accepts, but the stream itself is fine, so a 30 min recording is
  made playable in well under a second and no GPU is involved.
* ``transcode``: everything else (HEVC, MPEG-2, MJPEG, 10-bit H.264, ...) needs
  NVIDIA NVDEC + NVENC; CPU fallback stays disabled so a preview never starves
  inference for minutes.
"""
from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

# Browsers decode these H.264 pixel formats natively inside MP4; anything else
# (High 10, 4:2:2, 4:4:4) has to be transcoded.
COPY_CODECS = {"h264"}
COPY_PIX_FMTS = {"yuv420p", "yuvj420p"}


def remux_copy(source: Path, target: Path, progress) -> bool:
    """Copy a browser-decodable H.264 stream into MP4; False means transcode."""
    # PyAV bundles FFmpeg; no separate ffmpeg executable is required.
    import av
    with av.open(str(source)) as incoming:
        if not incoming.streams.video:
            raise ValueError("No video stream")
        video = incoming.streams.video[0]
        context = video.codec_context
        if context.name.lower() not in COPY_CODECS or context.pix_fmt not in COPY_PIX_FMTS:
            return False
        duration = float(video.duration * video.time_base) if video.duration else (incoming.duration or 0) / 1_000_000
        rate = video.average_rate or video.guessed_rate
        # Transport streams carry no packet duration; the MP4 muxer needs one
        # for the last sample or that frame is lost.
        fallback = round(1 / float(rate) / float(video.time_base)) if rate else 0
        start = None
        pending = None
        count = 0

        def stamp(packet):
            return packet.dts if packet.dts is not None else packet.pts

        def mux(packet, stream, following=None):
            nonlocal start, count
            current = stamp(packet)
            if current is None:
                return
            if start is None:
                start = current
            if not packet.duration:
                later = stamp(following) if following is not None else None
                delta = later - current if later is not None else 0
                packet.duration = delta if delta > 0 else fallback
            if packet.dts is None:
                packet.dts = current
            packet.dts -= start
            if packet.pts is not None:
                packet.pts -= start
            packet.stream = stream
            output.mux(packet)
            count += 1
            if count % 250 == 0 and duration > 0:
                progress(min(99, round(float(packet.dts * packet.time_base) / duration * 100)))

        try:
            with av.open(str(target), "w", format="mp4", options={"movflags": "+faststart"}) as output:
                stream = output.add_stream_from_template(video)
                # Packet copy keeps the original encoding untouched; only the
                # container and its timestamp origin change.
                for packet in incoming.demux(video):
                    if packet.dts is None and packet.pts is None:
                        continue  # demuxer flush marker
                    if pending is not None:
                        mux(pending, stream, packet)
                    pending = packet
                if pending is not None:
                    mux(pending, stream)
                if not count:
                    raise ValueError("No decodable video packets")
        except Exception:
            target.unlink(missing_ok=True)
            raise
    return True


def transcode_gpu(source: Path, target: Path, progress, *, cq: int = 20, gop_seconds: float = 1.0,
                  maxrate_kbps: int = 0) -> None:
    # PyAV bundles FFmpeg; no separate ffmpeg executable is required.
    import av
    with av.open(str(source)) as incoming:
        if not incoming.streams.video:
            raise ValueError("No video stream")
        video = incoming.streams.video[0]
        # CUVID is hardware-only. Unlike HWAccel in PyAV 16, it does not use
        # the get_format callback that can segfault on unsupported input.
        codec = video.codec_context.name
        supported = {"h264", "hevc", "av1", "vp8", "vp9", "mjpeg", "mpeg1video", "mpeg2video", "mpeg4", "vc1"}
        if codec not in supported:
            raise ValueError(f"Selected video stream has no NVIDIA decoder: {codec}")
        decoder_name = {"mpeg1video": "mpeg1", "mpeg2video": "mpeg2"}.get(codec, codec) + "_cuvid"
        decoder = av.CodecContext.create(decoder_name, "r")
        decoder.extradata = video.codec_context.extradata
        decoder.options = {"gpu": "0"}
        width, height = video.codec_context.width, video.codec_context.height
        if width <= 0 or height <= 0 or width * height > 33_177_600:
            raise ValueError("Invalid video dimensions or video exceeds 8K")
        duration = float(video.duration * video.time_base) if video.duration else (incoming.duration or 0) / 1_000_000
        rate = video.average_rate or Fraction(25)
        time_base = Fraction(1, 90000)
        with av.open(str(target), "w", format="mp4", options={"movflags": "+faststart"}) as output:
            stream = output.add_stream("h264_nvenc", rate=rate)
            stream.width, stream.height = width + width % 2, height + height % 2
            stream.pix_fmt = "nv12"
            stream.time_base = time_base
            stream.codec_context.time_base = time_base
            stream.options = {
                "gpu": "0", "preset": "p4", "rc": "vbr", "cq": str(cq), "b": "0",
                "g": str(max(1, round(float(rate) * gop_seconds))), "bf": "0",
            }
            if maxrate_kbps > 0:
                # Bounded VBR keeps review copies small without destroying detail.
                stream.options.update({"maxrate": f"{maxrate_kbps}k", "bufsize": f"{maxrate_kbps * 2}k"})
            start = None
            last_pts = -1
            count = 0
            for frame in (frame for packet in incoming.demux(video) for frame in decoder.decode(packet)):
                # Normalize surveillance timestamps; keep variable frame timing for analysis.
                if frame.pts is not None:
                    stamp = frame.pts * frame.time_base
                    if start is None:
                        start = stamp
                    pts = round((stamp - start) / time_base)
                else:
                    pts = last_pts + max(1, round(1 / rate / time_base))
                # CUVID transfers NVDEC output to host memory. Reformat is a
                # no-op for normal even-sized NV12 video; NVENC still encodes on
                # GPU. Pixel conversion (e.g. 10-bit -> 8-bit) is not decoding.
                frame = frame.reformat(width=stream.width, height=stream.height, format="nv12")
                frame.pts = max(last_pts + 1, pts)
                frame.time_base = time_base
                last_pts = frame.pts
                for packet in stream.encode(frame):
                    output.mux(packet)
                count += 1
                if count % 25 == 0:
                    progress(min(99, round(float(last_pts * time_base) / duration * 100)) if duration > 0 else 0)
            if not count:
                raise ValueError("No decodable video frames")
            for packet in stream.encode():
                output.mux(packet)


def convert(source: Path, target: Path, progress, *, mode: str = "transcode", cq: int = 20,
            gop_seconds: float = 1.0, maxrate_kbps: int = 0) -> str:
    """Build ``target`` and report which strategy was used (``remux``/``transcode``).

    ``auto`` tries the lossless container copy first and only re-encodes when
    the video codec itself is unplayable in a browser. A remux that fails on a
    malformed recording falls back to the GPU instead of failing the job.
    """
    if mode == "auto":
        try:
            if remux_copy(source, target, progress):
                return "remux"
        except Exception as exc:
            target.unlink(missing_ok=True)
            print(f"Remux unavailable, using GPU transcode: {exc}", file=sys.stderr, flush=True)
    transcode_gpu(source, target, progress, cq=cq, gop_seconds=gop_seconds, maxrate_kbps=maxrate_kbps)
    return "transcode"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hardware-optional H.264 conversion worker")
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("--mode", choices=("auto", "transcode"), default="transcode",
                        help="auto = lossless H.264 copy when possible, GPU transcode otherwise")
    parser.add_argument("--cq", type=int, default=20, help="constant-quality level (lower is better)")
    parser.add_argument("--gop", type=float, default=1.0, help="keyframe interval in seconds")
    parser.add_argument("--maxrate", type=int, default=0, help="bitrate cap in kbps (0 = uncapped)")
    args = parser.parse_args()
    try:
        convert(Path(args.source), Path(args.target), lambda value: print(value, flush=True), mode=args.mode,
                cq=args.cq, gop_seconds=args.gop, maxrate_kbps=args.maxrate)
    except Exception as exc:
        print(str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1) from None
