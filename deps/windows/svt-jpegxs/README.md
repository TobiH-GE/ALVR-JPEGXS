# SVT-JPEG-XS (test-only dependency)

This folder is **not** checked in with content and is **not** required for a normal
ALVR build. It only exists to support the JPEG XS side-by-side comparison test
(`cpp/alvr_server/JpegXsSideEncoder.h`, wired up in
`cpp/platform/win32/CEncoder.cpp` behind `#ifdef ALVR_JPEGXS`).

`alvr/server_openvr/build.rs` auto-detects this folder: if `include/` exists here when
building on Windows, it defines `ALVR_JPEGXS`, compiles `JpegXsSideEncoder.cpp` in, and
links against `SvtJpegxs.lib`. If this folder is empty/missing, the build is completely
unaffected (no JPEG XS code is compiled, no new dependency is required) -- this is how
the original HEVC/AMF/NVENC dump instrumentation (`StreamDumper`) already behaves on its
own, this just adds an optional extra tap.

## Where this comes from

Intel's open source SVT-JPEG-XS encoder/decoder library (BSD-2-Clause-Patent license):
https://github.com/OpenVisualCloud/SVT-JPEG-XS

## How to populate this folder (do this once, on your Windows build machine)

1. Clone the repo somewhere outside of ALVR, e.g.:
   ```powershell
   git clone https://github.com/OpenVisualCloud/SVT-JPEG-XS.git
   ```
2. Install build requirements (see the repo's own README): Visual Studio 2022 or 2019,
   CMake >= 3.16, and the YASM assembler (`yasm.exe` on PATH).
3. Build it:
   ```powershell
   cd SVT-JPEG-XS\Build\windows
   build.bat 2022
   ```
   This produces `SVT-JPEG-XS\Bin\Release\SvtJpegxs.dll` and `SvtJpegxs.lib`.
4. Copy the pieces into this folder so the layout matches `deps/windows/ffmpeg`:
   ```powershell
   xcopy /E /I SVT-JPEG-XS\Source\API                 deps\windows\svt-jpegxs\include
   xcopy /I SVT-JPEG-XS\Bin\Release\SvtJpegxs.lib      deps\windows\svt-jpegxs\lib\
   xcopy /I SVT-JPEG-XS\Bin\Release\SvtJpegxs.dll      deps\windows\svt-jpegxs\bin\
   ```
   Resulting layout:
   ```
   deps/windows/svt-jpegxs/
     include/SvtJpegxs.h
     include/SvtJpegxsEnc.h
     include/SvtJpegxsDec.h   (unused by ALVR, harmless to include)
     lib/SvtJpegxs.lib
     bin/SvtJpegxs.dll
   ```
5. Rebuild ALVR server as usual (e.g. `cargo xtask build-server --release`). Watch the
   build output for `cargo:warning=Building with JPEG XS test encoder enabled (...)` to
   confirm it was picked up.
6. Copy `SvtJpegxs.dll` next to the built driver (`build/alvr_streamer_windows/bin/...`,
   same place `avcodec-61.dll` etc. already live) so the driver process can load it at
   runtime.

## What it produces

While ALVR is streaming, in addition to the real HEVC/H.264/AV1 stream sent to the
headset (completely unaffected), you'll get:

- `C:\Temp\alvr_stream_jpegxs_svtjpegxs.jxs` -- raw JPEG XS codestream (frames
  concatenated back to back).
- `C:\Temp\alvr_stream_jpegxs_svtjpegxs_stats.csv` -- FPS/bitrate CSV in the same format
  as the HEVC dump, for a direct diff.

The target bitrate (`bpp_numerator`/`bpp_denominator`, ~6 bpp on 4:4:4 8-bit by default)
is a constant at the top of `JpegXsSideEncoder.cpp` -- adjust it to get closer to
whatever HEVC bitrate you're comparing against.
