#include "VideoEncoderJpegXs.h"

#ifdef ALVR_JPEGXS

#include "alvr_server/Logger.h"
#include "alvr_server/Settings.h"
#include "alvr_server/bindings.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib> // _putenv_s, for the CUDA opt-in below
#include <cstring>
#include <stdexcept>
#include <string>

// Also used (beyond the shader-reflection diagnostic below) to compile our
// own RGB->YUV420 pixel shaders at runtime via D3DCompile -- see
// EnsureGpuPipeline().
#include <d3d11shader.h>
#include <d3dcompiler.h>
#pragma comment(lib, "d3dcompiler.lib")

namespace {

// Fallback fraction of hardware_concurrency() the JPEG XS encoder's thread
// pool uses when the Dashboard's "Encoder thread count" is left at 0
// (auto), mirroring SoftwareEncodingConfig's thread_count=0="auto"
// convention. Deliberately well under 1.0: unlike AMF/NVENC (dedicated GPU
// hardware, ~zero sustained CPU cost), this is a software encoder running
// alongside SteamVR, the actual VR application (same machine, OpenVR/
// SteamVR driver model), and ALVR's own network/statistics threads.
// Consuming every logical core for our own thread pool starves those --
// observed in practice (back when this ran as an additive side channel
// next to a real hardware-encoded stream) to collapse that other stream's
// adaptive bitrate control to near-nothing, presumably because its own
// threads got starved and it misread the resulting stall as bad network
// conditions. 0.5 leaves half the machine's logical cores for the rest of
// the system. If the Dashboard thread count is set explicitly (nonzero),
// that value is used as-is instead.
constexpr double kJpegXsAutoCpuFraction = 0.5;

// SVT-JPEG-XS's own SVT_LOG() calls default to plain fprintf(stdout/stderr, ...) (see
// SvtLog.c's default_logger) -- invisible in this process, since vrserver.exe runs without
// an attached console. Redirected here into ALVR's own Warn() (not Info(): Warn also routes
// to DriverLog(), landing in vrserver.txt, which Info() alone does not -- see Logger.cpp) so
// the library's own messages (encoder init details, the CUDA-RC opt-in banner from the
// federated-swimming-squirrel work, real errors) are actually visible somewhere greppable.
//
// The callback API gives every call unfiltered (SvtJpegxs.h's own doc comment) -- turns out
// this includes real per-frame SVT_DEBUG() calls (InitStageProcess.c etc., level
// SVT_LOG_DEBUG), confirmed live: ~2 calls/frame at 90fps. Forwarding those through Warn()
// (string formatting + DriverLog() + a file write, every one) would add real per-frame
// overhead to the encode path -- exactly the class of regression this whole project has
// spent a long time hunting elsewhere. Gate on level instead: FATAL/ERROR/WARN (<=
// SVT_LOG_WARN) always pass, plus SVT_LOG_ALL (-1) since that's what the untagged plain
// SVT_LOG() macro always uses regardless of the message's real importance (e.g. the
// CUDA-RC opt-in banner) -- SVT_LOG_INFO/DEBUG (tagged, the per-frame chatter) are dropped.
void JpegXsLogCallback(void* /*context*/, SvtLogLevel level, const char* tag, const char* fmt, va_list args) {
    if (level > SVT_LOG_WARN && level != SVT_LOG_ALL) {
        return;
    }
    char buf[1024];
    vsnprintf(buf, sizeof(buf), fmt, args);
    if (tag && tag[0] != '\0') {
        Warn("SvtJpegxs[%s]: %s", tag, buf);
    } else {
        Warn("SvtJpegxs: %s", buf);
    }
}

} // namespace

VideoEncoderJpegXs::VideoEncoderJpegXs(
    std::shared_ptr<CD3DRender> d3dRender, uint32_t width, uint32_t height, bool sideChannelOnly
)
    : m_d3dRender(d3dRender)
    , m_width(width)
    , m_height(height)
    , m_chromaWidth(width / 2)
    , m_chromaHeight(height / 2)
    , m_sideChannelOnly(sideChannelOnly) { }

VideoEncoderJpegXs::~VideoEncoderJpegXs() { Shutdown(); }

void VideoEncoderJpegXs::Initialize() {
    // Decide the input path FIRST: it changes how the staging textures below are
    // created, and those are created well before the _putenv_s block further down
    // that turns the same setting into the library's own opt-in.
    //
    // With CUDA encoding on, the frame never leaves the GPU. We hand the encoder
    // the D3D11 texture and it maps it through CUDA interop, the same shape NVENC
    // is used in: the encoder owns the input surface and each frame is a GPU-side
    // CopyResource into it. That removes a Map()/memcpy() readback of the full
    // frame plus the upload of the same pixels back to the device.
    //
    // There is deliberately no CPU fallback on this path -- see device_input_only
    // in SvtJpegxs.h. The host planes are never filled, so a frame the GPU path
    // cannot take is dropped rather than encoded from whatever they still hold.
    // Its own switch, not implied by CUDA. The compute offload and the zero-copy
    // input are independent choices, and only the second one touches D3D11 -- so
    // when the handover misbehaves, the encoder can keep the whole GPU pipeline
    // and go back to the readback for its input, rather than the two standing or
    // falling together.
    //
    // CPU/GPU hybrid ("CUDA: left half on the CPU, right half on the GPU"): the left
    // column(s) of the split encode on the CPU, the right one(s) on the GPU, concurrently
    // -- see the per-column loop further down. The CPU columns read the host planes, so
    // the hybrid needs the readback and turns the zero-copy input off: on that path the
    // host planes are never filled.
    const bool cudaHybrid = Settings::Instance().m_jpegXsUseCuda && Settings::Instance().m_jpegXsCudaHybridCpuLeft
        && Settings::Instance().m_jpegXsColumnsNum > 1;
    // Hybrid without the round trip ("read back only the CPU's columns"): the GPU columns take
    // the same device pointers the zero-copy input hands out, and only the CPU columns' part of
    // the picture is copied to system memory. What the plain hybrid does instead is read the
    // WHOLE frame back and then let the library upload the GPU half again -- pixels travelling
    // down and straight back up. At 1728x512 that readback measures 0.31-0.5 ms; at 4288x1664 it
    // is eight times the pixels, so 2.5-4 ms against an 11.1 ms frame budget, half of it wasted.
    m_hybridZeroCopy = cudaHybrid && Settings::Instance().m_jpegXsCudaD3d11Input
        && Settings::Instance().m_jpegXsCudaHybridZeroCopy;
    m_useD3d11Input = Settings::Instance().m_jpegXsUseCuda && Settings::Instance().m_jpegXsCudaD3d11Input
        && (!cudaHybrid || m_hybridZeroCopy);
    m_stagingSlots = m_useD3d11Input ? 3 : 2;
    // Columns 0..m_cpuColumns-1 encode on the CPU, the rest on the GPU. Kept as a member because
    // both the readback (which copies exactly this region) and the per-column encode loop (which
    // decides host vs device pointers per column) need the same number.
    m_cpuColumns = cudaHybrid ? Settings::Instance().m_jpegXsColumnsNum / 2 : 0;

    // svt_jpeg_xs_set_log_callback() is a process-wide global (not per-instance) --
    // register once regardless of how many VideoEncoderJpegXs instances end up calling
    // Initialize() (the real encoder, plus a possible additive side-channel instance).
    // See JpegXsLogCallback's own doc comment for why this exists.
    {
        static std::atomic_bool s_logCallbackRegistered { false };
        bool expected = false;
        if (s_logCallbackRegistered.compare_exchange_strong(expected, true)) {
            svt_jpeg_xs_set_log_callback(JpegXsLogCallback, nullptr);
        }
    }

    // CPU-readable staging copies of our GPU-produced Y/Cb/Cr planes. Two of
    // each, for double-buffered readback -- see Transmit(). The GPU
    // render pipeline itself (m_gpuYTex/m_gpuCbTex/m_gpuCrTex and friends)
    // is set up lazily on the first Transmit() call instead of here -- see
    // EnsureGpuPipeline() for why. Three separate ordinary-format staging
    // textures (not one NV12 texture's subresources) to match
    // m_gpuYTex/m_gpuCbTex/m_gpuCrTex -- see the top-of-file comment for why
    // we don't use a planar NV12 resource here.
    D3D11_TEXTURE2D_DESC yDesc = {};
    yDesc.Width = m_width;
    yDesc.Height = m_height;
    yDesc.MipLevels = 1;
    yDesc.ArraySize = 1;
    yDesc.Format = DXGI_FORMAT_R8_UNORM;
    yDesc.SampleDesc.Count = 1;
    // With the encoder reading these textures directly, they must be ordinary GPU
    // resources: a STAGING resource cannot be registered with CUDA. Only the
    // readback path needs CPU access, and on the CUDA path there is no readback.
    if (m_useD3d11Input) {
        yDesc.Usage = D3D11_USAGE_DEFAULT;
        yDesc.CPUAccessFlags = 0;
        yDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    } else {
        yDesc.Usage = D3D11_USAGE_STAGING;
        yDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        yDesc.BindFlags = 0;
    }

    // Plain R8_UNORM, chroma-sized -- not one interleaved R8G8 texture. See
    // m_gpuCbTex/m_gpuCrTex in the header for why: the GPU now writes Cb/Cr
    // as two separate render targets, so these staging copies (and the
    // readback below) never need to deinterleave anything.
    D3D11_TEXTURE2D_DESC chromaDesc = yDesc;
    chromaDesc.Width = m_chromaWidth;
    chromaDesc.Height = m_chromaHeight;

    for (int i = 0; i < m_stagingSlots; i++) {
        HRESULT hr = m_d3dRender->GetDevice()->CreateTexture2D(&yDesc, nullptr, &m_stagingYTex[i]);
        if (FAILED(hr)) {
            throw MakeException(
                "VideoEncoderJpegXs: failed to create Y staging texture %d (hr=0x%08lx).", i, hr
            );
        }
        hr = m_d3dRender->GetDevice()->CreateTexture2D(&chromaDesc, nullptr, &m_stagingCbTex[i]);
        if (FAILED(hr)) {
            throw MakeException(
                "VideoEncoderJpegXs: failed to create Cb staging texture %d (hr=0x%08lx).", i, hr
            );
        }
        hr = m_d3dRender->GetDevice()->CreateTexture2D(&chromaDesc, nullptr, &m_stagingCrTex[i]);
        if (FAILED(hr)) {
            throw MakeException(
                "VideoEncoderJpegXs: failed to create Cr staging texture %d (hr=0x%08lx).", i, hr
            );
        }
    }

    // The narrow CPU-column textures are created further down, once m_columnsNum and
    // m_columnWidth are known -- see "hybrid zero-copy" after the column setup.

    SvtJxsErrorType_t err = svt_jpeg_xs_encoder_load_default_parameters(
        SVT_JPEGXS_API_VER_MAJOR, SVT_JPEGXS_API_VER_MINOR, &m_enc
    );
    if (err != SvtJxsErrorNone) {
        throw MakeException(
            "VideoEncoderJpegXs: svt_jpeg_xs_encoder_load_default_parameters failed (%d).", (int)err
        );
    }

    // Real-time throughput lever: by default svt_jpeg_xs_encoder_load_default_parameters()
    // leaves threads_num at 0 ("lowest possible number of threads"), which is
    // nowhere near enough to keep up with a 90 Hz VR stream. "Encoder thread
    // count" in the Dashboard's JPEG XS settings (0 = auto) controls this;
    // see kJpegXsAutoCpuFraction above for the auto heuristic.
    unsigned int hwThreads = std::thread::hardware_concurrency();
    if (hwThreads == 0) {
        hwThreads = 8;
    }
    uint32_t configuredThreads = Settings::Instance().m_jpegXsThreadCount;
    uint32_t encodeThreads = configuredThreads > 0
        ? configuredThreads
        : (uint32_t)std::max(1.0, std::round(hwThreads * kJpegXsAutoCpuFraction));

    // Bits-per-pixel target, from the Dashboard's JPEG XS settings (default
    // 6.0, a "high quality" starting point for direct comparison against
    // HEVC). SVT-JPEG-XS defines bpp relative to the luma sample count
    // regardless of chroma format, so this carries over unchanged from the
    // earlier 4:2:2 CPU implementation. bpp_denominator=100 gives 0.01 bpp
    // precision on the configured float value.
    float configuredBpp = Settings::Instance().m_jpegXsBitsPerPixel;
    uint32_t bppDenominator = 100;
    uint32_t bppNumerator = (uint32_t)std::max(1.0f, std::round(configuredBpp * (float)bppDenominator));

    m_enc.source_width = m_width;
    m_enc.source_height = m_height;
    m_enc.input_bit_depth = 8;
    // 4:2:0: the RGB->YUV conversion now happens on the GPU via ALVR's own
    // rgbtoyuv420.hlsl shader (see EnsureGpuPipeline()), which produces
    // half-width/half-height 4:2:0 chroma (into m_gpuCbTex/m_gpuCrTex) -- one step more
    // aggressive chroma subsampling than the previous CPU implementation's
    // 4:2:2, in exchange for eliminating almost all CPU conversion work and
    // roughly halving the GPU readback size again (~1.5 bytes/pixel vs. ~2
    // for 4:2:2).
    m_enc.colour_format = COLOUR_FORMAT_PLANAR_YUV420;
    m_enc.bpp_numerator = bppNumerator;
    m_enc.bpp_denominator = bppDenominator;
    m_enc.threads_num = encodeThreads;
    m_enc.verbose = VERBOSE_ERRORS;

    // Coding-tool configuration, stated EXPLICITLY rather than inherited.
    //
    // Everything above overrides a field; everything not named here keeps whatever
    // svt_jpeg_xs_encoder_load_default_parameters() set a few lines earlier. Those
    // defaults are real and non-zero (EncHandle.c:545-593), which is easy to miss and
    // has already caused one round of live failures in the CUDA offload work: that
    // effort was built on the assumption this struct was zero-initialised, and
    // rate_control_mode silently defaults to RC_CBR_PER_PRECINCT_MOVE_PADDING (1),
    // coding_significance to 1, and coding_raw_disable to 0 (i.e. raw-mode ENABLED).
    // Naming them here means a reader sees ALVR's real configuration in one place
    // instead of having to cross-reference the library's defaults.
    //
    // The two values below are also what the CUDA whole-frame path requires, and it
    // rejects the config otherwise (falling back to CPU, which is safe but pointless):
    //
    //   rate_control_mode = RC_CBR_PER_PRECINCT rather than ..._MOVE_PADDING. The
    //   difference is not the rate-control decision -- PackStageProcess.c calls the
    //   identical rate_control_precinct() either way -- but a sequential carry of one
    //   precinct's leftover padding bytes into the next precinct's budget within a
    //   slice, which breaks the "every precinct independent" assumption the batched
    //   GPU search is built on.
    //
    //   coding_raw_disable = 1. precinct_get_budget_bytes() has a raw-packet selection
    //   branch that the CUDA cost model does not implement.
    //
    // Measured cost of both, at 4288x1664 / bpp 6.0 / identical bitrate: 35.79 dB
    // PSNR-Y before, 35.52 dB after -- 0.27 dB, roughly 3% bitrate. Notably raw-mode
    // contributes 0.00 dB of that once MOVE_PADDING is off (it only mattered in
    // combination with it), so the whole cost is the padding carry.
    //
    // coding_significance stays ENABLED: it is worth 0.44 dB and the CUDA path now
    // models it (SVT-JPEG-XS commit 8e02406, byte-identical to the CPU encode).
    //
    // Written as a literal because the RateControlType enum lives in the library's
    // internal Encoder.h, not in the public SvtJpegxsEnc.h, which types this field as a
    // plain uint32_t. 0 == RC_CBR_PER_PRECINCT, 1 == RC_CBR_PER_PRECINCT_MOVE_PADDING.
    // (The public header's own comment lists both 0 and 1 as "CBR budget per precinct" --
    // it does not mention the padding movement that actually distinguishes them.)
    // Dashboard "Use CUDA (NVIDIA GPU) for encoding" -> the library's own opt-in.
    //
    // SvtJpegxs gates its whole-frame GPU path on getenv("ALVR_JXS_CUDA_FRAME"), cached in
    // a function-local static on first use (PictureControlSet.c's jxs_cuda_frame_opt_in),
    // so this must happen BEFORE svt_jpeg_xs_encoder_init() below. Driving the env var
    // from here rather than expecting the user to set it is the point of the setting: an
    // env var has to exist before STEAM starts, since SteamVR inherits Steam's environment
    // and not the current desktop session's -- which makes it very easy to believe the GPU
    // path is active when it silently is not.
    //
    // The empty-string branch is not redundant. _putenv_s with "" REMOVES the variable on
    // Windows, which is what makes turning the setting off actually work even when a
    // persistent user-level ALVR_JXS_CUDA_FRAME exists in the environment -- otherwise the
    // dashboard toggle would appear to do nothing in exactly the setup most likely to have
    // that variable left over from manual testing.
    _putenv_s("ALVR_JXS_CUDA_FRAME", Settings::Instance().m_jpegXsUseCuda ? "1" : "");
    // Entropy coding on the GPU only makes sense on top of the stage offload, so it
    // is gated on both. Same _putenv_s("") removal reasoning as above.
    const bool entropyOnGpu = Settings::Instance().m_jpegXsUseCuda && Settings::Instance().m_jpegXsCudaEntropyOnGpu;
    _putenv_s("ALVR_JXS_CUDA_PACK", entropyOnGpu ? "1" : "");
    // Where the library appends its periodic per-frame timing report (every 600
    // frames per column): the GPU stage split plus the full path of a frame through
    // the encoder -- queue wait, GPU precompute, pack + final stage, output queue.
    // The same report goes to stderr, which inside a SteamVR driver DLL goes nowhere;
    // this file is the only place it can be read after a live run. Next to
    // jpegxs_server_diag.log and alvr_graph_stats.csv on purpose.
    _putenv_s("ALVR_JXS_CUDA_REPORT_FILE", Settings::Instance().m_jpegXsUseCuda ? "C:\\Temp\\jpegxs_cuda_stages.log" : "");
    // How threads wait for the GPU. CUDA's default (cudaDeviceScheduleAuto) SPINS whenever the
    // process has fewer CUDA contexts than logical processors -- i.e. always here -- so every
    // host-side sync in the encoder (rate-control readback, per-stage joins, the D3D11
    // handover; about 30 per column and frame) keeps a core at 100 % while the GPU works.
    // Under a game those waits add up to several ms per frame and column: roughly a whole core
    // taken from the game for nothing. The library applies this before its first CUDA call
    // (svt_jpeg_xs_encoder_init and the D3D11 handover); the state is logged after init below.
    _putenv_s(
        "ALVR_JXS_CUDA_BLOCKING_SYNC",
        (Settings::Instance().m_jpegXsUseCuda && Settings::Instance().m_jpegXsCudaBlockingSync) ? "1" : ""
    );
    Info("VideoEncoderJpegXs: CUDA encoding %s, entropy coding on GPU %s, input %s.\n",
         Settings::Instance().m_jpegXsUseCuda ? "ENABLED (dashboard setting)" : "disabled",
         entropyOnGpu ? "ENABLED" : "disabled",
         m_useD3d11Input ? "the D3D11 texture directly (no readback, no CPU fallback)"
                         : "read back to system memory");

    m_enc.rate_control_mode = 0;
    m_enc.coding_raw_disable = 1;
    m_enc.coding_significance = 1;
    m_enc.coding_vertical_prediction_mode = 0;
    m_enc.coding_signs_handling = 0;

    // Column split ("Column splits" in the Dashboard's JPEG XS settings, default 2): NOT
    // implemented via SVT-JPEG-XS's own native column-tiled precincts (svt_jpeg_xs_encoder_
    // api_t::columns_num, added for that attempt) -- that field's plumbing all the way
    // through calc_precinct_dimension() genuinely exists, but the actual bit-PACKING
    // pipeline (PackStageProcess.c/PrecinctEnc.c/GcStageProcess.c) never learned to write
    // more than one precinct per row, and worse, the DWT stage computes each row's wavelet
    // transform directly from raw pixels with no per-column boundary handling at all --
    // fixing that properly means correct wavelet edge handling at the column seam, real
    // codec-level DSP work with zero existing test coverage upstream.
    //
    // Instead: our own simple, orthogonal scheme. Each "column" is encoded as its own
    // completely independent, ordinary (single-column) JPEG XS image -- source_width =
    // m_columnWidth instead of m_width, everything else identical -- using N separate
    // encoder instances (m_columnEncs). The four resulting codestreams are concatenated
    // behind a tiny custom header (see the "ALVS" magic in EncodeWorkerLoop) into one
    // logical "frame" that VideoSend() ships as a single unit, same as before. Since each
    // piece is just an ordinary, already fully-working JPEG XS image at a narrower width,
    // this sidesteps the wavelet-boundary problem entirely -- there's no seam to get
    // wrong, because each half's edges are genuine picture edges, not tile boundaries.
    // Splits into m_columnsNum <= 1 columns, or falls back to 1 if source_width doesn't
    // divide evenly (see Initialize() below) -- disabled and behaves exactly as before in
    // that case, m_enc is still the only encoder and the wrapper header is never emitted.
    m_columnsNum = Settings::Instance().m_jpegXsColumnsNum;
    if (m_columnsNum < 1) {
        m_columnsNum = 1;
    }
    if (m_columnsNum > 1 && (m_width % m_columnsNum) != 0) {
        Warn(
            "VideoEncoderJpegXs: width %u doesn't divide evenly into %u columns -- falling back "
            "to 1 (no split).\n",
            m_width,
            m_columnsNum
        );
        m_columnsNum = 1;
    }
    m_columnWidth = m_width / m_columnsNum;

    // Hybrid zero-copy: a SECOND, narrow set of textures, CPU-readable and exactly as wide as
    // the CPU columns. The set created above is a DEFAULT resource now, because CUDA cannot
    // register a STAGING one -- so the CPU needs its own, and there is no reason for it to be
    // wider than the part the CPU actually encodes. CopySubresourceRegion fills these from the
    // same GPU planes, cropped. Recomputed here rather than at the input-mode decision above,
    // because m_columnsNum can still fall back to 1 in between.
    m_cpuColumns = cudaHybrid ? m_columnsNum / 2 : 0;
    if (m_hybridZeroCopy && m_cpuColumns == 0) {
        // No CPU columns left to read back (the split fell back to a single column). Everything
        // runs on the GPU from the device pointers, which is exactly the plain zero-copy path.
        m_hybridZeroCopy = false;
    }
    if (m_hybridZeroCopy) {
        m_cpuReadbackWidth = m_cpuColumns * m_columnWidth;
        m_cpuReadbackChromaWidth = m_cpuReadbackWidth / (m_width / m_chromaWidth);

        D3D11_TEXTURE2D_DESC cpuYDesc = {};
        cpuYDesc.Width = m_cpuReadbackWidth;
        cpuYDesc.Height = m_height;
        cpuYDesc.MipLevels = 1;
        cpuYDesc.ArraySize = 1;
        cpuYDesc.Format = DXGI_FORMAT_R8_UNORM;
        cpuYDesc.SampleDesc.Count = 1;
        cpuYDesc.Usage = D3D11_USAGE_STAGING;
        cpuYDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        cpuYDesc.BindFlags = 0;
        D3D11_TEXTURE2D_DESC cpuChromaDesc = cpuYDesc;
        cpuChromaDesc.Width = m_cpuReadbackChromaWidth;
        cpuChromaDesc.Height = m_chromaHeight;

        for (int i = 0; i < m_stagingSlots; i++) {
            HRESULT hr
                = m_d3dRender->GetDevice()->CreateTexture2D(&cpuYDesc, nullptr, &m_cpuStagingYTex[i]);
            if (SUCCEEDED(hr)) {
                hr = m_d3dRender->GetDevice()->CreateTexture2D(
                    &cpuChromaDesc, nullptr, &m_cpuStagingCbTex[i]
                );
            }
            if (SUCCEEDED(hr)) {
                hr = m_d3dRender->GetDevice()->CreateTexture2D(
                    &cpuChromaDesc, nullptr, &m_cpuStagingCrTex[i]
                );
            }
            if (FAILED(hr)) {
                throw MakeException(
                    "VideoEncoderJpegXs: failed to create CPU-column staging texture %d "
                    "(hr=0x%08lx).",
                    i,
                    hr
                );
            }
        }
        // Warn, not Info: Info never reaches vrserver.txt, and this is the line that shows the
        // round trip is actually gone.
        Warn(
            "VideoEncoderJpegXs: hybrid zero-copy -- GPU columns read the D3D11 texture directly, "
            "only columns 0..%u are read back (%ux%u luma instead of %ux%u).\n",
            m_cpuColumns - 1,
            m_cpuReadbackWidth,
            m_height,
            m_width,
            m_height
        );
    }

    // Only meaningful with a split, and only for the encoder that actually streams: a
    // side-channel instance must never touch VideoSend (see EncodeWorkerLoop's own check).
    m_pipelineColumns
        = Settings::Instance().m_jpegXsPipelineColumns && m_columnsNum > 1 && !m_sideChannelOnly;
    m_slicePackets
        = Settings::Instance().m_jpegXsSlicePackets && m_columnsNum > 1 && !m_sideChannelOnly;
    if (m_slicePackets) {
        // Pin the slice height rather than inherit the library default: the client rebuilds a
        // column from packet indices, so both sides must agree on how many units a column has.
        m_enc.slice_height = kSliceHeightLines;
        m_enc.slice_packetization_mode = 1;
        m_unitsPerColumn = (m_height + kSliceHeightLines - 1) / kSliceHeightLines + 1;
        Warn(
            "VideoEncoderJpegXs: slice packetization on -- %u units per column (1 header + %u "
            "slices), %u packets per frame. Dump to disk is not written in this mode.\n",
            m_unitsPerColumn,
            m_unitsPerColumn - 1,
            m_unitsPerColumn * m_columnsNum
        );
    }
    if (m_pipelineColumns && !m_slicePackets) {
        Warn("VideoEncoderJpegXs: column pipelining on -- one packet per column, sent as soon "
             "as that column is encoded. Dump to disk is not written in this mode.\n");
    }

    if (m_columnsNum <= 1) {
        // Unchanged from before this feature existed: m_enc is source_width = m_width, one
        // instance, no wrapper header.
        m_enc.source_width = m_width;
        err = svt_jpeg_xs_encoder_init(SVT_JPEGXS_API_VER_MAJOR, SVT_JPEGXS_API_VER_MINOR, &m_enc);
        if (err != SvtJxsErrorNone) {
            throw MakeException("VideoEncoderJpegXs: svt_jpeg_xs_encoder_init failed (%d).", (int)err);
        }
        m_encInitialized = true;
    } else {
        m_columnEncs.resize(m_columnsNum);
        m_columnBitstreamBufs.resize(m_columnsNum);

        // Per-column thread budget. Until the send/get split in EncodeWorkerLoop (see its
        // "Phase 1"/"Phase 2" comments), columns encoded strictly SEQUENTIALLY -- only one
        // instance's pool was ever active at a time, so handing each instance the full
        // encodeThreads count happened to land on the intended total by accident. Now that
        // all columns encode concurrently, that same arithmetic would multiply real thread
        // demand by m_columnsNum and blow straight past kJpegXsAutoCpuFraction's "leave
        // half the machine for the game" intent.
        //
        // So: treat the configured (or auto-derived) value as the TOTAL budget across all
        // columns and divide it -- but add SVT's fixed per-instance overhead back on top,
        // because it is not usable parallelism: svt_jpeg_xs_encoder_init() computes
        // pack_stage_threads_num = MAX(1, threads_num - 2 - dwt_stage_threads_num)
        // (EncHandle.c), i.e. every instance spends 2 threads plus the DWT stage before any
        // pack-stage parallelism starts. Dividing without compensating would quietly
        // collapse each column to a single pack thread.
        uint32_t perColumnThreads = std::max(3u, encodeThreads / m_columnsNum + 2u);
        Info(
            "VideoEncoderJpegXs: thread budget %u total -> %u per column across %u columns "
            "(+2 each for SVT's own fixed per-instance overhead).\n",
            encodeThreads,
            perColumnThreads,
            m_columnsNum
        );

        // CPU/GPU hybrid: the left half of the columns (the smaller half for an odd count)
        // opts out of the library's process-wide CUDA path via cuda_disable and encodes on
        // the CPU; the rest stay on the GPU. The thread budget then belongs to the CPU
        // columns. A GPU column with GPU entropy coding needs only SVT's fixed minimum;
        // without it, its packer still runs on the CPU and keeps the normal share.
        // Same number the readback crop and the per-column pointer choice use -- one source of
        // truth, set with m_columnWidth above.
        const uint32_t cpuColumns = m_cpuColumns;
        uint32_t cpuColumnThreads = perColumnThreads;
        // A GPU column with GPU entropy coding does ~0.3 ms of CPU work per frame (pack+final),
        // so SVT's fixed minimum is all it can use. Handing it the auto budget (half the machine)
        // only put more encoder threads in the scheduler next to the game's. If the GPU path ever
        // refuses a frame, that frame encodes on the CPU with these 3 threads: slower, but it
        // stays correct, and it is counted in the CUDA report.
        // Dashboard: "CUDA: GPU columns get SVT's minimum thread count". Off restores the full
        // budget for GPU-only, which is what it did before the setting existed; the hybrid below
        // always takes the minimum, as it did then too.
        const bool gpuColumnMinThreads = Settings::Instance().m_jpegXsCudaGpuColumnMinThreads;
        uint32_t gpuColumnThreads = (entropyOnGpu && gpuColumnMinThreads) ? 3u : perColumnThreads;
        if (cudaHybrid) {
            gpuColumnThreads = entropyOnGpu ? 3u : perColumnThreads;
            cpuColumnThreads = std::max(3u, encodeThreads / cpuColumns + 2u);
            // Warn, not Info: Info never reaches vrserver.txt, and this is the line that
            // shows the hybrid is actually in effect.
            Warn(
                "VideoEncoderJpegXs: CPU/GPU hybrid -- columns 0..%u on the CPU (%u threads each), "
                "columns %u..%u on the GPU (%u threads each), %s.\n",
                cpuColumns - 1,
                cpuColumnThreads,
                cpuColumns,
                m_columnsNum - 1,
                gpuColumnThreads,
                m_hybridZeroCopy ? "GPU columns on zero-copy input, CPU columns read back"
                                 : "zero-copy input off"
            );
        }

        if (!cudaHybrid && Settings::Instance().m_jpegXsUseCuda) {
            Warn(
                "VideoEncoderJpegXs: GPU-only -- %u encoder threads per column (%s).\n",
                gpuColumnThreads,
                !entropyOnGpu       ? "entropy coding on the CPU, full budget"
                    : gpuColumnMinThreads ? "entropy coding on the GPU, SVT's fixed minimum"
                                          : "entropy coding on the GPU, full budget by setting"
            );
        }
        for (uint32_t i = 0; i < m_columnsNum; i++) {
            const bool onCpu = i < cpuColumns;
            m_columnEncs[i] = m_enc; // copy the shared bpp/colour_format/threads/verbose settings
            m_columnEncs[i].source_width = m_columnWidth;
            m_columnEncs[i].threads_num = onCpu ? cpuColumnThreads : gpuColumnThreads;
            m_columnEncs[i].cuda_disable = onCpu ? 1 : 0;
            if (m_slicePackets) {
                // Wakes the interleaved collect in EncodeWorkerLoop when a unit is ready.
                m_columnEncs[i].callback_get_data_available = &VideoEncoderJpegXs::OnEncoderOutputAvailable;
                m_columnEncs[i].callback_get_data_available_context = this;
            }
            err = svt_jpeg_xs_encoder_init(SVT_JPEGXS_API_VER_MAJOR, SVT_JPEGXS_API_VER_MINOR, &m_columnEncs[i]);
            if (err != SvtJxsErrorNone) {
                throw MakeException(
                    "VideoEncoderJpegXs: svt_jpeg_xs_encoder_init failed for column %u/%u (%d).",
                    i,
                    m_columnsNum,
                    (int)err
                );
            }
        }
        m_encInitialized = true;
        Info(
            "VideoEncoderJpegXs: column split active -- %u columns of %u px each (source width "
            "%u), independent encoder instances, custom ALVS wrapper framing.\n",
            m_columnsNum,
            m_columnWidth,
            m_width
        );
    }

    if (Settings::Instance().m_jpegXsUseCuda) {
        // Warn, not Info: Info never reaches vrserver.txt, and this is the line that says whether
        // the encoder's GPU waits sleep or spin -- "requested" is not "applied".
        const int syncState = svt_jpeg_xs_cuda_blocking_sync_state();
        Warn(
            "VideoEncoderJpegXs: CUDA blocking sync %s.\n",
            syncState == 1   ? "APPLIED -- threads waiting for the GPU sleep"
                : syncState < 0 ? "requested but NOT applied (CUDA context already existed) -- waits spin"
                                : "off -- threads waiting for the GPU spin a core"
        );
    }

    size_t planeSize = (size_t)m_width * m_height;
    size_t chromaPlaneSize = (size_t)m_chromaWidth * m_chromaHeight;

    // Encode stage's local working buffers.
    m_planeY.resize(planeSize);
    m_planeCb.resize(chromaPlaneSize);
    m_planeCr.resize(chromaPlaneSize);

    // Readback's private scratch buffers and the mailbox it hands off to.
    m_readbackScratchY.resize(planeSize);
    m_readbackScratchCb.resize(chromaPlaneSize);
    m_readbackScratchCr.resize(chromaPlaneSize);
    m_yuvMailboxY.resize(planeSize);
    m_yuvMailboxCb.resize(chromaPlaneSize);
    m_yuvMailboxCr.resize(chromaPlaneSize);

    // Generous margin over the nominal target size, in case a frame briefly
    // overshoots the rate-control budget.
    uint32_t targetBytes = (uint32_t
    )(((uint64_t)m_width * m_height * m_enc.bpp_numerator / m_enc.bpp_denominator + 7) / 8);
    m_bitstreamBuf.resize((size_t)targetBytes * 2 + 4096);

    if (m_columnsNum > 1) {
        // Same margin-over-target logic as m_bitstreamBuf above, scaled down to one
        // column's pixel count (columns_num divides the width evenly -- see the fallback-
        // to-1 check above -- so this is exact, not approximate).
        uint32_t columnTargetBytes = (uint32_t)(
            ((uint64_t)m_columnWidth * m_height * m_enc.bpp_numerator / m_enc.bpp_denominator + 7) / 8
        );
        size_t columnBufSize = (size_t)columnTargetBytes * 2 + 4096;
        for (uint32_t i = 0; i < m_columnsNum; i++) {
            m_columnBitstreamBufs[i].resize(columnBufSize);
        }
        // Header (magic + columns_num + length table) + all columns at their nominal size;
        // EncodeWorkerLoop grows this further on demand if a frame ever exceeds it.
        m_wrapperBuf.resize(4 + 4 + (size_t)m_columnsNum * 4 + columnBufSize * m_columnsNum);
    }

    if (Settings::Instance().m_jpegXsDumpToDisk) {
        const std::string& outputDir = Settings::Instance().m_jpegXsOutputDirectory;
        m_streamDumper = std::make_unique<StreamDumper>("jpegxs", ".jxs", "svtjpegxs", outputDir);
        if (!m_streamDumper->StreamFileOpen()) {
            Error(
                "VideoEncoderJpegXs: failed to open JPEG XS dump file in %s.\n", outputDir.c_str()
            );
        } else {
            Info(
                "VideoEncoderJpegXs: dumping the real JPEG XS stream to %s (bpp=%u/%u, GPU "
                "RGB->YUV420, %u encode threads = %u/%u logical cores). Note: this is a "
                "software encoder -- unlike AMF/NVENC it competes with the VR application's "
                "own CPU usage on this machine.\n",
                outputDir.c_str(),
                bppNumerator,
                bppDenominator,
                encodeThreads,
                encodeThreads,
                hwThreads
            );
        }
    } else {
        Info(
            "VideoEncoderJpegXs: running without disk dump (Dashboard 'Dump stream to disk' "
            "disabled). bpp=%u/%u, GPU RGB->YUV420, %u encode threads = %u/%u logical cores. "
            "Note: this is a software encoder -- unlike AMF/NVENC it competes with the VR "
            "application's own CPU usage on this machine.\n",
            bppNumerator,
            bppDenominator,
            encodeThreads,
            encodeThreads,
            hwThreads
        );
    }

    m_exiting = false;
    m_readbackThread = std::thread(&VideoEncoderJpegXs::ReadbackWorkerLoop, this);
    m_encodeThread = std::thread(&VideoEncoderJpegXs::EncodeWorkerLoop, this);
}

bool VideoEncoderJpegXs::EnsureGpuPipeline(ID3D11Texture2D* sourceTexture) {
    if (m_gpuPipelineReady) {
        return true;
    }
    if (m_gpuSetupFailed) {
        return false;
    }

    try {
        auto* device = m_d3dRender->GetDevice();
        auto* context = m_d3dRender->GetContext();

        std::vector<uint8_t> quadShaderCSO(
            QUAD_SHADER_CSO_PTR, QUAD_SHADER_CSO_PTR + QUAD_SHADER_CSO_LEN
        );
        m_quadVertexShader = d3d_render_utils::CreateVertexShader(device, quadShaderCSO);

        // TEMPORARY DIAGNOSTIC: reflect the actual input signature of
        // QuadVertexShader.cso instead of continuing to guess whether it's
        // SV_VertexID-procedural or needs real POSITION/TEXCOORD/VIEW
        // vertex data.
        {
            Microsoft::WRL::ComPtr<ID3D11ShaderReflection> reflector;
            // __uuidof(...) instead of the extern IID_ID3D11ShaderReflection
            // symbol -- the latter lives in dxguid.lib, which isn't linked
            // here; __uuidof is resolved by the compiler itself, no extra
            // lib needed.
            HRESULT hr = D3DReflect(
                quadShaderCSO.data(),
                quadShaderCSO.size(),
                __uuidof(ID3D11ShaderReflection),
                (void**)reflector.GetAddressOf()
            );
            if (SUCCEEDED(hr)) {
                D3D11_SHADER_DESC shaderDesc;
                reflector->GetDesc(&shaderDesc);
                Info(
                    "VideoEncoderJpegXs: QUAD SHADER REFLECTION: %u input parameters.\n",
                    shaderDesc.InputParameters
                );
                for (UINT i = 0; i < shaderDesc.InputParameters; i++) {
                    D3D11_SIGNATURE_PARAMETER_DESC paramDesc;
                    reflector->GetInputParameterDesc(i, &paramDesc);
                    Info(
                        "VideoEncoderJpegXs: QUAD SHADER REFLECTION: input[%u] = %s%u "
                        "(register=%u, mask=0x%x, componentType=%d)\n",
                        i,
                        paramDesc.SemanticName,
                        paramDesc.SemanticIndex,
                        paramDesc.Register,
                        paramDesc.Mask,
                        (int)paramDesc.ComponentType
                    );
                }
            } else {
                Info("VideoEncoderJpegXs: QUAD SHADER REFLECTION: D3DReflect failed hr=0x%08lx\n", (unsigned long)hr);
            }
        }

        // Own input layout + vertex buffer for the quad shader -- see the
        // long comment on the member declarations in the header for why
        // this is necessary (the shader is not SV_VertexID-procedural).
        // Layout matches FrameRender::SimpleVertex exactly (POSITION
        // float4 @0, TEXCOORD float2 @16, VIEW uint @24), validated
        // against our own copy of the same QuadVertexShader.cso bytes, so
        // this is guaranteed compatible with the shader we're binding.
        {
            D3D11_INPUT_ELEMENT_DESC layout[] = {
                { "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA,
                  0 },
                { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 16, D3D11_INPUT_PER_VERTEX_DATA, 0 },
                { "VIEW", 0, DXGI_FORMAT_R32_UINT, 0, 24, D3D11_INPUT_PER_VERTEX_DATA, 0 },
            };
            OK_OR_THROW(
                device->CreateInputLayout(
                    layout,
                    3,
                    quadShaderCSO.data(),
                    quadShaderCSO.size(),
                    &m_quadInputLayout
                ),
                "VideoEncoderJpegXs: failed to create quad input layout."
            );

            struct QuadVertex {
                float pos[4];
                float tex[2];
                uint32_t view;
            };
            // Plain full-screen NDC quad (z/w chosen so the shader's clip
            // divide is a no-op), triangle-strip winding matching
            // FrameRender's own convention (TL, BR, TR, BL). TEXCOORD is
            // Y-flipped relative to position (v=1 at the NDC top) to match
            // ALVR's own top-left-origin texture convention -- see
            // FrameRender.cpp's "Left View" vertex block for the same
            // pairing.
            QuadVertex vertices[4] = {
                { { -1.0f, 1.0f, 0.5f, 1.0f }, { 0.0f, 1.0f }, 0 }, // top-left
                { { 1.0f, -1.0f, 0.5f, 1.0f }, { 1.0f, 0.0f }, 0 }, // bottom-right
                { { 1.0f, 1.0f, 0.5f, 1.0f }, { 1.0f, 1.0f }, 0 }, // top-right
                { { -1.0f, -1.0f, 0.5f, 1.0f }, { 0.0f, 0.0f }, 0 }, // bottom-left
            };

            D3D11_BUFFER_DESC vbDesc = {};
            vbDesc.Usage = D3D11_USAGE_IMMUTABLE;
            vbDesc.ByteWidth = sizeof(vertices);
            vbDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
            D3D11_SUBRESOURCE_DATA vbData = {};
            vbData.pSysMem = vertices;
            OK_OR_THROW(
                device->CreateBuffer(&vbDesc, &vbData, &m_quadVertexBuffer),
                "VideoEncoderJpegXs: failed to create quad vertex buffer."
            );
        }

        struct YUVParams {
            float offset[4];
            float yCoeff[4];
            float uCoeff[4];
            float vCoeff[4];
        };

        // sourceTexture (FrameRender's non-HDR composition texture) is
        // created as fully-typed DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, not
        // TYPELESS -- so unlike a typeless resource, we cannot create an SRV
        // on it with a different (non-_SRGB) format to suppress the
        // automatic gamma decode; D3D11 rejects that (CreateShaderResourceView
        // fails outright). ALVR's own FrameRender.cpp hits the same
        // constraint for its color-correction pass and works around it in
        // shader code instead (see its "inputColorAdjust" logic) -- we do
        // the same here: detect the _SRGB formats host-side and pass a flag
        // into the shader (packed into offset.w, otherwise unused) that
        // tells it to re-encode (gamma-compress) the already-linearized
        // Sample() result before applying the BT.709 coefficients, which
        // expect gamma-encoded R'G'B' input, not scene-linear values.
        bool sourceNeedsSrgbEncode = false;
        {
            D3D11_TEXTURE2D_DESC sourceDesc;
            sourceTexture->GetDesc(&sourceDesc);
            sourceNeedsSrgbEncode = sourceDesc.Format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
                || sourceDesc.Format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB
                || sourceDesc.Format == DXGI_FORMAT_B8G8R8X8_UNORM_SRGB;
        }

        // BT.709, 8-bit, full range -- matches the color math the CPU-side
        // RGB->YUV conversion used before this GPU rewrite, for a
        // consistent look between runs. (ALVR's own HDR path uses BT.2020
        // instead, since it's converting a wide-gamut source; not
        // applicable here -- we always treat the source as ordinary SDR.)
        YUVParams paramStruct
            = { { 0.0000000f, 0.5000000f, 0.5000000f, sourceNeedsSrgbEncode ? 1.0f : 0.0f }, // offset
                { 0.2126000f, 0.7152000f, 0.0722000f, 0.0f }, // yCoeff
                { -0.1145721f, -0.3854279f, 0.5000000f, 0.0f }, // uCoeff
                { 0.5000000f, -0.4541529f, -0.0458471f, 0.0f } }; // vCoeff

        m_yuvParamBuffer = d3d_render_utils::CreateBuffer(device, paramStruct);

        // Our own minimal RGB->YUV420 pixel shaders, compiled at runtime via
        // D3DCompile (we already link d3dcompiler.lib for the shader
        // reflection diagnostic above) rather than reusing ALVR's
        // rgbtoyuv420.hlsl / RGBTOYUV420_CSO -- see the long comment on
        // m_yPixelShader in the header for why. No manual texel-index math:
        // each pass just samples the source at the interpolated `uv`
        // texcoord, which is correct as long as the bound viewport actually
        // matches the render target's real dimensions (see m_gpuViewport /
        // m_uvViewport below). The sampler is declared inline in the HLSL
        // itself (a "shader-literal" sampler, compiled directly into the
        // shader bytecode) -- same technique ALVR's own rgbtoyuv420.hlsl
        // uses, no host-side CreateSamplerState()/PSSetSamplers() call
        // needed.
        static const char* kYuvShaderSource = R"HLSL(
            cbuffer YUVParams {
                float4 offset;
                float4 yCoeff;
                float4 uCoeff;
                float4 vCoeff;
            };

            Texture2D<float4> sourceTexture;

            SamplerState bilinearSampler {
                Filter = MIN_MAG_LINEAR_MIP_POINT;
                AddressU = CLAMP;
                AddressV = CLAMP;
            };

            // offset.w != 0 means sourceTexture is bound through an _SRGB
            // SRV, so Sample() already linearized it -- re-encode
            // (gamma-compress) back to gamma space before applying the
            // BT.709 coefficients below, which expect gamma-encoded R'G'B',
            // not scene-linear light. Standard IEC 61966-2-1 sRGB transfer
            // function, branchless (offset.w is uniform for the whole draw
            // call, so this costs nothing on real hardware).
            float3 reencodeSrgbIfNeeded(float3 rgb) {
                float3 lo = rgb * 12.92;
                float3 hi = 1.055 * pow(max(rgb, 0.0), 1.0 / 2.4) - 0.055;
                float3 encoded = lerp(lo, hi, step(0.0031308, rgb));
                return offset.w > 0.5 ? encoded : rgb;
            }

            float4 mainY(float2 uv : TEXCOORD0) : SV_Target0 {
                float3 rgb = reencodeSrgbIfNeeded(sourceTexture.Sample(bilinearSampler, uv).rgb);
                float y = dot(rgb, yCoeff.rgb) + offset.x;
                return float4(y, 0, 0, 1);
            }

            struct UVOutput {
                float cb : SV_Target0;
                float cr : SV_Target1;
            };

            // Two separate scalar outputs (one per render target) instead of
            // packing both into one float4 -- see m_gpuCbTex/m_gpuCrTex in
            // the header for why: this writes Cb/Cr directly as the two
            // separate planes SVT-JPEG-XS wants, no CPU-side deinterleave.
            UVOutput mainUV(float2 uv : TEXCOORD0) {
                float3 rgb = reencodeSrgbIfNeeded(sourceTexture.Sample(bilinearSampler, uv).rgb);
                UVOutput result;
                result.cb = dot(rgb, uCoeff.rgb) + offset.y;
                result.cr = dot(rgb, vCoeff.rgb) + offset.z;
                return result;
            }
        )HLSL";

        auto compileYuvShader = [&](const char* entryPoint, ID3D11PixelShader** outShader) {
            Microsoft::WRL::ComPtr<ID3DBlob> bytecode;
            Microsoft::WRL::ComPtr<ID3DBlob> errors;
            HRESULT hr = D3DCompile(
                kYuvShaderSource,
                strlen(kYuvShaderSource),
                "VideoEncoderJpegXs_yuv.hlsl",
                nullptr,
                nullptr,
                entryPoint,
                "ps_5_0",
                0,
                0,
                bytecode.GetAddressOf(),
                errors.GetAddressOf()
            );
            if (FAILED(hr)) {
                std::string errorText = errors
                    ? std::string(
                          reinterpret_cast<const char*>(errors->GetBufferPointer()),
                          errors->GetBufferSize()
                      )
                    : "(no error blob)";
                throw std::runtime_error(
                    "VideoEncoderJpegXs: D3DCompile failed for entry point '" + std::string(entryPoint)
                    + "': " + errorText
                );
            }
            OK_OR_THROW(
                device->CreatePixelShader(
                    bytecode->GetBufferPointer(), bytecode->GetBufferSize(), nullptr, outShader
                ),
                "VideoEncoderJpegXs: failed to create YUV pixel shader."
            );
        };
        compileYuvShader("mainY", m_yPixelShader.GetAddressOf());
        compileYuvShader("mainUV", m_uvPixelShader.GetAddressOf());

        // Two separate, ordinary render targets -- not a shared NV12
        // resource. See the top-of-file comment for why: relying on a
        // pixel shader writing directly into a planar NV12 texture's two
        // plane-specific views turned out to silently no-op on at least one
        // real-world GPU/driver, despite every D3D11 call reporting success.
        // Plain R8_UNORM/R8G8_UNORM 2D textures have no such ambiguity.
        m_gpuYTex = d3d_render_utils::CreateTexture(device, m_width, m_height, DXGI_FORMAT_R8_UNORM);
        m_gpuCbTex = d3d_render_utils::CreateTexture(
            device, m_chromaWidth, m_chromaHeight, DXGI_FORMAT_R8_UNORM
        );
        m_gpuCrTex = d3d_render_utils::CreateTexture(
            device, m_chromaWidth, m_chromaHeight, DXGI_FORMAT_R8_UNORM
        );

        // Plain nullptr desc -- inherits sourceTexture's native format. We
        // tried overriding the format here to strip _SRGB and avoid the
        // automatic gamma decode, but that fails outright: the composition
        // texture is created as fully-typed DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
        // (FrameRender::Startup()), not TYPELESS, so D3D11 does not permit a
        // differently-formatted view of it at all. Instead, the gamma
        // decode is compensated for in the pixel shader -- see
        // sourceNeedsSrgbEncode / offset.w above and
        // reencodeSrgbIfNeeded() in the shader source.
        OK_OR_THROW(
            device->CreateShaderResourceView(sourceTexture, nullptr, &m_sourceSRV),
            "VideoEncoderJpegXs: failed to create source texture SRV."
        );

        OK_OR_THROW(
            device->CreateRenderTargetView(m_gpuYTex.Get(), nullptr, &m_rtvY),
            "VideoEncoderJpegXs: failed to create Y render target view."
        );

        OK_OR_THROW(
            device->CreateRenderTargetView(m_gpuCbTex.Get(), nullptr, &m_rtvCb),
            "VideoEncoderJpegXs: failed to create Cb render target view."
        );

        OK_OR_THROW(
            device->CreateRenderTargetView(m_gpuCrTex.Get(), nullptr, &m_rtvCr),
            "VideoEncoderJpegXs: failed to create Cr render target view."
        );

        m_gpuViewport = {};
        m_gpuViewport.Width = (float)m_width;
        m_gpuViewport.Height = (float)m_height;
        m_gpuViewport.MinDepth = 0.0f;
        m_gpuViewport.MaxDepth = 1.0f;
        m_gpuViewport.TopLeftX = 0.0f;
        m_gpuViewport.TopLeftY = 0.0f;

        // Genuinely half-size viewport for the UV pass -- matches
        // m_gpuCbTex/m_gpuCrTex's real dimensions exactly. See the comment on
        // m_uvViewport in the header for why this (rather than reusing
        // m_gpuViewport for both passes) is the actual fix for the chroma
        // corruption.
        m_uvViewport = {};
        m_uvViewport.Width = (float)m_chromaWidth;
        m_uvViewport.Height = (float)m_chromaHeight;
        m_uvViewport.MinDepth = 0.0f;
        m_uvViewport.MaxDepth = 1.0f;
        m_uvViewport.TopLeftX = 0.0f;
        m_uvViewport.TopLeftY = 0.0f;

        // Explicit, known-good state -- see the member declarations' comment
        // for why this is needed (we don't control what state the shared
        // context was left in before our draws run).
        D3D11_RASTERIZER_DESC rasterDesc = {};
        rasterDesc.FillMode = D3D11_FILL_SOLID;
        rasterDesc.CullMode = D3D11_CULL_NONE;
        rasterDesc.FrontCounterClockwise = FALSE;
        rasterDesc.DepthClipEnable = TRUE;
        rasterDesc.ScissorEnable = FALSE;
        OK_OR_THROW(
            device->CreateRasterizerState(&rasterDesc, &m_rasterizerState),
            "VideoEncoderJpegXs: failed to create rasterizer state."
        );

        D3D11_BLEND_DESC blendDesc = {};
        blendDesc.RenderTarget[0].BlendEnable = FALSE;
        blendDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        OK_OR_THROW(
            device->CreateBlendState(&blendDesc, &m_blendState),
            "VideoEncoderJpegXs: failed to create blend state."
        );

        D3D11_DEPTH_STENCIL_DESC depthDesc = {};
        depthDesc.DepthEnable = FALSE;
        depthDesc.StencilEnable = FALSE;
        OK_OR_THROW(
            device->CreateDepthStencilState(&depthDesc, &m_depthStencilState),
            "VideoEncoderJpegXs: failed to create depth-stencil state."
        );

        // TEMPORARY DIAGNOSTIC: fully synchronous smoke test -- Clear,
        // CopyResource, Flush, Map, all on this one thread, right here,
        // completely decoupled from the async submit/readback pipeline
        // (which spans three threads: whatever calls Transmit,
        // ReadbackWorkerLoop, and FrameRender's own rendering thread). Every
        // previous test (real render, then an explicit clear-before-draw)
        // came back as all zeros even though Map() itself reports success.
        // If this synchronous test ALSO reads back zero, the bug is in
        // basic resource/view identity or CopyResource, nothing to do with
        // threading. If this succeeds, the bug is specific to the async
        // multi-threaded path. Reuses m_stagingYTex[0], which Initialize()
        // (called before any Transmit/EnsureGpuPipeline call) has
        // already created. Remove once the green issue is confirmed fixed.
        {
            float testColor[4] = { 222.0f / 255.0f, 0.0f, 0.0f, 1.0f };
            context->ClearRenderTargetView(m_rtvY.Get(), testColor);
            context->CopyResource(m_stagingYTex[0].Get(), m_gpuYTex.Get());
            context->Flush();
            D3D11_MAPPED_SUBRESOURCE mapped;
            HRESULT hr = context->Map(m_stagingYTex[0].Get(), 0, D3D11_MAP_READ, 0, &mapped);
            if (SUCCEEDED(hr)) {
                uint8_t firstByte = *reinterpret_cast<const uint8_t*>(mapped.pData);
                Info(
                    "VideoEncoderJpegXs: SYNC SMOKE TEST: cleared Y to 222, synchronous "
                    "readback first byte = %u (RowPitch=%u)\n",
                    (unsigned)firstByte,
                    (unsigned)mapped.RowPitch
                );
                context->Unmap(m_stagingYTex[0].Get(), 0);
            } else {
                Info(
                    "VideoEncoderJpegXs: SYNC SMOKE TEST: Map() failed hr=0x%08lx\n",
                    (unsigned long)hr
                );
            }
        }

        m_gpuPipelineReady = true;
        Info(
            "VideoEncoderJpegXs: GPU RGB->YUV420 pipeline ready (%ux%u Y + %ux%u UV, separate "
            "plain render targets, own runtime-compiled shaders, own two-pass Y/UV renderer, "
            "correctly-sized viewport per pass).\n",
            m_width,
            m_height,
            m_chromaWidth,
            m_chromaHeight
        );
        return true;
    } catch (const std::exception& e) {
        Error(
            "VideoEncoderJpegXs: GPU pipeline setup failed, disabling JPEG XS test dump: %s\n",
            e.what()
        );
        m_gpuSetupFailed = true;
        return false;
    }
}

void VideoEncoderJpegXs::RenderYuvGpu(ID3D11DeviceContext* context) {
    // Pin down rasterizer/blend/depth state explicitly -- see the member
    // declarations' comment in the header. Without this, whatever state the
    // real path (or CEncoder/VideoEncoder submission in between) left the
    // shared context in applies to our draws too; a leftover scissor rect,
    // non-default cull mode, or restricted color write mask can silently
    // discard every pixel we try to write, leaving the output texture at its
    // all-zero creation-time content (which decodes to solid green).
    context->RSSetState(m_rasterizerState.Get());
    context->OMSetBlendState(m_blendState.Get(), nullptr, 0xffffffff);
    context->OMSetDepthStencilState(m_depthStencilState.Get(), 0);

    // Bind our own input layout + vertex buffer -- see the long comment on
    // the member declarations in the header. UPDATE: shader reflection
    // (D3DReflect) now proves QuadVertexShader.cso has exactly one input
    // parameter, SV_VertexID -- it *is* fully procedural after all, no
    // vertex buffer needed. Keeping this bound anyway is harmless (it's
    // simply unused by the shader), so leaving it in rather than ripping it
    // back out; just noting the earlier theory (FrameRender-inherited
    // vertex buffer) is now known to be wrong.
    context->IASetInputLayout(m_quadInputLayout.Get());
    ID3D11Buffer* vertexBuffers[] = { m_quadVertexBuffer.Get() };
    UINT stride = sizeof(float) * 4 + sizeof(float) * 2 + sizeof(uint32_t); // 28 bytes
    UINT offset = 0;
    context->IASetVertexBuffers(0, 1, vertexBuffers, &stride, &offset);
    context->IASetIndexBuffer(nullptr, DXGI_FORMAT_UNKNOWN, 0);

    context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    context->VSSetShader(m_quadVertexShader.Get(), nullptr, 0);
    context->PSSetConstantBuffers(0, 1, m_yuvParamBuffer.GetAddressOf());

    // Unbind ALL render targets before binding sourceTexture as an SRV. The
    // real cause of the green output: sourceTexture is ALVR's own compositor
    // output texture, which the main render path (FrameRender::RenderFrame)
    // leaves bound as an active render target on this same immediate context
    // -- it's never had a reason to unbind it before now, since nothing else
    // previously tried to read it as a shader input. D3D11's resource-hazard
    // tracker silently forces a shader-resource slot back to nullptr if the
    // same resource is simultaneously bound as a render target anywhere on
    // the context (no error, no exception -- this is why every prior
    // diagnostic came back clean). That's exactly what the pre-Draw
    // diagnostic below has been showing: srv=MISMATCH on every single frame,
    // everything else MATCH. Clearing render targets first breaks the
    // hazard.
    ID3D11RenderTargetView* clearRtv[] = { nullptr };
    context->OMSetRenderTargets(1, clearRtv, nullptr);

    ID3D11ShaderResourceView* srvs[] = { m_sourceSRV.Get() };
    context->PSSetShaderResources(0, 1, srvs);

    // Two single-target passes instead of one dual-target pass -- see the
    // long comment on the declaration of this method for why. Each pass now
    // uses its own pixel shader (mainY / mainUV) and its own, genuinely
    // correctly-sized viewport (m_gpuViewport for the full-res Y target,
    // m_uvViewport for the half-res UV target) -- see the comment on
    // m_yPixelShader in the header for why sharing one oversized viewport
    // between both passes was the actual bug.
    context->PSSetShader(m_yPixelShader.Get(), nullptr, 0);
    context->RSSetViewports(1, &m_gpuViewport);
    ID3D11RenderTargetView* rtvY[] = { m_rtvY.Get() };
    context->OMSetRenderTargets(1, rtvY, nullptr);
    context->Draw(4, 0);

    // MRT: mainUV writes Cb to SV_Target0 and Cr to SV_Target1 in one draw
    // call -- see UVOutput in the shader source above.
    context->PSSetShader(m_uvPixelShader.Get(), nullptr, 0);
    context->RSSetViewports(1, &m_uvViewport);
    ID3D11RenderTargetView* rtvUV[] = { m_rtvCb.Get(), m_rtvCr.Get() };
    context->OMSetRenderTargets(2, rtvUV, nullptr);
    context->Draw(4, 0);

    // Leave render targets unbound afterwards rather than pointing at our
    // own textures -- the real path sets its own targets before it next
    // needs them, but there's no reason to leave stale state bound in the
    // meantime. Same reasoning for restoring rasterizer/blend/depth state to
    // D3D11 defaults (nullptr = default state) rather than leaving our own
    // (e.g. CullMode=NONE) bound -- non-invasive in both directions.
    // NumViews=2 here, not 1 -- the UV pass above bound two RTVs (slots 0
    // and 1); a NumViews=1 call would only clear slot 0 and leave m_rtvCr
    // (slot 1) still bound.
    ID3D11RenderTargetView* none[] = { nullptr, nullptr };
    context->OMSetRenderTargets(2, none, nullptr);
    context->RSSetState(nullptr);
    context->OMSetBlendState(nullptr, nullptr, 0xffffffff);
    context->OMSetDepthStencilState(nullptr, 0);
}

// See the member declarations' doc comment (VideoEncoderJpegXs.h) for why this exists.
// Cheap in the steady-state case (just a mutex + a few counter increments); the actual
// file I/O only happens once every ~1s window, from whichever thread happens to cross
// that boundary first.
void VideoEncoderJpegXs::DiagTick(const char* stage, double durationMs, bool skipped) {
    std::lock_guard<std::mutex> lock(m_diagMutex);
    auto now = std::chrono::steady_clock::now();
    if (m_diagWindowStart.time_since_epoch().count() == 0) {
        m_diagWindowStart = now;
    }
    if (std::strcmp(stage, "transmit") == 0) {
        m_diagTransmitCalls++;
        if (skipped)
            m_diagTransmitSkips++;
    } else if (std::strcmp(stage, "readback") == 0) {
        m_diagReadbackCount++;
        m_diagReadbackMsTotal += durationMs;
    } else if (std::strcmp(stage, "encode") == 0) {
        m_diagEncodeCount++;
        m_diagEncodeMsTotal += durationMs;
    } else if (std::strcmp(stage, "full_span") == 0) {
        m_diagFullSpanCount++;
        m_diagFullSpanMsTotal += durationMs;
        m_diagFullSpanMsMax = std::max(m_diagFullSpanMsMax, durationMs);
    }
    double elapsedS = std::chrono::duration<double>(now - m_diagWindowStart).count();
    if (elapsedS >= 1.0) {
        FILE* f = fopen("C:\\Temp\\jpegxs_server_diag.log", "a");
        if (f) {
            fprintf(
                f,
                "[%s][%.3fs window] transmit: calls=%d skips=%d (%.0f%%) -> ~%.1f accepted/s | "
                "readback: n=%d avg=%.2fms | encode: n=%d avg=%.2fms | "
                "FULL SPAN (Transmit->VideoSend, incl. thread-wakeup waits): n=%d avg=%.2fms max=%.2fms\n",
                m_sideChannelOnly ? "SIDE-CHANNEL" : "PRIMARY",
                elapsedS,
                m_diagTransmitCalls,
                m_diagTransmitSkips,
                m_diagTransmitCalls > 0 ? 100.0 * m_diagTransmitSkips / m_diagTransmitCalls : 0.0,
                (m_diagTransmitCalls - m_diagTransmitSkips) / elapsedS,
                m_diagReadbackCount,
                m_diagReadbackCount > 0 ? m_diagReadbackMsTotal / m_diagReadbackCount : 0.0,
                m_diagEncodeCount,
                m_diagEncodeCount > 0 ? m_diagEncodeMsTotal / m_diagEncodeCount : 0.0,
                m_diagFullSpanCount,
                m_diagFullSpanCount > 0 ? m_diagFullSpanMsTotal / m_diagFullSpanCount : 0.0,
                m_diagFullSpanMsMax
            );
            // Per-column collect timing, written by the encode thread. "waited" near zero on a
            // column means it was finished before its turn came round in the index-ordered,
            // blocking Phase 2 drain -- its packets sat in the encoder instead of going out.
            {
                std::lock_guard<std::mutex> lock(m_columnDiagMutex);
                if (!m_columnDiagLine.empty()) {
                    fprintf(f, "    columns: %s\n", m_columnDiagLine.c_str());
                }
            }
            fclose(f);
        }
        m_diagWindowStart = now;
        m_diagTransmitCalls = 0;
        m_diagTransmitSkips = 0;
        m_diagReadbackCount = 0;
        m_diagReadbackMsTotal = 0.0;
        m_diagEncodeCount = 0;
        m_diagEncodeMsTotal = 0.0;
        m_diagFullSpanCount = 0;
        m_diagFullSpanMsTotal = 0.0;
        m_diagFullSpanMsMax = 0.0;
    }
}

void VideoEncoderJpegXs::Transmit(
    ID3D11Texture2D* sourceTexture,
    uint64_t /*presentationTime*/,
    uint64_t targetTimestampNs,
    bool /*insertIDR*/
) {
    if (!m_encInitialized || sourceTexture == nullptr) {
        return;
    }
    if (!EnsureGpuPipeline(sourceTexture)) {
        return; // GPU pipeline unavailable; skip this (and all future) frames.
    }

    auto t0 = std::chrono::steady_clock::now();
    auto* context = m_d3dRender->GetContext();

    int writeIndex = m_stagingWriteIndex;
    // Both paths hand off the slot written by THIS call.
    //
    // The readback path used to hand off the one written by the PREVIOUS call, to
    // give the GPU a full frame period to finish the copy before
    // ReadbackWorkerLoop Map()s it. That cost every frame 11.1ms at 90fps: measured
    // with the CPU/GPU hybrid (2026-09-11), encode 5.0ms but FULL SPAN 16.7ms, ALVR
    // encoder_latency 16.5ms against 7.9ms on the D3D11 path. ReadbackWorkerLoop now
    // waits for the copy itself, with MapWhenReady(), which never holds the immediate
    // context while the GPU is still busy -- so the wait costs only as long as the
    // copy actually takes.
    //
    // On the D3D11 device-input path that wait is pure cost. It exists because
    // ReadbackWorkerLoop Map()s the staging texture from the CPU, and a Map() on a
    // copy the GPU has not finished stalls that thread. Here nobody maps anything:
    // the consumer is CUDA reading the texture in place, and the ordering between
    // our CopyResource and the encoder's kernels is enforced by the graphics-interop
    // map inside svt_jpeg_xs_d3d11_plane_to_device() -- which this very call already
    // performed for writeIndex, right after its own Flush(), so the device pointer
    // for THIS frame is valid by the time we reach the hand-off below.
    //
    // Measured cost of the delay: the diagnostic FULL SPAN (Transmit->VideoSend) ran
    // 18.5ms while the encode stage itself took 5.7ms and readback 0.00ms. One frame
    // period at 90fps is 11.1ms, and 11.1 + 5.7 + ~1.7ms of thread wakeup is exactly
    // the 18.5ms observed. It is the single largest term in ALVR's encoder_latency.
    int readIndex = writeIndex;

    // If the slot we're about to overwrite is still reserved (readback
    // hasn't Map()'d it yet), skip this frame entirely rather than
    // overwriting GPU data mid-readback. With the mailbox limiting us to one
    // outstanding pending frame, this should be rare in steady state -- it
    // means readback has fallen behind.
    bool writeSlotReserved;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        writeSlotReserved = m_stagingReserved[writeIndex];
    }
    if (writeSlotReserved) {
        auto t1 = std::chrono::steady_clock::now();
        LogPeriod(
            "VideoEncoderJpegXs.Submit",
            "submit skipped, staging slot %d still reserved by readback = %.2fms",
            writeIndex,
            std::chrono::duration<double, std::milli>(t1 - t0).count()
        );
        DiagTick("transmit", 0.0, /*skipped=*/true);
        return;
    }

    // GPU RGB->YUV420 conversion: a couple of cheap fullscreen-quad draw
    // calls (reusing ALVR's own already-compiled rgbtoyuv420.hlsl pipeline)
    // into our own m_gpuYTex/m_gpuCbTex/m_gpuCrTex, followed by three
    // CopyResource calls into this frame's staging slot. The actual GPU
    // *wait* (Map()) still
    // happens on ReadbackWorkerLoop, never here -- but the synchronous
    // smoke test in EnsureGpuPipeline() (Clear+Copy+Flush+Map, all on this
    // same thread) proved the underlying mechanism works, while the exact
    // same sequence *without* Flush(), read back later from a *different*
    // thread (ReadbackWorkerLoop), consistently came back all-zero. Without
    // an explicit Flush(), the driver is free to leave these commands
    // sitting recorded-but-unsubmitted indefinitely; Map()'s implicit
    // wait-for-GPU is apparently not enough to guarantee that on its own
    // when the Map() happens on a different thread than the one that
    // recorded the copy. Flush() here forces submission right away, so by
    // the time ReadbackWorkerLoop's Map() call runs -- right after the hand-off
    // below, polling until the copy is done (MapWhenReady) -- the GPU has
    // unambiguously already started the work.
    RenderYuvGpu(context);
    // The per-slot copies exist for the readback path, whose worker Map()s a slot while the
    // next frame is already being rendered. On the D3D11-input path nobody maps them: the
    // handover below copies each plane into the slot's own linear CUDA buffer (scratch slot
    // writeIndex * 3 + p) and waits for that copy before returning, so by the time the next
    // Transmit renders into m_gpuYTex again, this frame's pixels are already in CUDA memory.
    // The slot textures were a second full-frame copy on the game's GPU -- ~10.7 MB at
    // 4288x1664, every frame -- that nothing ever read (2026-09-24).
    // KEPT ON THE D3D11 PATH FOR NOW (2026-09-24). Skipping them there really does save the
    // ~10.7 MB, and nothing reads the slots on that path -- but they are also work on the
    // immediate context between the YUV render and the CUDA handover, so they order the two.
    // They go out again only together with an explicit ordering (a flush, or a fence the
    // handover waits on). Left in while the trailing that came with this commit is unexplained.
    context->CopyResource(m_stagingYTex[writeIndex].Get(), m_gpuYTex.Get());
    context->CopyResource(m_stagingCbTex[writeIndex].Get(), m_gpuCbTex.Get());
    context->CopyResource(m_stagingCrTex[writeIndex].Get(), m_gpuCrTex.Get());
    if (m_hybridZeroCopy) {
        // The CPU columns' part of the picture, and only that, into the narrow CPU-readable
        // textures. The GPU columns never leave the device. The source box is the left
        // m_cpuReadbackWidth pixels; D3D11 wants a half-open box, back inclusive-exclusive in z.
        D3D11_BOX yBox = { 0, 0, 0, m_cpuReadbackWidth, m_height, 1 };
        D3D11_BOX chromaBox = { 0, 0, 0, m_cpuReadbackChromaWidth, m_chromaHeight, 1 };
        context->CopySubresourceRegion(
            m_cpuStagingYTex[writeIndex].Get(), 0, 0, 0, 0, m_gpuYTex.Get(), 0, &yBox
        );
        context->CopySubresourceRegion(
            m_cpuStagingCbTex[writeIndex].Get(), 0, 0, 0, 0, m_gpuCbTex.Get(), 0, &chromaBox
        );
        context->CopySubresourceRegion(
            m_cpuStagingCrTex[writeIndex].Get(), 0, 0, 0, 0, m_gpuCrTex.Get(), 0, &chromaBox
        );
    }
    context->Flush();

    // The CUDA handover, on this thread, right after the flush that makes the copies
    // real. See m_devicePlane in the header for why it cannot happen on the encode
    // thread. Every plane of this slot must succeed or the slot carries nothing --
    // there is deliberately no CPU fallback on this path, so a frame we cannot hand
    // over is a frame that gets dropped, not one encoded from stale planes.
    if (m_useD3d11Input) {
        // The render targets themselves, see the note at RenderYuvGpu above; the slot is the
        // CUDA scratch buffer the handover copies into.
        ID3D11Texture2D* planes[3] = { m_gpuYTex.Get(), m_gpuCbTex.Get(), m_gpuCrTex.Get() };
        const uint32_t planeW[3] = { m_width, m_chromaWidth, m_chromaWidth };
        const uint32_t planeH[3] = { m_height, m_chromaHeight, m_chromaHeight };
        bool allMapped = true;
        for (int p = 0; p < 3; p++) {
            void* dev = nullptr;
            uint32_t pitch = 0;
            if (svt_jpeg_xs_d3d11_plane_to_device(
                    planes[p], planeW[p], planeH[p], 0, writeIndex * 3 + p, &dev, &pitch
                )) {
                m_devicePlane[writeIndex][p] = dev;
                m_devicePitch[writeIndex][p] = pitch;
            } else {
                allMapped = false;
                LogPeriod(
                    "VideoEncoderJpegXs.D3D11",
                    "handover failed for plane %d; this frame will be dropped\n",
                    p
                );
                break;
            }
        }
        m_deviceSlotValid[writeIndex] = allMapped;
    }

    m_stagingTimestampNs[writeIndex] = targetTimestampNs;
    m_stagingSubmitInstant[writeIndex] = t0;
    m_stagingHasData[writeIndex] = true;

    if (readIndex >= 0 && m_stagingHasData[readIndex]) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_hasPendingFrame) {
            m_pendingStagingIndex = readIndex;
            m_pendingTimestampNs = m_stagingTimestampNs[readIndex];
            m_pendingSubmitInstant = m_stagingSubmitInstant[readIndex];
            m_hasPendingFrame = true;
            m_stagingReserved[readIndex] = true;
            m_cv.notify_one();
        }
    }

    m_stagingPrevIndex = writeIndex;
    m_stagingWriteIndex = (writeIndex + 1) % m_stagingSlots;

    auto t1 = std::chrono::steady_clock::now();
    LogPeriod(
        "VideoEncoderJpegXs.Submit",
        "submit (GPU render + CopyResource enqueue, no GPU wait here) = %.2fms",
        std::chrono::duration<double, std::milli>(t1 - t0).count()
    );
    DiagTick("transmit", 0.0, /*skipped=*/false);
}

// Map() a staging texture for reading once the GPU has finished writing it, without
// blocking inside the immediate context while it has not. A plain Map() on an
// unfinished copy waits inside the call, i.e. while holding the context that the
// compositor thread (Transmit) also drives -- and a mutual wait on that context has
// already hung the device once in this encoder. D3D11_MAP_FLAG_DO_NOT_WAIT returns
// DXGI_ERROR_WAS_STILL_DRAWING immediately instead, so the wait happens out here,
// with the context free. Needed since Transmit hands off the slot it has only just
// filled (see readIndex there). After 50ms it falls back to the blocking Map() so a
// stuck GPU can never lose the frame silently.
static HRESULT MapWhenReady(ID3D11DeviceContext* context, ID3D11Resource* resource, D3D11_MAPPED_SUBRESOURCE* mapped) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(50);
    for (;;) {
        HRESULT hr = context->Map(resource, 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, mapped);
        if (hr != DXGI_ERROR_WAS_STILL_DRAWING) {
            return hr;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            return context->Map(resource, 0, D3D11_MAP_READ, 0, mapped);
        }
        std::this_thread::yield();
    }
}

void VideoEncoderJpegXs::ReadbackWorkerLoop() {
    while (true) {
        uint64_t timestampNs = 0;
        int stagingIndex = -1;
        std::chrono::steady_clock::time_point submitInstant {};

        // Stage 1 input: dequeue the staging-slot index Transmit handed
        // off (no pixel data crosses this mailbox -- see header comment).
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv.wait(lock, [this] {
                return m_hasPendingFrame || m_exiting.load(std::memory_order_acquire);
            });
            if (!m_hasPendingFrame && m_exiting.load(std::memory_order_acquire)) {
                break;
            }
            stagingIndex = m_pendingStagingIndex;
            timestampNs = m_pendingTimestampNs;
            submitInstant = m_pendingSubmitInstant;
            m_hasPendingFrame = false;
        }

        auto tReadback0 = std::chrono::steady_clock::now();

        // GPU readback: Map() blocks until the earlier render+CopyResource
        // have landed. Three separate, ordinary staging textures now (not
        // one NV12 texture's subresources -- see the top-of-file comment),
        // all mapped at subresource 0. Cb/Cr come off the GPU as already-
        // separate planes (see m_gpuCbTex/m_gpuCrTex), so this is a plain
        // per-row memcpy for all three planes -- no CPU-side deinterleaving
        // left anywhere in this path. Landing in our own private scratch
        // buffers here (not directly in the yuv mailbox) lets us release the
        // GPU staging slot immediately below, instead of holding it reserved
        // for however long encode takes to free up the next mailbox slot.
        // Nothing to read back on the D3D11-input path: the encoder maps these
        // textures through CUDA itself, so this stage only forwards the slot. It
        // stays a stage of its own rather than being bypassed, because it is also
        // where the one-frame-of-slack handoff to encode lives.
        if (!m_useD3d11Input) {
            auto* context = m_d3dRender->GetContext();

            D3D11_MAPPED_SUBRESOURCE mappedY;
            HRESULT hrY = MapWhenReady(context, m_stagingYTex[stagingIndex].Get(), &mappedY);
            if (SUCCEEDED(hrY)) {
                const uint8_t* src = reinterpret_cast<const uint8_t*>(mappedY.pData);
                for (uint32_t y = 0; y < m_height; y++) {
                    memcpy(
                        m_readbackScratchY.data() + (size_t)y * m_width,
                        src + (size_t)y * mappedY.RowPitch,
                        m_width
                    );
                }
                context->Unmap(m_stagingYTex[stagingIndex].Get(), 0);
            } else {
                // TEMPORARY DIAGNOSTIC: this was previously failing silently
                // -- if Map() fails, the scratch buffer (zero-initialized at
                // Initialize() time) never gets touched again, which reads
                // as a permanent, deterministic all-zero plane forever. Log
                // the actual HRESULT plus device-removed status so we can
                // tell a Map() failure apart from "Map() succeeded but the
                // GPU genuinely produced zero data".
                HRESULT removedReason = m_d3dRender->GetDevice()->GetDeviceRemovedReason();
                LogPeriod(
                    "VideoEncoderJpegXs.Diag",
                    "Y staging Map() FAILED hr=0x%08lx, GetDeviceRemovedReason=0x%08lx",
                    (unsigned long)hrY,
                    (unsigned long)removedReason
                );
            }

            // Plain planar copies, no deinterleaving -- Cb/Cr come off the
            // GPU as two separate R8_UNORM render targets now (see
            // m_gpuCbTex/m_gpuCrTex), the same shape SVT-JPEG-XS wants
            // directly. This replaces the old single-Map R8G8 readback that
            // unpacked each row byte-by-byte (rowCb[x]=rowSrc[x*2+0], etc.)
            // into these same scratch buffers.
            D3D11_MAPPED_SUBRESOURCE mappedCb;
            HRESULT hrCb = MapWhenReady(context, m_stagingCbTex[stagingIndex].Get(), &mappedCb);
            if (SUCCEEDED(hrCb)) {
                const uint8_t* src = reinterpret_cast<const uint8_t*>(mappedCb.pData);
                for (uint32_t y = 0; y < m_chromaHeight; y++) {
                    memcpy(
                        m_readbackScratchCb.data() + (size_t)y * m_chromaWidth,
                        src + (size_t)y * mappedCb.RowPitch,
                        m_chromaWidth
                    );
                }
                context->Unmap(m_stagingCbTex[stagingIndex].Get(), 0);
            } else {
                LogPeriod(
                    "VideoEncoderJpegXs.Diag", "Cb staging Map() FAILED hr=0x%08lx", (unsigned long)hrCb
                );
            }

            D3D11_MAPPED_SUBRESOURCE mappedCr;
            HRESULT hrCr = MapWhenReady(context, m_stagingCrTex[stagingIndex].Get(), &mappedCr);
            if (SUCCEEDED(hrCr)) {
                const uint8_t* src = reinterpret_cast<const uint8_t*>(mappedCr.pData);
                for (uint32_t y = 0; y < m_chromaHeight; y++) {
                    memcpy(
                        m_readbackScratchCr.data() + (size_t)y * m_chromaWidth,
                        src + (size_t)y * mappedCr.RowPitch,
                        m_chromaWidth
                    );
                }
                context->Unmap(m_stagingCrTex[stagingIndex].Get(), 0);
            } else {
                LogPeriod(
                    "VideoEncoderJpegXs.Diag", "Cr staging Map() FAILED hr=0x%08lx", (unsigned long)hrCr
                );
            }
        } else if (m_hybridZeroCopy) {
            // Same readback, but only the CPU columns' part of the picture, out of the narrow
            // textures Transmit() cropped into. The scratch planes are left at their full
            // allocation and filled COMPACTLY at the narrow width -- the encode loop hands the
            // CPU columns a matching stride, so nothing downstream has to know about the crop
            // beyond that one number.
            auto* context = m_d3dRender->GetContext();
            struct {
                ID3D11Texture2D* tex;
                uint8_t* dst;
                uint32_t width;
                uint32_t height;
                const char* name;
            } planes[3] = {
                { m_cpuStagingYTex[stagingIndex].Get(),
                  m_readbackScratchY.data(),
                  m_cpuReadbackWidth,
                  m_height,
                  "Y" },
                { m_cpuStagingCbTex[stagingIndex].Get(),
                  m_readbackScratchCb.data(),
                  m_cpuReadbackChromaWidth,
                  m_chromaHeight,
                  "Cb" },
                { m_cpuStagingCrTex[stagingIndex].Get(),
                  m_readbackScratchCr.data(),
                  m_cpuReadbackChromaWidth,
                  m_chromaHeight,
                  "Cr" },
            };
            for (auto& p : planes) {
                D3D11_MAPPED_SUBRESOURCE mapped;
                HRESULT hr = MapWhenReady(context, p.tex, &mapped);
                if (SUCCEEDED(hr)) {
                    const uint8_t* src = reinterpret_cast<const uint8_t*>(mapped.pData);
                    for (uint32_t y = 0; y < p.height; y++) {
                        memcpy(
                            p.dst + (size_t)y * p.width, src + (size_t)y * mapped.RowPitch, p.width
                        );
                    }
                    context->Unmap(p.tex, 0);
                } else {
                    LogPeriod(
                        "VideoEncoderJpegXs.Diag",
                        "CPU-column %s staging Map() FAILED hr=0x%08lx",
                        p.name,
                        (unsigned long)hr
                    );
                }
            }
        }
        // Only the readback path is finished with the slot here. On the D3D11-input
        // path the pixels are still in that texture and the encoder has not read
        // them yet, so it is released after the encode instead -- see
        // EncodeWorkerLoop. Releasing it here would let Transmit CopyResource a
        // newer frame over the one being encoded.
        if (!m_useD3d11Input) {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stagingReserved[stagingIndex] = false;
        }

        auto tReadback1 = std::chrono::steady_clock::now();
        double readbackMs = std::chrono::duration<double, std::milli>(tReadback1 - tReadback0).count();
        LogPeriod(
            "VideoEncoderJpegXs.Readback",
            "readback (Map Y+Cb+Cr, own thread) = %.2fms",
            readbackMs
        );
        DiagTick("readback", readbackMs);

        // Handoff to encode: wait until it has picked up whatever we gave it
        // last time, so we never get more than one frame ahead of it. This
        // is what lets readback(N+1) genuinely overlap with encode(N)
        // instead of racing ahead unboundedly.
        {
            std::unique_lock<std::mutex> lock(m_yuvMutex);
            m_yuvCv.wait(lock, [this] {
                return !m_yuvReady || m_exiting.load(std::memory_order_acquire);
            });
            if (m_exiting.load(std::memory_order_acquire)) {
                break;
            }
            m_yuvMailboxY.swap(m_readbackScratchY);
            m_yuvMailboxCb.swap(m_readbackScratchCb);
            m_yuvMailboxCr.swap(m_readbackScratchCr);
            m_yuvTimestampNs = timestampNs;
            m_yuvSubmitInstant = submitInstant;
            m_yuvStagingIndex = m_useD3d11Input ? stagingIndex : -1;
            m_yuvReady = true;
        }
        m_yuvCv.notify_one();
    }
}

// Called by an encoder's final-stage thread each time it has queued output. Only wakes the
// collector; the unit itself is fetched there.
void VideoEncoderJpegXs::OnEncoderOutputAvailable(svt_jpeg_xs_encoder_api_t* /*encoder*/, void* context) {
    auto* self = static_cast<VideoEncoderJpegXs*>(context);
    {
        std::lock_guard<std::mutex> lock(self->m_outputReadyMutex);
        self->m_outputReadySeq++;
    }
    self->m_outputReadyCv.notify_one();
}

void VideoEncoderJpegXs::EncodeWorkerLoop() {
    while (true) {
        uint64_t timestampNs = 0;
        std::chrono::steady_clock::time_point submitInstant {};
        // D3D11-input path only: the staging slot this frame still lives in. -1 on
        // the readback path, where the slot was released as soon as it was copied out.
        int stagingIndex = -1;

        {
            std::unique_lock<std::mutex> lock(m_yuvMutex);
            m_yuvCv.wait(lock, [this] {
                return m_yuvReady || m_exiting.load(std::memory_order_acquire);
            });
            if (!m_yuvReady && m_exiting.load(std::memory_order_acquire)) {
                break;
            }
            m_planeY.swap(m_yuvMailboxY);
            m_planeCb.swap(m_yuvMailboxCb);
            m_planeCr.swap(m_yuvMailboxCr);
            timestampNs = m_yuvTimestampNs;
            submitInstant = m_yuvSubmitInstant;
            stagingIndex = m_yuvStagingIndex;
            m_yuvReady = false;
        }
        // Hands the staging slot back exactly once, however this iteration ends --
        // including the early continue on encode failure below. Until it runs,
        // Transmit must not CopyResource a newer frame into that slot, because on
        // the D3D11-input path the encoder is reading the texture directly and
        // would otherwise pick up whichever frame won the race. Does nothing when
        // stagingIndex is -1, i.e. on the readback path.
        struct StagingRelease {
            VideoEncoderJpegXs* self;
            int index;
            ~StagingRelease() {
                if (index >= 0) {
                    std::lock_guard<std::mutex> lock(self->m_mutex);
                    self->m_stagingReserved[index] = false;
                }
            }
        } stagingRelease { this, stagingIndex };
        // Wake the readback stage immediately: the mailbox slot is free
        // again, it can hand off the next frame while we encode this one.
        m_yuvCv.notify_one();

        m_encodeBusy.store(true, std::memory_order_release);
        auto t0 = std::chrono::steady_clock::now();

        // sendBuffer/sendSize point at whichever buffer this iteration actually wants
        // dumped/sent below -- m_bitstreamBuf (unwrapped, single image) when column split
        // is off, or m_wrapperBuf (the "ALVS"-tagged concatenation of N independent column
        // codestreams) when it's on. Either way the dump/send tail below is identical and
        // doesn't need to know which case produced it. Non-const to match VideoSend()'s
        // `unsigned char*` parameter (bindings.h) -- both source vectors are non-const.
        uint8_t* sendBuffer = nullptr;
        uint32_t sendSize = 0;
        bool encodeOk = false;

        if (m_columnsNum <= 1) {
            svt_jpeg_xs_image_buffer_t img = {};
            img.data_yuv[0] = m_planeY.data();
            img.data_yuv[1] = m_planeCb.data();
            img.data_yuv[2] = m_planeCr.data();
            img.stride[0] = m_width;
            img.stride[1] = m_chromaWidth;
            img.stride[2] = m_chromaWidth;
            img.alloc_size[0] = (uint32_t)m_planeY.size();
            img.alloc_size[1] = (uint32_t)m_planeCb.size();
            img.alloc_size[2] = (uint32_t)m_planeCr.size();
            // Zero-copy input: hand over the textures themselves. data_yuv above
            // still points at correctly sized buffers because send_picture
            // validates them, but they hold an older frame -- device_input_only is
            // what tells the encoder that, so it drops this frame rather than
            // encoding those bytes if the GPU path does not run. Set even when the
            // slot is missing, which is exactly when dropping is the right answer.
            if (m_useD3d11Input) {
                img.device_input_only = 1;
                if (stagingIndex >= 0 && m_deviceSlotValid[stagingIndex]) {
                    for (int p = 0; p < 3; p++) {
                        img.data_yuv_device[p] = m_devicePlane[stagingIndex][p];
                        img.stride_device[p] = m_devicePitch[stagingIndex][p];
                    }
                }
            }

            svt_jpeg_xs_bitstream_buffer_t bitstream = {};
            bitstream.buffer = m_bitstreamBuf.data();
            bitstream.allocation_size = (uint32_t)m_bitstreamBuf.size();
            bitstream.used_size = 0;

            svt_jpeg_xs_frame_t frame = {};
            frame.image = img;
            frame.bitstream = bitstream;
            frame.user_prv_ctx_ptr = nullptr;

            SvtJxsErrorType_t err = svt_jpeg_xs_encoder_send_picture(&m_enc, &frame, 1 /*blocking*/);
            if (err == SvtJxsErrorNone) {
                svt_jpeg_xs_frame_t out = {};
                err = svt_jpeg_xs_encoder_get_packet(&m_enc, &out, 1 /*blocking*/);
                if (err == SvtJxsErrorNone) {
                    sendBuffer = out.bitstream.buffer;
                    sendSize = out.bitstream.used_size;
                    encodeOk = true;
                }
            }
        } else {
            // Column-split path (Dashboard "Column splits" > 1): each column is its own
            // completely independent, ordinary JPEG XS image at m_columnWidth -- see the
            // long comment in Initialize() for why this sidesteps SVT-JPEG-XS's own
            // unimplemented native column-tiling instead of using it. Zero-copy crop: the
            // pointer offset below picks out column i's leftmost pixel, and stride stays at
            // the FULL row width, so each column encoder just reads m_columnWidth pixels
            // per row starting from its own offset -- no memcpy needed to isolate a column.
            uint32_t chromaColumnWidth = m_columnWidth / 2;
            std::vector<uint32_t> columnSizes(m_columnsNum, 0);
            encodeOk = true;

            // Phase 1: hand EVERY column's picture to its own encoder instance before
            // collecting any result.
            //
            // This used to be one loop doing send_picture(i) immediately followed by
            // get_packet(i), which meant column 0 encoded to completion before column 1
            // even started -- the column split delivered zero encoder parallelism, which is
            // the entire reason it exists (each column is an independent, ordinary JPEG XS
            // image with its own encoder instance; see Initialize()'s long comment).
            //
            // Splitting the loop is safe because send_picture() does NOT wait for encoding:
            // with blocking_flag=1 it blocks only on svt_jxs_get_empty_object() for a free
            // INPUT-queue slot, copies the frame struct, posts it, and returns
            // (EncHandle.c) -- the API doc says "blocked until frame is sent to encoder".
            // Each column has its own encoder instance, so their input queues are entirely
            // independent and one column's send can never be gated by another's progress.
            //
            // Lifetime note: `frame`/`img`/`bitstream` below are stack locals that go out of
            // scope before Phase 2 runs, which is fine -- send_picture() copies the struct
            // ("input_item->enc_input = *enc_input"). The BUFFERS they point at must outlive
            // the encode, and they do: m_planeY/Cb/Cr and m_columnBitstreamBufs[i] are all
            // members. Crucially m_columnBitstreamBufs is per-column, so concurrent encoders
            // write to disjoint buffers.
            uint32_t sentCount = 0;
            for (uint32_t i = 0; i < m_columnsNum; i++) {
                svt_jpeg_xs_image_buffer_t img = {};
                img.data_yuv[0] = m_planeY.data() + (size_t)i * m_columnWidth;
                img.data_yuv[1] = m_planeCb.data() + (size_t)i * chromaColumnWidth;
                img.data_yuv[2] = m_planeCr.data() + (size_t)i * chromaColumnWidth;
                img.stride[0] = m_width;
                img.stride[1] = m_chromaWidth;
                img.stride[2] = m_chromaWidth;
                // Bytes from THIS column's pointer to the end of the plane, not the whole
                // plane: the pointer is offset into it. The library page-locks
                // [data_yuv, data_yuv + alloc_size) for its CUDA upload, so the whole-plane
                // size ran past the end of the buffer by the column offset and locked
                // someone else's heap pages -- after which the GPU column's uploads failed
                // with "invalid argument" on every frame and silently fell back to the CPU
                // (CPU/GPU hybrid, 2026-09-11). EncHandle.c's own check is exactly this
                // "remaining bytes" contract: stride * (height - 1) + width.
                img.alloc_size[0] = (uint32_t)(m_planeY.size() - (size_t)i * m_columnWidth);
                img.alloc_size[1] = (uint32_t)(m_planeCb.size() - (size_t)i * chromaColumnWidth);
                img.alloc_size[2] = (uint32_t)(m_planeCr.size() - (size_t)i * chromaColumnWidth);
                // Same zero-copy hand-over as the single-image path, except every
                // column encoder gets the SAME full-width textures and picks its own
                // vertical slice out of them by offset -- the exact analogue of the
                // pointer-offset crop above. The offset is in bytes; these planes are
                // R8_UNORM, so one byte per pixel makes it the pixel offset too.
                if (m_hybridZeroCopy && i < m_cpuColumns) {
                    // CPU column: host pointers into the narrow readback planes. Those hold ONLY
                    // the CPU columns, packed, so the stride is that width and the column offset
                    // indexes within it -- not the full frame width the all-host path uses.
                    const uint32_t cpuChromaColumnWidth
                        = m_cpuReadbackChromaWidth / std::max(1u, m_cpuColumns);
                    img.data_yuv[0] = m_planeY.data() + (size_t)i * m_columnWidth;
                    img.data_yuv[1] = m_planeCb.data() + (size_t)i * cpuChromaColumnWidth;
                    img.data_yuv[2] = m_planeCr.data() + (size_t)i * cpuChromaColumnWidth;
                    img.stride[0] = m_cpuReadbackWidth;
                    img.stride[1] = m_cpuReadbackChromaWidth;
                    img.stride[2] = m_cpuReadbackChromaWidth;
                    // Remaining bytes from this column's pointer, the same contract as the
                    // all-host path: the library page-locks [data_yuv, data_yuv + alloc_size).
                    img.alloc_size[0]
                        = (uint32_t)((size_t)m_cpuReadbackWidth * m_height - (size_t)i * m_columnWidth);
                    img.alloc_size[1] = (uint32_t)(
                        (size_t)m_cpuReadbackChromaWidth * m_chromaHeight
                        - (size_t)i * cpuChromaColumnWidth
                    );
                    img.alloc_size[2] = img.alloc_size[1];
                } else if (m_useD3d11Input) {
                    img.device_input_only = 1;
                    if (stagingIndex >= 0 && m_deviceSlotValid[stagingIndex]) {
                        // The whole plane was mapped once; a column is a byte offset
                        // into it with the stride left at the full width, the same
                        // zero-copy crop the host planes get above.
                        const uint32_t colOff[3]
                            = { i * m_columnWidth, i * chromaColumnWidth, i * chromaColumnWidth };
                        for (int p = 0; p < 3; p++) {
                            img.data_yuv_device[p]
                                = (uint8_t*)m_devicePlane[stagingIndex][p] + colOff[p];
                            img.stride_device[p] = m_devicePitch[stagingIndex][p];
                        }
                    }
                }

                svt_jpeg_xs_bitstream_buffer_t bitstream = {};
                bitstream.buffer = m_columnBitstreamBufs[i].data();
                bitstream.allocation_size = (uint32_t)m_columnBitstreamBufs[i].size();
                bitstream.used_size = 0;

                svt_jpeg_xs_frame_t frame = {};
                frame.image = img;
                frame.bitstream = bitstream;
                frame.user_prv_ctx_ptr = nullptr;

                SvtJxsErrorType_t err
                    = svt_jpeg_xs_encoder_send_picture(&m_columnEncs[i], &frame, 1 /*blocking*/);
                if (err != SvtJxsErrorNone) {
                    encodeOk = false;
                    break;
                }
                sentCount++;
            }

            // Phase 2: collect. By the time the first get_packet() returns, every column
            // has already been encoding concurrently for the whole duration of Phase 1.
            //
            // Iterates over sentCount, NOT m_columnsNum, and uses `continue` rather than
            // `break` on error: every picture handed to an encoder in Phase 1 MUST be
            // collected, even when we've already decided this frame is a write-off. Leaving
            // a frame in flight would permanently occupy that instance's input-queue slot,
            // so the next frame's send_picture() would block against a queue that never
            // drains -- a slow-motion deadlock that would look like the encoder simply
            // stalling a few frames in.
            // Per-column collect timing: "waited" is the time to the column's first unit, "drained"
            // to its last. Whole-column mode drains the columns in index order with a BLOCKING
            // get_packet, so there a near-zero wait means the column was done and merely waited its
            // turn; slice mode collects the columns interleaved (below), so both times measure the
            // encoder itself. The units of a slice buffer point into the column's own bitstream
            // buffer, which outlives the send, so nothing is copied.
            std::vector<double> colFirstWaitMs(sentCount, 0.0);
            std::vector<double> colSpanMs(sentCount, 0.0);

            // Slice packetization: collect the columns INTERLEAVED (2026-09-25). This used to drain
            // column 0 completely with a blocking get_packet before touching column 1, so column 1's
            // slices sat in its encoder -- and, 28 output slots later, stalled its pack stage --
            // until column 0 was done, and the headset received the left half, then the right. Now
            // every column hands over whatever unit it has ready, one per column per turn, and the
            // collector sleeps on the encoders' "output available" callback while all are empty.
            // Both halves arrive spread over the whole frame, so both column decoders work the whole
            // frame period instead of one after the other.
            //
            // One rule on order: column 0's header unit (packet index 0) goes out before anything
            // else of the frame. The client takes index 0 as the start of a frame when the server
            // repeats a timestamp; a column-1 unit ahead of it would be filed into the previous
            // frame.
            if (m_slicePackets && sentCount > 0) {
                if (!m_sentConfig) {
                    m_sentConfig = true;
                    SetVideoConfigNals(nullptr, 0, ALVR_CODEC_JPEGXS);
                }
                const auto tCollectStart = std::chrono::steady_clock::now();
                std::vector<uint32_t> unitIdx(sentCount, 0);
                std::vector<bool> colDone(sentCount, false);
                uint32_t remaining = sentCount;
                bool frameHeaderSent = false;
                while (remaining > 0) {
                    uint64_t seqBefore;
                    {
                        std::lock_guard<std::mutex> lock(m_outputReadyMutex);
                        seqBefore = m_outputReadySeq;
                    }
                    bool progressed = false;
                    for (uint32_t i = 0; i < sentCount; i++) {
                        if (colDone[i] || (!frameHeaderSent && i != 0)) {
                            continue;
                        }
                        svt_jpeg_xs_frame_t unit = {};
                        const SvtJxsErrorType_t uerr
                            = svt_jpeg_xs_encoder_get_packet(&m_columnEncs[i], &unit, 0 /*non-blocking*/);
                        if (uerr == SvtJxsErrorNoErrorEmptyQueue) {
                            continue;
                        }
                        progressed = true;
                        if (unitIdx[i] == 0) {
                            colFirstWaitMs[i] = std::chrono::duration<double, std::milli>(
                                                    std::chrono::steady_clock::now() - tCollectStart
                            )
                                                    .count();
                        }
                        // A unit that reports an encode error still has to be collected (an
                        // uncollected unit keeps its queue slot and the next frame blocks); the
                        // frame is a write-off from here on and nothing more of it is sent.
                        if (uerr != SvtJxsErrorNone) {
                            encodeOk = false;
                        }
                        if (encodeOk && unitIdx[i] < m_unitsPerColumn) {
                            VideoSend(
                                timestampNs,
                                unit.bitstream.buffer,
                                (int)unit.bitstream.used_size,
                                /*isIdr=*/true,
                                (unsigned short)(i * m_unitsPerColumn + unitIdx[i]),
                                (unsigned short)(m_columnsNum * m_unitsPerColumn),
                                (unsigned char)i
                            );
                        }
                        if (i == 0) {
                            frameHeaderSent = true;
                        }
                        unitIdx[i]++;
                        if (unit.bitstream.last_packet_in_frame) {
                            colDone[i] = true;
                            remaining--;
                            colSpanMs[i] = std::chrono::duration<double, std::milli>(
                                               std::chrono::steady_clock::now() - tCollectStart
                            )
                                               .count();
                        }
                    }
                    if (!progressed && remaining > 0) {
                        if (m_exiting.load(std::memory_order_acquire)) {
                            break;
                        }
                        // The callback may have fired between the look above and here; the
                        // sequence number catches that. The timeout is only a safety net.
                        std::unique_lock<std::mutex> lock(m_outputReadyMutex);
                        m_outputReadyCv.wait_for(lock, std::chrono::milliseconds(2), [&] {
                            return m_outputReadySeq != seqBefore;
                        });
                    }
                }
                for (uint32_t i = 0; i < sentCount; i++) {
                    if (unitIdx[i] != m_unitsPerColumn) {
                        // Geometry drifted from what Initialize() computed, so the indices the
                        // client is rebuilding from are wrong. Said where it can be read: LogPeriod
                        // never reaches vrserver.txt (2026-09-22).
                        static std::atomic<uint32_t> s_sliceCountWarnings { 0 };
                        if (s_sliceCountWarnings.fetch_add(1) % 900 == 0) {
                            Warn(
                                "VideoEncoderJpegXs: column %u produced %u units, expected %u "
                                "(height %u, slice height %u) -- client indices will be off\n",
                                i,
                                unitIdx[i],
                                m_unitsPerColumn,
                                m_height,
                                kSliceHeightLines
                            );
                        }
                    }
                }
            }

            for (uint32_t i = 0; i < sentCount && !m_slicePackets; i++) {
                auto tColStart = std::chrono::steady_clock::now();
                // Whole-column packets (no slice packetization); slices are collected above.
                svt_jpeg_xs_frame_t out = {};
                SvtJxsErrorType_t err
                    = svt_jpeg_xs_encoder_get_packet(&m_columnEncs[i], &out, 1 /*blocking*/);
                colFirstWaitMs[i]
                    = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - tColStart
                    )
                          .count();
                colSpanMs[i] = colFirstWaitMs[i];
                if (err != SvtJxsErrorNone) {
                    encodeOk = false;
                    continue;
                }
                columnSizes[i] = out.bitstream.used_size;

                // Column pipelining: this column is a complete, independently decodable JPEG XS
                // codestream, so it can go out now instead of waiting for its neighbours. The
                // client starts decoding it while the rest is still being transmitted -- at the
                // full resolution a whole frame is 8.6 ms of transfer even on a 5 Gbps link.
                //
                // If a later column fails after this one is on the wire the client is left with
                // an incomplete frame; it drops it once a newer timestamp completes, the same
                // outcome as the whole-frame path, which drops the frame outright.
                if (m_pipelineColumns) {
                    if (!m_sentConfig) {
                        m_sentConfig = true;
                        SetVideoConfigNals(nullptr, 0, ALVR_CODEC_JPEGXS);
                    }
                    VideoSend(
                        timestampNs,
                        m_columnBitstreamBufs[i].data(),
                        (int)columnSizes[i],
                        /*isIdr=*/true,
                        (unsigned short)i,
                        (unsigned short)m_columnsNum,
                        (unsigned char)i
                    );
                }
            }

            if (m_columnsNum > 1 && sentCount == m_columnsNum) {
                // One line per frame period, columns in collect order. A column whose "waited"
                // is near zero was finished before its turn came -- its packets sat in the
                // encoder while a lower-numbered column was still being drained and sent.
                char colLine[256];
                int off = 0;
                for (uint32_t i = 0; i < sentCount && off < (int)sizeof(colLine) - 48; i++) {
                    off += snprintf(
                        colLine + off,
                        sizeof(colLine) - off,
                        "%scol%u(%s) waited %.2fms, drained %.2fms",
                        i == 0 ? "" : " | ",
                        i,
                        i < m_cpuColumns ? "CPU" : "GPU",
                        colFirstWaitMs[i],
                        colSpanMs[i]
                    );
                }
                // Into the diagnostics file, not LogPeriod: that one goes through Rust's
                // log::warn!, which lands in ALVR's own log and never reaches vrserver.txt --
                // checked after the first attempt produced no output anywhere readable.
                {
                    std::lock_guard<std::mutex> lock(m_columnDiagMutex);
                    m_columnDiagLine = colLine;
                }
            }

            if (encodeOk && !m_pipelineColumns && !m_slicePackets) {
                // Assemble the "ALVS" wrapper: magic + columns_num + per-column length
                // table + concatenated column codestreams, in that order -- see the header
                // comment on m_wrapperBuf. The client-side decoder detects this magic
                // before treating incoming bytes as a plain single-column JPEG XS stream,
                // so this format and the original unwrapped one coexist on the wire without
                // any other protocol change.
                static constexpr uint8_t kSplitMagic[4] = { 'A', 'L', 'V', 'S' };
                size_t headerSize = sizeof(kSplitMagic) + 4 + (size_t)m_columnsNum * 4;
                size_t totalDataSize = 0;
                for (uint32_t i = 0; i < m_columnsNum; i++) {
                    totalDataSize += columnSizes[i];
                }
                size_t neededSize = headerSize + totalDataSize;
                if (m_wrapperBuf.size() < neededSize) {
                    m_wrapperBuf.resize(neededSize);
                }
                uint8_t* w = m_wrapperBuf.data();
                std::memcpy(w, kSplitMagic, sizeof(kSplitMagic));
                w += sizeof(kSplitMagic);
                std::memcpy(w, &m_columnsNum, 4);
                w += 4;
                for (uint32_t i = 0; i < m_columnsNum; i++) {
                    std::memcpy(w, &columnSizes[i], 4);
                    w += 4;
                }
                for (uint32_t i = 0; i < m_columnsNum; i++) {
                    std::memcpy(w, m_columnBitstreamBufs[i].data(), columnSizes[i]);
                    w += columnSizes[i];
                }
                sendBuffer = m_wrapperBuf.data();
                sendSize = (uint32_t)neededSize;
            }
        }

        if (!encodeOk) {
            m_encodeBusy.store(false, std::memory_order_release);
            continue;
        }

        auto t1 = std::chrono::steady_clock::now();
        double encodeMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        LogPeriod("VideoEncoderJpegXs.Encode", "encode=%.2fms", encodeMs);
        DiagTick("encode", encodeMs);

        // With column pipelining nothing is left to send here (the columns went out in Phase 2)
        // and sendSize stays 0, but the diagnostics below must still run. Dumping to disk needs
        // the assembled wrapper, so it is simply off in that mode -- see Initialize()'s warning.
        if (sendSize > 0 || m_pipelineColumns || m_slicePackets) {
            if (m_streamDumper && sendSize > 0) {
                m_streamDumper->WritePacket(sendBuffer, sendSize);
                // JPEG XS is intra-only: every frame is effectively a keyframe.
                m_streamDumper->OnFrameEncoded(sendSize, true);
            }

            // Real network send -- the same FFI calls H264/HEVC/AV1 use
            // (bindings.h), called directly rather than through
            // ParseFrameNals()/NalParsing.cpp: JPEG XS has no NAL-unit
            // structure at all, so there's nothing for that NAL scanner to
            // do here. SetVideoConfigNals() is sent once, empty -- JPEG
            // XS's codestream is self-describing per frame (like AV1's own
            // in-band sequence header), so there's no persistent SPS/PPS-
            // equivalent parameter set to extract up front.
            //
            // Skipped entirely for a side-channel (dump-only) instance: only the ONE real,
            // network-streamed encoder may ever call these -- otherwise a side instance
            // running alongside e.g. HEVC would corrupt the real stream's video config/
            // packet sequence (VideoSend()'s target is the single active stream, not
            // per-encoder).
            if (!m_sideChannelOnly) {
                if (!m_sentConfig) {
                    m_sentConfig = true;
                    SetVideoConfigNals(nullptr, 0, ALVR_CODEC_JPEGXS);
                }
                // Already sent, per column or per slice, back in Phase 2.
                if (!m_pipelineColumns && !m_slicePackets) {
                    VideoSend(
                        timestampNs,
                        sendBuffer,
                        (int)sendSize,
                        /*isIdr=*/true,
                        0,
                        /*packetsPerFrame=*/1,
                        /*column=*/0
                    );
                }
                // This is the same Transmit()-to-VideoSend() window ALVR's own "encoder
                // latency" stat measures (frame_composed to frame_encoded, statistics.rs)
                // -- logged right here, right before the same VideoSend() call that
                // stat's report_frame_encoded() is keyed off of, so the two should read
                // the same. Meaningless for a side-channel instance (no VideoSend() to
                // anchor against), so skipped there too.
                double fullSpanMs = std::chrono::duration<double, std::milli>(
                                         std::chrono::steady_clock::now() - submitInstant
                )
                                         .count();
                DiagTick("full_span", fullSpanMs);
            }
        }

        m_encodeBusy.store(false, std::memory_order_release);
    }
}

void VideoEncoderJpegXs::Shutdown() {
    if (m_exiting.exchange(true)) {
        return; // already shut down
    }
    m_cv.notify_all();
    m_yuvCv.notify_all();
    if (m_readbackThread.joinable()) {
        m_readbackThread.join();
    }
    if (m_encodeThread.joinable()) {
        m_encodeThread.join();
    }

    if (m_encInitialized) {
        if (m_columnsNum <= 1) {
            svt_jpeg_xs_encoder_close(&m_enc);
        } else {
            for (auto& columnEnc : m_columnEncs) {
                svt_jpeg_xs_encoder_close(&columnEnc);
            }
        }
        m_encInitialized = false;
    }

    if (m_streamDumper) {
        m_streamDumper->Close();
    }
}

#endif // ALVR_JPEGXS
