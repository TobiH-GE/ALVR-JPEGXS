#pragma once

// ---------------------------------------------------------------------------
// VideoEncoderJpegXs
//
// Real VideoEncoder implementation using Intel's open-source SVT-JPEG-XS
// encoder. Selected by CEncoder::Initialize() when Settings::m_codec ==
// ALVR_CODEC_JPEGXS -- see the dispatch logic there. Originally built (and
// still structurally identical apart from where its output goes) as an
// additive, disk-dump-only side channel for comparing JPEG XS against the
// real HEVC/H.264/AV1 stream; promoted to a real, selectable, network-
// streamed codec by routing its encoded output through SetVideoConfigNals()/
// VideoSend() (see EncodeWorkerLoop()) instead of only StreamDumper.
//
// RGB->YUV420 conversion happens on the GPU, using our own small pixel
// shaders (compiled at runtime via D3DCompile, see EnsureGpuPipeline()) and
// our own minimal two-pass render sequence rather than ALVR's
// RenderPipelineYUV class -- see the comment on RenderYuvGpu() for why. We
// render Y and UV into two *separate*, ordinary (non-planar) textures of our
// own -- never a shared NV12 resource, and never touching FrameRender's
// m_pStagingTexture -- so this is purely additive. We initially reused
// ALVR's own already-compiled rgbtoyuv420.hlsl for this, but its chroma
// addressing depends on a viewport-larger-than-render-target clipping trick
// that turned out not to hold up on real hardware (see m_yPixelShader's
// comment in the header) -- our own shaders sidestep that entirely.
//
// Note: we deliberately do NOT render into a single NV12-format texture
// (one resource, two planar RTVs) the way ALVR's own RenderPipelineYUV does.
// That mechanism is only ever exercised by ALVR's real path when HDR is
// enabled (FrameRender.cpp, gated on Settings::m_enableHdr) -- i.e. it's
// essentially unvalidated on most systems, including whichever GPU/driver
// this was developed against, where it silently produced an all-zero
// (solid green when decoded) result despite every D3D11 call succeeding.
// Two ordinary R8_UNORM/R8G8_UNORM render targets sidestep that planar-RTV
// codepath entirely and are about as vanilla as GPU render-to-texture gets.
//
// Uses Intel's open source SVT-JPEG-XS encoder/decoder library
// (https://github.com/OpenVisualCloud/SVT-JPEG-XS, BSD-2-Clause-Patent).
// This is a build-time optional dependency: the whole class compiles out
// unless ALVR_JPEGXS is defined, which build.rs only does when it finds
// deps/windows/svt-jpegxs/{include,lib} populated. See
// deps/windows/svt-jpegxs/README.md for how to build/place that dependency.
//
// Output (network): SetVideoConfigNals()/VideoSend() (bindings.h), the same
// FFI calls H264/HEVC/AV1 use -- see EncodeWorkerLoop(). No NAL-unit framing
// (JPEG XS has none) -- these are called directly, bypassing NalParsing.cpp
// entirely, matching how AV1 already bypasses NAL scanning.
// Output (optional disk dump, Dashboard "Dump stream to disk"):
//  - <output dir>\alvr_stream_jpegxs_svtjpegxs.jxs   raw JPEG XS codestream
//    (concatenated per-frame codestreams; FFmpeg's raw JPEG-XS demuxer,
//    merged upstream in Dec 2025, understands this layout, as should any
//    JPEG-XS-enabled VLC build using that FFmpeg).
//  - <output dir>\alvr_stream_jpegxs_svtjpegxs_stats.csv  same FPS/bitrate
//    CSV format StreamDumper uses for the HEVC/AMF/AV1 dumps.
// ---------------------------------------------------------------------------

#ifdef ALVR_JPEGXS

#include "VideoEncoder.h"
#include "alvr_server/StreamDumper.h"
#include "d3d-render-utils/RenderUtils.h"
#include "shared/d3drender.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <d3d11.h>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <wrl.h>
#include <malloc.h>
#include <new>

extern "C" {
#include <SvtJpegxsEnc.h>
}

// Page-aligned, page-padded storage for the YUV planes handed to the encoder.
//
// On the CUDA path the library page-locks each input plane (cudaHostRegister) for its
// upload, and registration works on whole pages. The live chroma planes are small enough
// (864x256 = 221 KB) to come from the ordinary heap, where two of them can share a page --
// and the three buffers per plane rotate (readback scratch -> mailbox -> encode), so one
// plane's upload could then start inside another's registration. CUDA rejects such a copy
// with "invalid argument", which is what made the GPU column of the CPU/GPU hybrid fall
// back to the CPU on every frame (2026-09-11). With every plane on pages of its own that
// cannot happen. Stateless, so swap() between the buffers stays a pointer swap.
template <typename T> struct PageAlignedAllocator {
    using value_type = T;
    static constexpr size_t kPageBytes = 4096;

    PageAlignedAllocator() noexcept = default;
    template <typename U> PageAlignedAllocator(const PageAlignedAllocator<U>&) noexcept { }

    T* allocate(size_t n) {
        size_t bytes = (n * sizeof(T) + kPageBytes - 1) / kPageBytes * kPageBytes;
        void* p = _aligned_malloc(bytes == 0 ? kPageBytes : bytes, kPageBytes);
        if (p == nullptr) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(p);
    }
    void deallocate(T* p, size_t) noexcept { _aligned_free(p); }

    template <typename U> bool operator==(const PageAlignedAllocator<U>&) const noexcept { return true; }
    template <typename U> bool operator!=(const PageAlignedAllocator<U>&) const noexcept { return false; }
};
using PlaneBuffer = std::vector<uint8_t, PageAlignedAllocator<uint8_t>>;

class VideoEncoderJpegXs : public VideoEncoder {
public:
    // sideChannelOnly: true for the additive dump-only instance CEncoder constructs when
    // a DIFFERENT codec (e.g. HEVC) is the real, network-streamed one but JPEG XS's own
    // "Dump stream to disk" setting is also on -- runs the exact same GPU render/readback/
    // encode pipeline for testing/comparison, but never calls SetVideoConfigNals()/
    // VideoSend() (see EncodeWorkerLoop()), so it can't interfere with whatever the real
    // encoder is actually sending over the network.
    VideoEncoderJpegXs(
        std::shared_ptr<CD3DRender> d3dRender,
        uint32_t width,
        uint32_t height,
        bool sideChannelOnly = false
    );
    ~VideoEncoderJpegXs();

    // Throws (see MakeException) if the encoder could not be set up
    // (SVT-JPEG-XS init failure, etc) -- CEncoder::Initialize() catches this
    // and falls back to hardware encoder probing, exactly like AMF/NVENC's
    // own Initialize() failures already do. The GPU conversion pipeline
    // itself is set up lazily on the first Transmit() call (see there for
    // why).
    void Initialize() override;
    void Shutdown() override;

    // Genuinely non-blocking: only issues async GPU work (a render pass + a
    // CopyResource enqueue) and hands a staging-slot index off to the
    // readback thread -- the actual blocking GPU Map() / memcpy() readback
    // happens on ReadbackWorkerLoop, never on this (real-time critical)
    // thread. If the mailbox still holds an unconsumed frame, this frame is
    // silently dropped rather than stalling the caller. presentationTime/
    // insertIDR are accepted but unused -- JPEG XS is intra-only, every
    // frame is effectively a keyframe.
    void Transmit(
        ID3D11Texture2D* pTexture,
        uint64_t presentationTime,
        uint64_t targetTimestampNs,
        bool insertIDR
    ) override;

private:
    // One-time setup of the GPU RGB->YUV420 render pipeline, run on the
    // first Transmit() call (not in Initialize()) because it needs a
    // source texture to bind as input -- FrameRender's staging texture is a
    // stable, persistent GPU object reused every frame (created once in
    // FrameRender::Startup(), only its contents change), so binding it once
    // here and re-rendering every frame is safe. Returns false (and sets
    // m_gpuSetupFailed) if anything throws.
    bool EnsureGpuPipeline(ID3D11Texture2D* sourceTexture);

    // Renders Y and UV in two separate single-render-target passes, both
    // using the same rgbtoyuv420.hlsl pixel shader (it always computes all
    // three Y/U/V values; whichever output isn't bound to a target that
    // draw call is simply discarded). We deliberately do NOT use ALVR's own
    // d3d_render_utils::RenderPipelineYUV here, even though it wraps the
    // same shader for the real HDR path: its Render() binds Y and UV as two
    // simultaneous render targets in one draw call, which -- per a comment
    // left in RenderPipelineYUV.cpp itself ("HACK: Whyyyyyy does the Y
    // channel only render the top-left corner with both render targets
    // enabled????") -- only writes a small top-left region correctly in
    // that configuration. Its existing workaround re-draws Y alone
    // afterwards to fix Y specifically, but never does the equivalent for
    // UV, leaving most of the chroma plane unwritten/zero. That's fine for
    // casual HDR video viewing (average person may not choose to
    // pixel-diff their headset feed) but fatal for us: zero chroma decodes
    // as a hard green cast, which is exactly what showed up in the dumped
    // JPEG XS file. Doing two clean single-target passes here -- without
    // touching the shared RenderPipelineYUV class the other real encoders
    // (AMF/NVENC) also use -- fixes it for this class only.
    void RenderYuvGpu(ID3D11DeviceContext* context);

    // Two-stage pipeline: ReadbackWorkerLoop (GPU Map()/memcpy + UV
    // deinterleave) and EncodeWorkerLoop (SVT-JPEG-XS) run on separate
    // threads so encoding frame N can overlap with reading back frame N+1,
    // instead of running fully serially. They hand off through the m_yuv*
    // mailbox below. The RGB->YUV420 color conversion itself no longer runs
    // on the CPU at all -- the GPU does it before either of these stages.
    void ReadbackWorkerLoop();
    void EncodeWorkerLoop();

    std::shared_ptr<CD3DRender> m_d3dRender;

    // GPU-side RGB->YUV420 conversion, reusing ALVR's own rgbtoyuv420.hlsl
    // shader (see EnsureGpuPipeline()/RenderYuvGpu()) but with our own
    // minimal two-pass render sequence, not ALVR's RenderPipelineYUV class
    // -- see the long comment at the top of this file for why. m_gpuYTex /
    // m_gpuCbTex / m_gpuCrTex are our own plain render targets -- never
    // FrameRender's m_pStagingTexture -- so this never affects the real
    // stream.
    Microsoft::WRL::ComPtr<ID3D11VertexShader> m_quadVertexShader;
    // ALVR's "quad" vertex shader (QuadVertexShader.cso) is *not*
    // SV_VertexID-procedural -- it needs a real bound vertex buffer,
    // matching FrameRender's SimpleVertex layout (POSITION float4 @0,
    // TEXCOORD float2 @16, VIEW uint @24; see FrameRender.h). FrameRender's
    // own color-correction/FFR/HDR-YUV passes only work because they run
    // immediately after FrameRender's own eye-compositing draw in the same
    // function call, silently inheriting *that* draw's still-bound vertex
    // buffer + input layout. We have no such guarantee (different thread,
    // different call context entirely), so we create and bind our own
    // self-contained quad -- 4 vertices, plain full-screen NDC quad,
    // TEXCOORD 0..1 -- instead of relying on an accident of leftover state.
    Microsoft::WRL::ComPtr<ID3D11InputLayout> m_quadInputLayout;
    Microsoft::WRL::ComPtr<ID3D11Buffer> m_quadVertexBuffer;
    // Two separate, minimal pixel shaders of our own (compiled at runtime via
    // D3DCompile from an embedded HLSL string in VideoEncoderJpegXs.cpp) --
    // NOT ALVR's shared rgbtoyuv420.hlsl / RGBTOYUV420_CSO anymore. That
    // shared shader's chroma pass relies on a "full-size viewport bound to a
    // half-size render target, rely on rasterizer clipping to the physical
    // target extent, then *2-mod-width the raw clipped pixel index back into
    // a full-res source address" trick, which turned out to NOT reliably
    // clip the way the D3D11 spec's "viewport may exceed the bound render
    // target" wording suggests it should on real hardware -- in practice we
    // observed part of the source image rendered enlarged/misplaced into the
    // chroma plane instead of being cleanly clipped. Our own shaders instead
    // use a genuinely correctly-sized viewport per pass (see m_gpuViewport
    // for Y, m_uvViewport for UV, both set in EnsureGpuPipeline) and sample
    // the source directly at the interpolated `uv` texcoord with no manual
    // texel-index math at all -- standard, unambiguous GPU behavior.
    Microsoft::WRL::ComPtr<ID3D11PixelShader> m_yPixelShader;
    // Dual-output (SV_Target0=Cb, SV_Target1=Cr) -- writes both chroma planes
    // directly as separate R8_UNORM render targets, so the CPU-side readback
    // never has to deinterleave an R8G8 texture (see m_gpuCbTex/m_gpuCrTex
    // below and the comment in ReadbackWorkerLoop() this replaced).
    Microsoft::WRL::ComPtr<ID3D11PixelShader> m_uvPixelShader;
    Microsoft::WRL::ComPtr<ID3D11Buffer> m_yuvParamBuffer;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_sourceSRV;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> m_gpuYTex;
    // Separate plain R8_UNORM render targets for Cb/Cr -- not a single
    // interleaved R8G8 texture. SVT-JPEG-XS wants Cb/Cr as independent
    // planes anyway; writing them directly as two render targets in the same
    // draw call (MRT) means the CPU never has to unpack an interleaved
    // buffer -- see mainUV's UVOutput struct in EnsureGpuPipeline().
    Microsoft::WRL::ComPtr<ID3D11Texture2D> m_gpuCbTex;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> m_gpuCrTex;
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> m_rtvY;
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> m_rtvCb;
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> m_rtvCr;
    // Full-resolution viewport, used for the Y pass (matches m_gpuYTex's
    // real dimensions exactly -- no clipping trickery involved).
    D3D11_VIEWPORT m_gpuViewport {};
    // Half-resolution viewport (m_chromaWidth x m_chromaHeight), used for
    // the UV pass -- matches m_gpuCbTex/m_gpuCrTex's real dimensions
    // exactly. Having a
    // genuinely correctly-sized viewport per render target, rather than
    // sharing one oversized viewport between both passes, is what actually
    // fixes the chroma-plane corruption; see the comment on m_yPixelShader.
    D3D11_VIEWPORT m_uvViewport {};
    // Explicit, known-good pipeline state for RenderYuvGpu(). We share the
    // immediate context with FrameRender's own real-path rendering (and
    // whatever CEncoder/VideoEncoder submission work runs in between), so we
    // cannot assume the rasterizer/blend/depth state is at D3D11 defaults
    // when our draws run -- an inherited scissor rect, cull mode, or write
    // mask left over from earlier in the frame can silently discard every
    // pixel of our two-pass draw, leaving m_gpuYTex/m_gpuCbTex/m_gpuCrTex at
    // their all-zero creation-time content (which decodes to solid green). Binding
    // our own no-cull/no-scissor/opaque/depth-disabled state every call rules that
    // class of bug out entirely, independent of whatever the real path did
    // just before us.
    Microsoft::WRL::ComPtr<ID3D11RasterizerState> m_rasterizerState;
    Microsoft::WRL::ComPtr<ID3D11BlendState> m_blendState;
    Microsoft::WRL::ComPtr<ID3D11DepthStencilState> m_depthStencilState;
    bool m_gpuPipelineReady = false;
    bool m_gpuSetupFailed = false;

    // Feed the encoder the D3D11 textures instead of reading them back to the host.
    // Set in Initialize() when CUDA encoding is on: the staging ring is then created
    // as USAGE_DEFAULT (a STAGING resource cannot be registered with CUDA) and
    // ReadbackWorkerLoop stops mapping and copying entirely -- it only forwards the
    // slot. Mirrors how NVENC is driven here: the encoder owns input surfaces and
    // each frame is a GPU-side CopyResource into one of them.
    bool m_useD3d11Input = false;

    // Ring of GPU staging copies of m_gpuYTex/m_gpuCbTex/m_gpuCrTex. Transmit()
    // CopyResource()s into one slot per frame (cheap async GPU enqueue, three calls
    // -- one per plane) and hands the slot written one call EARLIER (so the GPU has
    // had a full frame period to get to it) off to ReadbackWorkerLoop.
    // m_stagingReserved marks a slot as handed off and not yet finished with, so
    // Transmit knows not to recycle it into a new CopyResource() underneath whoever
    // is still reading it.
    //
    // Two slots on the readback path, where a slot is released as soon as readback
    // has memcpy'd it out. Three on the D3D11-input path: there the ENCODER reads
    // the texture, so the slot stays reserved until the encode finishes -- roughly
    // two frame periods rather than a fraction of one, and with only two slots
    // Transmit would find its next write slot still reserved and drop every second
    // frame. m_stagingSlots says which of the two shapes is live.
    Microsoft::WRL::ComPtr<ID3D11Texture2D> m_stagingYTex[3];
    Microsoft::WRL::ComPtr<ID3D11Texture2D> m_stagingCbTex[3];
    Microsoft::WRL::ComPtr<ID3D11Texture2D> m_stagingCrTex[3];
    int m_stagingSlots = 2;

    // Hybrid zero-copy ("CUDA hybrid: read back only the CPU's columns"). The set above is then
    // a DEFAULT resource that CUDA registers for the GPU columns, which rules out mapping it
    // from the CPU -- so the CPU columns get their own, narrow, STAGING copies, filled by
    // CopySubresourceRegion with just their part of the picture. What this avoids is the plain
    // hybrid's round trip: whole frame down to system memory, GPU half straight back up again.
    // Latest per-column collect timing, written by the encode thread and printed by the
    // diagnostics window on the transmit thread -- hence the lock.
    std::mutex m_columnDiagMutex;
    std::string m_columnDiagLine;

    bool m_hybridZeroCopy = false;
    // Number of leading columns encoded on the CPU (0 when there is no hybrid).
    uint32_t m_cpuColumns = 0;
    // Width of the CPU columns' region, luma and chroma. Also the stride of the narrow readback
    // planes, since those hold that region packed.
    uint32_t m_cpuReadbackWidth = 0;
    uint32_t m_cpuReadbackChromaWidth = 0;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> m_cpuStagingYTex[3];
    Microsoft::WRL::ComPtr<ID3D11Texture2D> m_cpuStagingCbTex[3];
    Microsoft::WRL::ComPtr<ID3D11Texture2D> m_cpuStagingCrTex[3];
    uint64_t m_stagingTimestampNs[3] = { 0, 0, 0 };
    // Diagnostic only (2026-09-05): wall-clock instant Transmit() was called for this
    // slot's frame, propagated through the same mailbox hand-offs as the pixel data so
    // EncodeWorkerLoop can log the FULL Transmit()-to-VideoSend() span right before
    // calling VideoSend() -- this is what ALVR's own "encoder_latency" stat actually
    // measures (frame_composed to frame_encoded), unlike DiagTick's "readback"/"encode"
    // durations, which only time each stage's own compute and exclude however long that
    // stage's worker thread sat blocked on its condition_variable waiting to be scheduled
    // in the first place. Comparing the two pinpoints whether a gap is real compute cost
    // or OS thread-wakeup/scheduling latency.
    std::chrono::steady_clock::time_point m_stagingSubmitInstant[3];
    std::chrono::steady_clock::time_point m_pendingSubmitInstant;
    std::chrono::steady_clock::time_point m_yuvSubmitInstant;
    bool m_stagingHasData[3] = { false, false, false };
    bool m_stagingReserved[3] = { false, false, false };
    int m_stagingWriteIndex = 0;
    // Slot written by the PREVIOUS Transmit() call, or -1 before the first one --
    // that is the slot handed off, not the one just written. With two slots this is
    // the same thing the old alternating index computed; it is spelled out because
    // three slots no longer make it "the other one".
    int m_stagingPrevIndex = -1;

    // Device pointers for each staging slot, produced by Transmit() itself.
    //
    // The handover has to happen on THIS thread. CUDA graphics interop maps a
    // D3D11 resource through the device that owns it, and mapping from an encoder
    // thread while this one renders deadlocks the two: the renderer blocks in
    // Flush, the encoder blocks in the map, and the GPU is reset a few seconds
    // later. Reproduced outside SteamVR in the SVT-JPEG-XS repository under
    // Source/App/CudaD3D11Stress: from the owning thread it runs indefinitely,
    // from another thread it stops within a dozen frames. So Transmit does the
    // map right after its own Flush, and the encoder only ever sees plain device
    // pointers through data_yuv_device[].
    //
    // One scratch buffer per (slot, plane), because Transmit runs a frame or two
    // ahead of the encode and the buffers are reused -- a slot must not be
    // overwritten while an earlier frame is still being encoded from it.
    void* m_devicePlane[3][3] = {};
    uint32_t m_devicePitch[3][3] = {};
    bool m_deviceSlotValid[3] = { false, false, false };

    uint32_t m_width;
    uint32_t m_height;
    // 4:2:0 chroma planes are half-width, half-height (m_gpuCbTex/m_gpuCrTex's
    // actual dimensions).
    uint32_t m_chromaWidth;
    uint32_t m_chromaHeight;

    svt_jpeg_xs_encoder_api_t m_enc {};
    bool m_encInitialized = false;
    // See the constructor's doc comment. Const after construction, read from
    // EncodeWorkerLoop (no synchronization needed).
    const bool m_sideChannelOnly = false;

    // Column split ("Column splits" Dashboard setting; see the long comment in
    // Initialize()/.cpp). m_columnsNum == 1 means split is off -- m_enc above is the only
    // encoder and EncodeWorkerLoop takes the original, unwrapped single-image code path
    // unchanged. m_columnsNum > 1 means m_columnEncs holds that many independent encoder
    // instances instead (m_enc is left default-initialized/unused in that case), each
    // configured with source_width = m_columnWidth, encoding its own zero-copy pixel crop
    // (pointer offset into m_planeY/Cb/Cr, stride kept at the FULL frame width -- see
    // EncodeWorkerLoop). Both set once in Initialize(), read only from EncodeWorkerLoop
    // afterwards (no synchronization needed).
    uint32_t m_columnsNum = 1;
    uint32_t m_columnWidth = 0;
    std::vector<svt_jpeg_xs_encoder_api_t> m_columnEncs;
    // Per-column encode output buffers (EncodeWorkerLoop writes into these directly via
    // svt_jpeg_xs_encoder_get_packet, then copies each into m_wrapperBuf below). Sized once
    // in Initialize(), same margin-over-target-bitrate logic as m_bitstreamBuf but scaled to
    // one column's pixel count instead of the full frame's.
    std::vector<std::vector<uint8_t>> m_columnBitstreamBufs;
    // Assembled "ALVS" wrapper frame (see kSplitMagic in the .cpp): magic + columns_num +
    // per-column length table + concatenated column codestreams, in that order. Only used
    // when m_columnsNum > 1 -- this is what actually goes to StreamDumper/VideoSend in that
    // case, in place of m_bitstreamBuf. Resized on demand in EncodeWorkerLoop (grows to fit,
    // never shrinks, so no reallocation once a steady-state frame size is reached).
    std::vector<uint8_t> m_wrapperBuf;

    // Set after the one-time, empty SetVideoConfigNals() call on the first
    // successfully encoded frame (EncodeWorkerLoop) -- a member rather than
    // a function-local static so it resets correctly if this encoder is
    // ever destroyed and a new one constructed for a fresh stream.
    bool m_sentConfig = false;
    // Column pipelining: send each column the moment it is encoded, as its own video packet
    // carrying (columnIndex, columnCount), instead of concatenating the columns behind the
    // "ALVS" header and sending one packet per frame. Lets the client decode a column while
    // the next is still on the wire. Only ever set for the real streaming encoder with a
    // column split; the dump/side-channel path keeps the wrapper.
    bool m_pipelineColumns = false;
    // Slice packetization (RFC 9134): the library hands each column back as a header unit plus
    // one unit per slice, and each goes on the wire on its own. Implies per-packet sending, so
    // it supersedes m_pipelineColumns when both are set.
    bool m_slicePackets = false;
    // Header unit + one per slice. Computed, not counted, because the client rebuilds each
    // column from packet indices and needs the same arithmetic: slices are ceil(height /
    // slice_height) and the slice height is pinned below when this mode is on.
    uint32_t m_unitsPerColumn = 1;
    static constexpr uint32_t kSliceHeightLines = 16;
    // Interleaved collect (2026-09-25): with slice packets the columns' units are taken as they
    // become ready, in turn, instead of column 0 completely before column 1. The encoders'
    // "output available" callback bumps m_outputReadySeq, so the collector sleeps while every
    // column is empty rather than polling. See EncodeWorkerLoop.
    std::mutex m_outputReadyMutex;
    std::condition_variable m_outputReadyCv;
    uint64_t m_outputReadySeq = 0;
    static void OnEncoderOutputAvailable(svt_jpeg_xs_encoder_api_t* encoder, void* context);

    // Encode stage's local working copy of one frame's YUV420 planes (swapped
    // in from the m_yuv* mailbox below), plus its bitstream output buffer.
    // Touched only by EncodeWorkerLoop.
    PlaneBuffer m_planeY, m_planeCb, m_planeCr; // PlaneBuffer: see PageAlignedAllocator
    std::vector<uint8_t> m_bitstreamBuf;

    std::unique_ptr<StreamDumper> m_streamDumper;

    // Stage 1 mailbox: Transmit (real encode thread) -> ReadbackWorkerLoop.
    // Single-slot, guarded by m_mutex. Carries only a staging-texture slot
    // index + timestamp -- NOT pixel data. ReadbackWorkerLoop does the
    // actual Map()/memcpy() readback and UV deinterleave, which is the whole
    // point: Transmit never blocks on the GPU.
    int m_pendingStagingIndex = -1;
    uint64_t m_pendingTimestampNs = 0;
    bool m_hasPendingFrame = false;
    std::mutex m_mutex;
    std::condition_variable m_cv;

    // ReadbackWorkerLoop's private scratch buffers: Map()/memcpy()/
    // deinterleave lands here first, so the GPU staging slot (and its
    // m_stagingReserved flag) can be released immediately, before waiting
    // on the yuv mailbox below to free up. Only ever touched by
    // ReadbackWorkerLoop -- no locking needed.
    PlaneBuffer m_readbackScratchY, m_readbackScratchCb, m_readbackScratchCr;

    // Stage 2 mailbox: ReadbackWorkerLoop -> EncodeWorkerLoop. Single-slot,
    // guarded by m_yuvMutex. Readback blocks here (backpressure) if the slot
    // still holds a frame the encode stage hasn't picked up yet, so readback
    // never gets more than one frame ahead of encode -- this is what lets
    // readback(N+1) genuinely overlap with encode(N) instead of the two
    // running serially.
    PlaneBuffer m_yuvMailboxY, m_yuvMailboxCb, m_yuvMailboxCr;
    uint64_t m_yuvTimestampNs = 0;
    // CUDA input path only: which staging slot this frame lives in. The encoder
    // reads the textures directly, so the slot has to stay reserved until the
    // ENCODE finishes, not just until readback would have copied it out.
    int m_yuvStagingIndex = -1;
    bool m_yuvReady = false;
    std::mutex m_yuvMutex;
    std::condition_variable m_yuvCv;

    std::atomic_bool m_encodeBusy { false };
    std::atomic_bool m_exiting { false };
    std::thread m_readbackThread;
    std::thread m_encodeThread;

    // Plain-file frame-pacing diagnostic (2026-09-05): the client sees a rock-steady
    // ~10 new frames/sec no matter how fast decode runs, even though encode alone only
    // takes ~13-14ms (should support 70+fps) -- these track each pipeline stage's own
    // call rate/skip count/duration directly, written to a plain file (DiagFlushLocked())
    // bypassing ALVR's own Rust-routed LogPeriod, so the numbers are guaranteed to land
    // somewhere inspectable regardless of whether the Dashboard's live log view is open.
    std::mutex m_diagMutex;
    std::chrono::steady_clock::time_point m_diagWindowStart {};
    int m_diagTransmitCalls = 0;
    int m_diagTransmitSkips = 0;
    int m_diagReadbackCount = 0;
    double m_diagReadbackMsTotal = 0.0;
    int m_diagEncodeCount = 0;
    double m_diagEncodeMsTotal = 0.0;
    int m_diagFullSpanCount = 0;
    double m_diagFullSpanMsTotal = 0.0;
    double m_diagFullSpanMsMax = 0.0;
    void DiagTick(const char* stage, double durationMs, bool skipped = false);
};

#endif // ALVR_JPEGXS
