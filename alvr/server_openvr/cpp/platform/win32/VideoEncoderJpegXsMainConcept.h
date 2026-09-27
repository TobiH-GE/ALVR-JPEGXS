#pragma once

// ---------------------------------------------------------------------------
// VideoEncoderJpegXsMainConcept
//
// Test-only, additive side-channel encoder using MainConcept's commercial
// JPEG XS SDK (demo/evaluation build, C:\Temp\mainconcept at the time this was
// written) instead of Intel's open-source SVT-JPEG-XS (see VideoEncoderJpegXs
// for that one). Built only when deps/windows/mainconcept-jxs/include exists
// (see build.rs) -- absent on a normal checkout, exactly like the SVT-JPEG-XS
// dependency.
//
// Purpose: compare MainConcept's GPU/CUDA-accelerated encoder against
// SVT-JPEG-XS (CPU) and against the real HEVC/AV1 stream, on real ALVR
// frame content, without touching the live network path at all -- same
// "additive side-channel dump" pattern VideoEncoderJpegXs itself supports
// via sideChannelOnly=true, reusing StreamDumper for the raw output + a
// fps/bitrate stats CSV.
//
// Color path: FOURCC_YUY2, 4:2:2, produced by our own shader.
//
// MainConcept's GPU encoder cannot do 4:2:0 in any form. Asked for I420 it
// answers "Allowed values are: YUYV YUY2 v210 BGR4 RGB3 Y216 A64L", with or
// without allow_csc. So this encoder cannot simply take SVT-JPEG-XS's place:
// that stream is YUV 4:2:0 and the client decodes it as such. Of what is on
// offer, YUY2 is the closest -- still YCbCr, still three components, chroma
// merely at full vertical resolution instead of half -- so it asks the least
// of a client decoder written for 4:2:0. BGR4 was used first because it needs
// no shader at all (the compositor output is already RGBA), but it puts RGB
// 4:4:4 on the wire, which is a much larger step away from the client.
//
// The conversion reuses VideoEncoderJpegXs's shader maths unchanged, BT.709
// coefficients and sRGB re-encoding included -- that is the shader that was
// debugged until the picture stopped coming out green, and none of that
// reasoning depends on the packing. Only the output differs: one render target
// of HALF the width in R8G8B8A8_UNORM, each texel holding (Y0, Cb, Y1, Cr) for
// two neighbouring pixels, which in memory is exactly YUY2. The readback is
// then a memcpy per row instead of the per-pixel channel swizzle over 7.1M
// pixels a frame that the BGR4 path did on the CPU.
//
// Measured per column (2144x1664, bpp 1.2, GPU): 224fps at 4.5ms, and the
// codestreams decode back cleanly with MainConcept's own decoder.
//
// GPU acceleration: mc_enc_jxs_settings_t::gpu_accel is 1. It was 0, on the
// conclusion that the demo jpegxs-gpu-dll.dll hangs during CUDA init on an
// RTX 5090. It does not. The DLL carries no Blackwell cubin -- cuobjdump
// --list-elf stops at sm_80 -- but it does carry PTX for all 255 kernels, so
// the driver compiles them for sm_120 on first use. That takes about a
// second, once, and is cached in %APPDATA%\NVIDIA\ComputeCache afterwards:
// the next run starts in 15ms. What looked like a hang was that one-time
// compile, and PTX forward-compatibility is exactly the mechanism that makes
// an sm_80-era build run on Blackwell.
//
// The other thing that was wrong, and the reason this side channel never
// produced anything on either path: the level was left at what
// jxsOutVideoDefaults sets, 2K-1, which caps the image at 2048 wide. At
// ALVR's 4288 jxsOutVideoNew fails outright, so Initialize() threw and
// CEncoder logged a side channel that failed to start. It is Level_8k3 now
// (4K-1 stops at 4096, just under what ALVR uses).
//
// Measured with MainConcept's own sample at 4288x1664, bpp 1.2, 12 frames:
//
//                      GPU        CPU
//    speed         146 fps     93 fps
//    avg response   6.8 ms    10.8 ms
//    threads             0         72
//    process CPU     4.6 %     23.0 %
//
// MC_MEMORY_TYPE_CUDA (real device pointers, zero-copy) is still unexplored;
// the frame currently goes through MC_MEMORY_TYPE_SYSTEM, i.e. the readback.
// ---------------------------------------------------------------------------

#ifdef ALVR_JPEGXS_MAINCONCEPT

#include "VideoEncoder.h"
#include "alvr_server/StreamDumper.h"
#include "d3d-render-utils/RenderUtils.h"
#include "shared/d3drender.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <d3d11.h>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <wrl.h>

extern "C" {
#include "enc_jxs.h"
}

class VideoEncoderJpegXsMainConcept : public VideoEncoder {
public:
    // sideChannelOnly: dump to disk and never touch the network, which is what
    // this class did exclusively until it grew a real output path. false makes it
    // the encoder whose bitstream actually goes to the client.
    VideoEncoderJpegXsMainConcept(
        std::shared_ptr<CD3DRender> d3dRender,
        uint32_t width,
        uint32_t height,
        bool sideChannelOnly = true
    );
    ~VideoEncoderJpegXsMainConcept();

    void Initialize() override;
    void Shutdown() override;

    // Non-blocking: only issues an async GPU CopyResource and hands a staging
    // slot off to the encode thread -- mirrors VideoEncoderJpegXs::Transmit()
    // exactly (see its own comment for why: this runs on CEncoder's shared
    // render thread, right after the real encoder's own Transmit() call, so
    // it must never block on the encode itself).
    void Transmit(
        ID3D11Texture2D* pTexture,
        uint64_t presentationTime,
        uint64_t targetTimestampNs,
        bool insertIDR
    ) override;

private:
    // One-time setup of the shader, the half-width YUY2 render target and the
    // staging ring, run on the first Transmit() call once the real source texture
    // (and therefore its real format and size) is known -- mirrors
    // VideoEncoderJpegXs EnsureGpuPipeline() for the same reason.
    bool EnsureGpuPipeline(ID3D11Texture2D* sourceTexture);

    // One full-screen draw into the half-width YUY2 target. See m_yuy2Tex.
    void RenderYuy2(ID3D11DeviceContext* context);

    // Two draws: Y at full size, then Cb and Cr together into two half-size
    // targets. Deliberately not one draw with three targets -- see
    // VideoEncoderJpegXs on why a shared render pass writes only part of Y.
    void RenderI420(ID3D11DeviceContext* context);

    // Single background thread: waits for a staging slot, Map()s it, does
    // per-row copy out of the YUY2 staging texture, calls MainConcept
    // encoder, and dumps the result via StreamDumper. Combines what
    // VideoEncoderJpegXs splits into ReadbackWorkerLoop/EncodeWorkerLoop --
    // MainConcept's own PutFrameV() call already does the heavy lifting
    // (including its own internal GPU dispatch when gpu_accel=1), so there
    // is no separate CPU decode-and-transform stage worth overlapping here.
    void EncodeWorkerLoop();

    std::shared_ptr<CD3DRender> m_d3dRender;

    uint32_t m_width;
    uint32_t m_height;

    // RGB -> YUY2 on the GPU. The colour maths is VideoEncoderJpegXs's, coefficients
    // and sRGB handling unchanged -- that shader is the one that was debugged until
    // the picture stopped coming out green, and none of that reasoning changes here.
    // Only the packing differs: one render target of HALF the width in
    // R8G8B8A8_UNORM, where each texel holds (Y0, Cb, Y1, Cr) for two neighbouring
    // pixels. In memory that is R,G,B,A = byte 0,1,2,3, which is exactly YUY2, so
    // the readback is a plain per-row memcpy with nothing to unpack.
    Microsoft::WRL::ComPtr<ID3D11VertexShader> m_quadVertexShader;
    Microsoft::WRL::ComPtr<ID3D11InputLayout> m_quadInputLayout;
    Microsoft::WRL::ComPtr<ID3D11Buffer> m_quadVertexBuffer;
    Microsoft::WRL::ComPtr<ID3D11PixelShader> m_yuy2PixelShader;
    // The 4:2:0 alternative: three planes, Y at full size and Cb/Cr at half in both
    // directions, which is what SVT-JPEG-XS produces and what the client decodes
    // today. Only MainConcept's CPU engine accepts it -- the GPU engine's input
    // formats stop at 4:2:2 -- so this path trades the GPU offload for needing no
    // client change at all. Same two shader entry points VideoEncoderJpegXs uses.
    Microsoft::WRL::ComPtr<ID3D11PixelShader> m_yPixelShader;
    Microsoft::WRL::ComPtr<ID3D11PixelShader> m_uvPixelShader;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> m_yTex, m_cbTex, m_crTex;
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> m_rtvY, m_rtvCb, m_rtvCr;
    D3D11_VIEWPORT m_uvViewport {};
    // true: GPU engine, YUY2. false: CPU engine, I420. Follows the "Use CUDA"
    // setting, so switching between them costs a Dashboard toggle, not a rebuild.
    bool m_useGpuEngine = true;
    uint32_t m_chromaWidth = 0;
    uint32_t m_chromaHeight = 0;
    Microsoft::WRL::ComPtr<ID3D11Buffer> m_yuvParamBuffer;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_sourceSRV;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> m_yuy2Tex;
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> m_rtvYuy2;
    D3D11_VIEWPORT m_viewport {};
    // Bound explicitly every draw: this shares the immediate context with
    // FrameRender's own rendering, so an inherited scissor rect or cull mode could
    // otherwise discard every pixel and leave the target at its all-zero creation
    // content. Same reasoning as VideoEncoderJpegXs.
    Microsoft::WRL::ComPtr<ID3D11RasterizerState> m_rasterizerState;
    Microsoft::WRL::ComPtr<ID3D11BlendState> m_blendState;
    Microsoft::WRL::ComPtr<ID3D11DepthStencilState> m_depthStencilState;

    // [plane][slot]. The YUY2 path uses plane 0 only; the I420 path uses all three.
    Microsoft::WRL::ComPtr<ID3D11Texture2D> m_stagingTex[3][2];
    DXGI_FORMAT m_sourceFormat = DXGI_FORMAT_UNKNOWN;
    bool m_gpuPipelineReady = false;
    bool m_gpuSetupFailed = false;

    uint64_t m_stagingTimestampNs[2] = { 0, 0 };
    bool m_stagingHasData[2] = { false, false };
    bool m_stagingReserved[2] = { false, false };
    int m_stagingWriteIndex = 0;

    mc_enc_jxs_settings_t m_settings {};
    mc_callbacks_t m_callbacks {};

    // One encoder instance per column, exactly as VideoEncoderJpegXs does it, and
    // for the same reason: the client detects an "ALVS" wrapper and decodes the
    // columns inside it, so a stream meant for that client has to be shaped that
    // way. Note the reason is NOT the same as it was for SVT-JPEG-XS, where the
    // split existed to get two encoder instances working in parallel -- this one
    // reaches 146fps on a whole frame in a single instance. Here the split is
    // purely about matching what the client expects.
    //
    // Size 1 when the split is off; the wrapper is then skipped entirely and the
    // single codestream goes out as-is, which the client also accepts.
    std::vector<mc_enc_jxs_t*> m_columnEncs;
    uint32_t m_columnsNum = 1;
    uint32_t m_columnWidth = 0;
    std::vector<std::vector<uint8_t>> m_columnBitstreamBufs;
    // magic + columns_num + per-column length table + concatenated codestreams.
    std::vector<uint8_t> m_wrapperBuf;

    const bool m_sideChannelOnly = true;
    // Set after the one-time empty SetVideoConfigNals() on the first frame that
    // actually goes out. A member, not a function-local static, so it resets if
    // this encoder is destroyed and a new one built for a fresh stream.
    bool m_sentConfig = false;

    // Readback scratch: interleaved B,G,R,X, one row at a time from the
    // mapped staging texture -- touched only by EncodeWorkerLoop.
    // One buffer per plane. The YUY2 path fills only the first, as packed
    // (Y0,Cb,Y1,Cr); the I420 path fills all three as separate planes, which is
    // what mc_frame_t wants anyway -- plane[0..2] with their own strides.
    std::vector<uint8_t> m_scratch[3];

    std::unique_ptr<StreamDumper> m_streamDumper;

    int m_pendingStagingIndex = -1;
    uint64_t m_pendingTimestampNs = 0;
    bool m_hasPendingFrame = false;
    std::mutex m_mutex;
    std::condition_variable m_cv;

    std::atomic_bool m_exiting { false };
    std::thread m_encodeThread;
};

#endif // ALVR_JPEGXS_MAINCONCEPT
