#include "VideoEncoderJpegXsMainConcept.h"

#ifdef ALVR_JPEGXS_MAINCONCEPT

#include "alvr_server/Logger.h"
#include "alvr_server/bindings.h"
#include "alvr_server/Settings.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <chrono>
#include <string>
#include <windows.h>
#include <d3dcompiler.h>
#pragma comment(lib, "d3dcompiler.lib")

extern "C" {
#include "mcfourcc.h"
}

namespace {

// Loads MainConcept's GPU engine DLLs by full path, once, before the encoder is
// created.
//
// Without this the GPU engine fails with "JPEG-XS GPU encoder cannot be loaded.
// Please check that CUDA is available in the system. Error: [system:126]" --
// ERROR_MOD_NOT_FOUND, which is a missing DLL and not a missing CUDA. The chain
// demo_enc_jxs.dll -> demo_jpegxs_gpu_loader.dll -> jpegxs-gpu-dll.dll ->
// cudart64_12.dll is loaded at runtime by bare name, and Windows resolves a bare
// name against the directory of the EXE -- here SteamVR's vrserver.exe -- not
// against the directory of the DLL doing the loading. These sit next to
// driver_alvr_server.dll instead, so nothing on that search path ever finds them.
// MainConcept's own sample works only because it runs from the directory its DLLs
// are in.
//
// Loading each one by full path first fixes it without touching the process-wide
// search order: a later LoadLibrary by bare name matches an already-loaded module
// by its base name and returns that one. Order matters, dependencies first.
void PreloadMainConceptDlls() {
    static bool done = false;
    if (done) {
        return;
    }
    done = true;

    HMODULE self = nullptr;
    if (!GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&PreloadMainConceptDlls),
            &self
        )) {
        Warn("VideoEncoderJpegXsMainConcept: cannot locate own module; leaving the DLL search "
             "path alone.\n");
        return;
    }

    wchar_t path[MAX_PATH];
    DWORD len = GetModuleFileNameW(self, path, MAX_PATH);
    if (len == 0 || len >= MAX_PATH) {
        Warn("VideoEncoderJpegXsMainConcept: cannot read own module path.\n");
        return;
    }
    std::wstring dir(path, len);
    size_t slash = dir.find_last_of(L"\\/");
    if (slash == std::wstring::npos) {
        return;
    }
    dir.resize(slash + 1);

    // cudart before the GPU engine that imports it, the GPU engine before the
    // loader that opens it.
    static const wchar_t* kDlls[] = {
        L"cudart64_12.dll",
        L"jpegxs-dll.dll",
        L"jpegxs-gpu-dll.dll",
        L"demo_jpegxs_gpu_loader.dll",
    };
    for (const wchar_t* name : kDlls) {
        std::wstring full = dir + name;
        if (LoadLibraryW(full.c_str()) == nullptr) {
            // Not fatal on its own: the CPU engine needs only jpegxs-dll.dll, and
            // the encoder reports its own failure clearly enough if a GPU one is
            // missing. Logged so that failure has a cause next to it.
            Warn(
                "VideoEncoderJpegXsMainConcept: could not preload %ls (error %lu).\n",
                name,
                GetLastError()
            );
        }
    }
}

// Routes MainConcept's printf-style logging callbacks into ALVR's own
// Error/Warn/Info -- see mc_callbacks_t's doc comment (mccallbacks.h) for
// why these exist (memory hooks, license lookup, message printing); only
// the message-printing ones matter here, the rest are left null (the demo
// library falls back to its own defaults, same as running without a
// license file -- expected for an evaluation build).
void McErrPrintf(mc_context_t, const char* fmt, ...) {
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    Error("VideoEncoderJpegXsMainConcept: %s", buf);
}
void McWrnPrintf(mc_context_t, const char* fmt, ...) {
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    Warn("VideoEncoderJpegXsMainConcept: %s", buf);
}
void McInfPrintf(mc_context_t, const char* fmt, ...) {
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    Info("VideoEncoderJpegXsMainConcept: %s", buf);
}

} // namespace

VideoEncoderJpegXsMainConcept::VideoEncoderJpegXsMainConcept(
    std::shared_ptr<CD3DRender> d3dRender, uint32_t width, uint32_t height, bool sideChannelOnly
)
    : m_d3dRender(d3dRender)
    , m_width(width)
    , m_height(height)
    , m_sideChannelOnly(sideChannelOnly) { }

VideoEncoderJpegXsMainConcept::~VideoEncoderJpegXsMainConcept() { Shutdown(); }

void VideoEncoderJpegXsMainConcept::RenderYuy2(ID3D11DeviceContext* context) {
    // Every piece of state is set explicitly. This context is shared with
    // FrameRender's own rendering, so nothing about it can be assumed to be at
    // defaults when this draw runs -- an inherited scissor rect or cull mode would
    // silently discard every pixel and leave the target at its creation content.
    context->RSSetViewports(1, &m_viewport);
    context->RSSetState(m_rasterizerState.Get());
    const float blendFactor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    context->OMSetBlendState(m_blendState.Get(), blendFactor, 0xFFFFFFFF);
    context->OMSetDepthStencilState(m_depthStencilState.Get(), 0);

    context->IASetInputLayout(m_quadInputLayout.Get());
    UINT stride = 4 * sizeof(float) + 2 * sizeof(float) + sizeof(uint32_t);
    UINT offsetBytes = 0;
    ID3D11Buffer* vb = m_quadVertexBuffer.Get();
    context->IASetVertexBuffers(0, 1, &vb, &stride, &offsetBytes);
    context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

    context->VSSetShader(m_quadVertexShader.Get(), nullptr, 0);
    context->PSSetShader(m_yuy2PixelShader.Get(), nullptr, 0);
    ID3D11Buffer* cb = m_yuvParamBuffer.Get();
    context->PSSetConstantBuffers(0, 1, &cb);

    // Render targets cleared before the source is bound as a shader input -- see
    // RenderI420 for what happens otherwise. Only then may our own target go on.
    ID3D11RenderTargetView* clearRtv[] = { nullptr };
    context->OMSetRenderTargets(1, clearRtv, nullptr);
    ID3D11ShaderResourceView* srv = m_sourceSRV.Get();
    context->PSSetShaderResources(0, 1, &srv);
    ID3D11RenderTargetView* rtv = m_rtvYuy2.Get();
    context->OMSetRenderTargets(1, &rtv, nullptr);

    context->Draw(4, 0);

    // Unbind the source: it is FrameRender's composition texture, and leaving it
    // bound as an SRV would stop the next frame from rendering into it.
    ID3D11ShaderResourceView* nullSrv = nullptr;
    context->PSSetShaderResources(0, 1, &nullSrv);
    ID3D11RenderTargetView* nullRtv = nullptr;
    context->OMSetRenderTargets(1, &nullRtv, nullptr);
}

void VideoEncoderJpegXsMainConcept::RenderI420(ID3D11DeviceContext* context) {
    // Same explicit state as RenderYuy2, same reason.
    context->RSSetState(m_rasterizerState.Get());
    const float blendFactor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    context->OMSetBlendState(m_blendState.Get(), blendFactor, 0xFFFFFFFF);
    context->OMSetDepthStencilState(m_depthStencilState.Get(), 0);

    context->IASetInputLayout(m_quadInputLayout.Get());
    UINT stride = 4 * sizeof(float) + 2 * sizeof(float) + sizeof(uint32_t);
    UINT offsetBytes = 0;
    ID3D11Buffer* vb = m_quadVertexBuffer.Get();
    context->IASetVertexBuffers(0, 1, &vb, &stride, &offsetBytes);
    context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    context->VSSetShader(m_quadVertexShader.Get(), nullptr, 0);
    ID3D11Buffer* cb = m_yuvParamBuffer.Get();
    context->PSSetConstantBuffers(0, 1, &cb);

    // Unbind every render target BEFORE binding the source as a shader input.
    // sourceTexture is ALVR's compositor output, which FrameRender leaves bound as
    // an active render target on this same immediate context. D3D11's
    // resource-hazard tracker then silently forces the shader-resource slot back to
    // nullptr -- no error, no exception, Sample() just returns zero. Binding the SRV
    // first is what made this encoder send a perfectly valid, perfectly black
    // stream; VideoEncoderJpegXs carries the same line and the comment explaining
    // it, from the time this cost a debugging session and produced green.
    ID3D11RenderTargetView* clearRtv[] = { nullptr };
    context->OMSetRenderTargets(1, clearRtv, nullptr);
    ID3D11ShaderResourceView* srv = m_sourceSRV.Get();
    context->PSSetShaderResources(0, 1, &srv);

    // Luma, full size.
    ID3D11RenderTargetView* rtvY = m_rtvY.Get();
    context->OMSetRenderTargets(1, &rtvY, nullptr);
    context->RSSetViewports(1, &m_viewport);
    context->PSSetShader(m_yPixelShader.Get(), nullptr, 0);
    context->Draw(4, 0);

    // Chroma, half size, both planes from one draw. Two single-channel targets
    // rather than one interleaved one, so nothing has to be de-interleaved later.
    ID3D11RenderTargetView* uvTargets[2] = { m_rtvCb.Get(), m_rtvCr.Get() };
    context->OMSetRenderTargets(2, uvTargets, nullptr);
    context->RSSetViewports(1, &m_uvViewport);
    context->PSSetShader(m_uvPixelShader.Get(), nullptr, 0);
    context->Draw(4, 0);

    ID3D11ShaderResourceView* nullSrv = nullptr;
    context->PSSetShaderResources(0, 1, &nullSrv);
    ID3D11RenderTargetView* nullRtvs[2] = { nullptr, nullptr };
    context->OMSetRenderTargets(2, nullRtvs, nullptr);
}

void VideoEncoderJpegXsMainConcept::Initialize() {
    // Breadcrumbs at Warn level, on purpose. CEncoder::Initialize() runs while the
    // client connection is being set up, so anything slow or stuck in here shows up
    // as a connection that times out rather than as an encoder that failed -- which
    // is exactly what three live attempts looked like, with the encoder never
    // mentioned in any log. Info() goes to ALVR's own log channel and never reaches
    // vrserver.txt; Warn() does, and that is the log available without asking anyone
    // to turn on "Log to disk" first. Cheap: a handful of lines once per stream.
    const auto initStart = std::chrono::steady_clock::now();
    auto step = [&initStart](const char* what) {
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - initStart
        )
                        .count();
        Warn("VideoEncoderJpegXsMainConcept: init step '%s' reached at %.0fms\n", what, ms);
    };

    step("entered Initialize");
    PreloadMainConceptDlls();
    step("DLLs preloaded");

    memset(&m_callbacks, 0, sizeof(m_callbacks));
    m_callbacks.err_printf = McErrPrintf;
    m_callbacks.wrn_printf = McWrnPrintf;
    m_callbacks.inf_printf = McInfPrintf;

    // Column split, matching what the client expects on the wire (see the header).
    // The width has to divide evenly: a JPEG XS codestream carries its own width, so
    // uneven columns would decode fine individually but reassemble misaligned.
    m_columnsNum = Settings::Instance().m_jpegXsColumnsNum;
    if (m_columnsNum < 1) {
        m_columnsNum = 1;
    }
    while (m_columnsNum > 1 && (m_width % m_columnsNum) != 0) {
        m_columnsNum--;
    }
    m_columnWidth = m_width / m_columnsNum;

    jxsOutVideoDefaults(&m_settings);
    m_settings.width = (int32_t)m_columnWidth;
    m_settings.height = (int32_t)m_height;
    m_chromaWidth = m_width / 2;
    m_chromaHeight = m_height / 2;

    // Which engine, and therefore which colour format. The GPU engine's input
    // formats stop at 4:2:2, so 4:2:0 -- the format the client already decodes --
    // is only reachable on the CPU engine. Hung on the existing "Use CUDA" setting
    // rather than a new one, because that is exactly the choice being made.
    //
    // Measured at 4288x1664, bpp 1.2, 60 frames, with MainConcept's own sample:
    //
    //                    GPU / YUY2      CPU / I420
    //   speed               231 fps         185 fps
    //   avg response         4.3 ms          5.4 ms
    //   threads                   0              72
    //   process CPU           4.2 %          30.9 %
    //
    // So the CPU engine costs back roughly what moving JPEG XS onto the GPU saved,
    // and buys a stream this client can decode today.
    m_useGpuEngine = Settings::Instance().m_jpegXsUseCuda;
    m_settings.four_cc = m_useGpuEngine ? FOURCC_YUY2 : FOURCC_I420;
    m_settings.frameRate = 90.0;
    // Without this the encoder never starts at ALVR's resolution. jxsOutVideoDefaults
    // leaves the level at 2K-1, which caps the image at 2048 wide, and jxsOutVideoNew
    // then fails with "width must be <= 2048 ... for the 2k-1 level" -- so this class
    // threw in Initialize() and CEncoder logged it as a side channel that failed to
    // start. 8K-3 is the smallest level that covers 4288x1664 at 120Hz (4K-1 stops at
    // 4096 wide, just under what ALVR uses). Level_Unrestricted works too but writes
    // no level into the codestream, which a strict decoder is entitled to dislike.
    m_settings.level = Level_8k3;
    // GPU/CUDA acceleration. This was off, on the conclusion that MainConcept's
    // jpegxs-gpu-dll.dll hangs during CUDA init on an RTX 5090. It does not: the DLL
    // carries no Blackwell cubin (highest is sm_80) but it does carry PTX, so the
    // driver compiles 255 kernels on first use. That takes about a second, once, and
    // is then cached in %APPDATA%\NVIDIA\ComputeCache -- the second run starts in
    // 15ms. Measured here with MainConcept's own sample at 4288x1664, bpp 1.2:
    // 146fps and 4.6% process CPU on the GPU, against 93fps and 23% on the CPU.
    m_settings.gpu_accel = m_useGpuEngine ? 1 : 0;
    m_settings.device_id = 0;
    // Reuse the same Dashboard setting SVT-JPEG-XS's side channel uses, so
    // both side-by-side dumps encode at the same target bpp and are a fair
    // comparison against each other (and against SVT-JPEG-XS's live bpp).
    m_settings.bpp = (double)Settings::Instance().m_jpegXsBitsPerPixel;

    step("settings built");
    if (jxsOutVideoChkSettings(&m_callbacks, &m_settings)) {
        throw MakeException("VideoEncoderJpegXsMainConcept: jxsOutVideoChkSettings failed.");
    }

    // One instance per column. They are independent: each encodes an ordinary
    // JPEG XS image of m_columnWidth, reading its own slice straight out of the
    // full-width scratch by pointer offset (see EncodeWorkerLoop).
    step("settings checked");
    m_columnEncs.resize(m_columnsNum, nullptr);
    for (uint32_t i = 0; i < m_columnsNum; i++) {
        // The first of these is where the CUDA context comes up and, on a cold
        // compute cache, where 255 PTX kernels get compiled for this GPU.
        m_columnEncs[i] = jxsOutVideoNew(&m_callbacks, &m_settings);
        step("jxsOutVideoNew returned");
        if (!m_columnEncs[i]) {
            throw MakeException(
                "VideoEncoderJpegXsMainConcept: jxsOutVideoNew failed for column %u.", i
            );
        }
    }

    // Generous fixed size (not lossless-frame-size-derived like
    // VideoEncoderJpegXs's dst_buffer_size, since bpp here is always < the
    // uncompressed BGRX size) -- one column's BGRX bytes is already an upper
    // bound with a lot of headroom for a compressed JPEG XS frame, plus fixed
    // slack for headers. Per column, so concurrent encoders write to disjoint
    // buffers.
    m_columnBitstreamBufs.resize(m_columnsNum);
    for (uint32_t i = 0; i < m_columnsNum; i++) {
        m_columnBitstreamBufs[i].resize((size_t)m_columnWidth * m_height * 4 + 65536);
    }
    m_wrapperBuf.resize(4 + 4 + (size_t)m_columnsNum * 4
                        + ((size_t)m_columnWidth * m_height * 4 + 65536) * m_columnsNum);

    // A side-channel instance exists only to dump, so it always does. A real one
    // dumps only when asked: writing hundreds of megabytes a minute is not something
    // to do as a side effect of picking an encoder.
    if (m_sideChannelOnly || Settings::Instance().m_jpegXsDumpToDisk) {
        m_streamDumper
            = std::make_unique<StreamDumper>(std::string("jpegxs"), ".jxs", "mainconcept_gpu");
    }

    step("buffers allocated");
    m_exiting = false;
    m_encodeThread = std::thread([this] { EncodeWorkerLoop(); });
    step("Initialize complete");

    Info(
        "VideoEncoderJpegXsMainConcept: initialized (%ux%u, gpu_accel=%d, device_id=0, "
        "four_cc=BGR4, bpp=%.2f) -- additive side-channel, dumping to disk.\n",
        m_width,
        m_height,
        m_settings.gpu_accel,
        m_settings.bpp
    );
}

void VideoEncoderJpegXsMainConcept::Shutdown() {
    if (m_exiting.exchange(true)) {
        return; // already shut down
    }
    m_cv.notify_all();
    if (m_encodeThread.joinable()) {
        m_encodeThread.join();
    }
    for (size_t i = 0; i < m_columnEncs.size(); i++) {
        if (m_columnEncs[i]) {
            jxsOutVideoDone(m_columnEncs[i], 0);
            jxsOutVideoFree(m_columnEncs[i]);
            m_columnEncs[i] = nullptr;
        }
    }
    if (m_streamDumper) {
        m_streamDumper->Close();
        m_streamDumper.reset();
    }
}

bool VideoEncoderJpegXsMainConcept::EnsureGpuPipeline(ID3D11Texture2D* sourceTexture) {
    if (m_gpuPipelineReady) {
        return true;
    }
    if (m_gpuSetupFailed) {
        return false;
    }
    try {
        auto* device = m_d3dRender->GetDevice();

        D3D11_TEXTURE2D_DESC srcDesc {};
        sourceTexture->GetDesc(&srcDesc);
        m_sourceFormat = srcDesc.Format;

        // Half-width target: one texel per PAIR of source pixels, holding
        // (Y0, Cb, Y1, Cr). R8G8B8A8_UNORM puts those at bytes 0..3, which is YUY2.
        // Creates a render target plus its two staging copies in one go, since every
        // plane on either path needs exactly that.
        auto makePlane = [&](uint32_t w,
                             uint32_t h,
                             DXGI_FORMAT fmt,
                             int stagingPlane,
                             Microsoft::WRL::ComPtr<ID3D11Texture2D>& tex,
                             Microsoft::WRL::ComPtr<ID3D11RenderTargetView>& rtv) {
            D3D11_TEXTURE2D_DESC d {};
            d.Width = w;
            d.Height = h;
            d.MipLevels = 1;
            d.ArraySize = 1;
            d.Format = fmt;
            d.SampleDesc.Count = 1;
            d.Usage = D3D11_USAGE_DEFAULT;
            d.BindFlags = D3D11_BIND_RENDER_TARGET;
            OK_OR_THROW(
                device->CreateTexture2D(&d, nullptr, &tex),
                "VideoEncoderJpegXsMainConcept: failed to create a render target."
            );
            OK_OR_THROW(
                device->CreateRenderTargetView(tex.Get(), nullptr, &rtv),
                "VideoEncoderJpegXsMainConcept: failed to create a render target view."
            );

            D3D11_TEXTURE2D_DESC s = d;
            s.Usage = D3D11_USAGE_STAGING;
            s.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            s.BindFlags = 0;
            for (int i = 0; i < 2; i++) {
                OK_OR_THROW(
                    device->CreateTexture2D(&s, nullptr, &m_stagingTex[stagingPlane][i]),
                    "VideoEncoderJpegXsMainConcept: failed to create a staging texture."
                );
            }
        };

        if (m_useGpuEngine) {
            // One texel per pair of pixels: (Y0, Cb, Y1, Cr) at bytes 0..3, which is
            // YUY2 as the encoder wants it.
            const uint32_t packedWidth = m_width / 2;
            makePlane(packedWidth, m_height, DXGI_FORMAT_R8G8B8A8_UNORM, 0, m_yuy2Tex, m_rtvYuy2);
            m_viewport = { 0.0f, 0.0f, (float)packedWidth, (float)m_height, 0.0f, 1.0f };
            m_scratch[0].resize((size_t)m_width * m_height * 2);
        } else {
            // Three separate planes, which is both what 4:2:0 is and what
            // mc_frame_t wants -- plane[0..2] with their own strides, no packing.
            makePlane(m_width, m_height, DXGI_FORMAT_R8_UNORM, 0, m_yTex, m_rtvY);
            makePlane(m_chromaWidth, m_chromaHeight, DXGI_FORMAT_R8_UNORM, 1, m_cbTex, m_rtvCb);
            makePlane(m_chromaWidth, m_chromaHeight, DXGI_FORMAT_R8_UNORM, 2, m_crTex, m_rtvCr);
            m_viewport = { 0.0f, 0.0f, (float)m_width, (float)m_height, 0.0f, 1.0f };
            m_uvViewport
                = { 0.0f, 0.0f, (float)m_chromaWidth, (float)m_chromaHeight, 0.0f, 1.0f };
            m_scratch[0].resize((size_t)m_width * m_height);
            m_scratch[1].resize((size_t)m_chromaWidth * m_chromaHeight);
            m_scratch[2].resize((size_t)m_chromaWidth * m_chromaHeight);
        }

        // ALVR's own quad vertex shader needs a real bound vertex buffer with
        // FrameRender's SimpleVertex layout -- it is not SV_VertexID-procedural.
        // Same self-contained quad VideoEncoderJpegXs builds, and for the same
        // reason: we run on a different thread and call context than FrameRender, so
        // there is no leftover vertex buffer to inherit.
        {
            std::vector<uint8_t> quadShaderCSO(
                QUAD_SHADER_CSO_PTR, QUAD_SHADER_CSO_PTR + QUAD_SHADER_CSO_LEN
            );
            m_quadVertexShader = d3d_render_utils::CreateVertexShader(device, quadShaderCSO);

            D3D11_INPUT_ELEMENT_DESC layout[] = {
                { "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA,
                  0 },
                { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 16, D3D11_INPUT_PER_VERTEX_DATA, 0 },
                { "VIEW", 0, DXGI_FORMAT_R32_UINT, 0, 24, D3D11_INPUT_PER_VERTEX_DATA, 0 },
            };
            OK_OR_THROW(
                device->CreateInputLayout(
                    layout, 3, quadShaderCSO.data(), quadShaderCSO.size(), &m_quadInputLayout
                ),
                "VideoEncoderJpegXsMainConcept: failed to create the quad input layout."
            );

            struct QuadVertex {
                float pos[4];
                float tex[2];
                uint32_t view;
            };
            QuadVertex vertices[4] = {
                { { -1.0f, 1.0f, 0.5f, 1.0f }, { 0.0f, 1.0f }, 0 },
                { { 1.0f, -1.0f, 0.5f, 1.0f }, { 1.0f, 0.0f }, 0 },
                { { 1.0f, 1.0f, 0.5f, 1.0f }, { 1.0f, 1.0f }, 0 },
                { { -1.0f, -1.0f, 0.5f, 1.0f }, { 0.0f, 0.0f }, 0 },
            };
            D3D11_BUFFER_DESC vbDesc {};
            vbDesc.Usage = D3D11_USAGE_IMMUTABLE;
            vbDesc.ByteWidth = sizeof(vertices);
            vbDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
            D3D11_SUBRESOURCE_DATA vbData {};
            vbData.pSysMem = vertices;
            OK_OR_THROW(
                device->CreateBuffer(&vbDesc, &vbData, &m_quadVertexBuffer),
                "VideoEncoderJpegXsMainConcept: failed to create the quad vertex buffer."
            );
        }

        // BT.709, full range, and the sRGB question handled exactly as
        // VideoEncoderJpegXs handles it: the composition texture is fully-typed
        // _SRGB, so an SRV cannot suppress the automatic gamma decode. The shader
        // re-encodes instead, told by offset.w whether it needs to.
        struct YUVParams {
            float offset[4];
            float yCoeff[4];
            float uCoeff[4];
            float vCoeff[4];
        };
        const bool sourceNeedsSrgbEncode = srcDesc.Format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
            || srcDesc.Format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB
            || srcDesc.Format == DXGI_FORMAT_B8G8R8X8_UNORM_SRGB;
        YUVParams paramStruct { { 0.0f, 0.5f, 0.5f, sourceNeedsSrgbEncode ? 1.0f : 0.0f },
                                { 0.2126000f, 0.7152000f, 0.0722000f, 0.0f },
                                { -0.1145721f, -0.3854279f, 0.5000000f, 0.0f },
                                { 0.5000000f, -0.4541529f, -0.0458471f, 0.0f } };
        D3D11_BUFFER_DESC cbDesc {};
        cbDesc.Usage = D3D11_USAGE_IMMUTABLE;
        cbDesc.ByteWidth = sizeof(YUVParams);
        cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA cbData {};
        cbData.pSysMem = &paramStruct;
        OK_OR_THROW(
            device->CreateBuffer(&cbDesc, &cbData, &m_yuvParamBuffer),
            "VideoEncoderJpegXsMainConcept: failed to create the YUV parameter buffer."
        );

        // Load() rather than Sample(): the two source pixels behind a packed texel
        // are exact texels, and asking for them by index removes every question
        // about filter taps and sample positions.
        static const char* kYuy2ShaderSource = R"HLSL(
            cbuffer YUVParams {
                float4 offset;
                float4 yCoeff;
                float4 uCoeff;
                float4 vCoeff;
            };

            Texture2D<float4> sourceTexture;

            float3 reencodeSrgbIfNeeded(float3 rgb) {
                float3 lo = rgb * 12.92;
                float3 hi = 1.055 * pow(max(rgb, 0.0), 1.0 / 2.4) - 0.055;
                float3 encoded = lerp(lo, hi, step(0.0031308, rgb));
                return offset.w > 0.5 ? encoded : rgb;
            }

            float4 mainYUY2(float4 pos : SV_Position) : SV_Target0 {
                int2 t = int2(pos.xy);
                float3 a = reencodeSrgbIfNeeded(sourceTexture.Load(int3(2 * t.x, t.y, 0)).rgb);
                float3 b = reencodeSrgbIfNeeded(sourceTexture.Load(int3(2 * t.x + 1, t.y, 0)).rgb);
                // 4:2:2 keeps both luma samples and one chroma pair; averaging the
                // two pixels is the honest downsample, not picking the left one.
                float3 m = (a + b) * 0.5;
                float y0 = dot(a, yCoeff.rgb) + offset.x;
                float y1 = dot(b, yCoeff.rgb) + offset.x;
                float cb = dot(m, uCoeff.rgb) + offset.y;
                float cr = dot(m, vCoeff.rgb) + offset.z;
                return float4(y0, cb, y1, cr);
            }

            // 4:2:0: Y at full size in one draw, Cb and Cr together at half size
            // in another. Lifted from VideoEncoderJpegXs, which writes exactly
            // these three planes for SVT-JPEG-XS.
            SamplerState bilinearSampler {
                Filter = MIN_MAG_LINEAR_MIP_POINT;
                AddressU = CLAMP;
                AddressV = CLAMP;
            };

            float4 mainY(float2 uv : TEXCOORD0) : SV_Target0 {
                float3 rgb = reencodeSrgbIfNeeded(sourceTexture.Sample(bilinearSampler, uv).rgb);
                return float4(dot(rgb, yCoeff.rgb) + offset.x, 0, 0, 1);
            }

            struct UVOutput {
                float cb : SV_Target0;
                float cr : SV_Target1;
            };

            UVOutput mainUV(float2 uv : TEXCOORD0) {
                float3 rgb = reencodeSrgbIfNeeded(sourceTexture.Sample(bilinearSampler, uv).rgb);
                UVOutput result;
                result.cb = dot(rgb, uCoeff.rgb) + offset.y;
                result.cr = dot(rgb, vCoeff.rgb) + offset.z;
                return result;
            }
        )HLSL";

        auto compileShader = [&](const char* entryPoint, ID3D11PixelShader** outShader) {
            Microsoft::WRL::ComPtr<ID3DBlob> bytecode;
            Microsoft::WRL::ComPtr<ID3DBlob> errors;
            HRESULT hr = D3DCompile(
                kYuy2ShaderSource,
                strlen(kYuy2ShaderSource),
                nullptr,
                nullptr,
                nullptr,
                entryPoint,
                "ps_5_0",
                0,
                0,
                &bytecode,
                &errors
            );
            if (FAILED(hr)) {
                throw MakeException(
                    "VideoEncoderJpegXsMainConcept: D3DCompile failed for %s: %s",
                    entryPoint,
                    errors ? (const char*)errors->GetBufferPointer() : "no compiler output"
                );
            }
            OK_OR_THROW(
                device->CreatePixelShader(
                    bytecode->GetBufferPointer(), bytecode->GetBufferSize(), nullptr, outShader
                ),
                "VideoEncoderJpegXsMainConcept: failed to create a pixel shader."
            );
        };
        if (m_useGpuEngine) {
            compileShader("mainYUY2", m_yuy2PixelShader.GetAddressOf());
        } else {
            compileShader("mainY", m_yPixelShader.GetAddressOf());
            compileShader("mainUV", m_uvPixelShader.GetAddressOf());
        }

        OK_OR_THROW(
            device->CreateShaderResourceView(sourceTexture, nullptr, &m_sourceSRV),
            "VideoEncoderJpegXsMainConcept: failed to create the source SRV."
        );

        {
            D3D11_RASTERIZER_DESC rd {};
            rd.FillMode = D3D11_FILL_SOLID;
            rd.CullMode = D3D11_CULL_NONE;
            rd.DepthClipEnable = FALSE;
            rd.ScissorEnable = FALSE;
            OK_OR_THROW(
                device->CreateRasterizerState(&rd, &m_rasterizerState),
                "VideoEncoderJpegXsMainConcept: failed to create the rasterizer state."
            );

            D3D11_BLEND_DESC bd {};
            bd.RenderTarget[0].BlendEnable = FALSE;
            bd.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
            OK_OR_THROW(
                device->CreateBlendState(&bd, &m_blendState),
                "VideoEncoderJpegXsMainConcept: failed to create the blend state."
            );

            D3D11_DEPTH_STENCIL_DESC dsd {};
            dsd.DepthEnable = FALSE;
            dsd.StencilEnable = FALSE;
            OK_OR_THROW(
                device->CreateDepthStencilState(&dsd, &m_depthStencilState),
                "VideoEncoderJpegXsMainConcept: failed to create the depth-stencil state."
            );
        }


        m_gpuPipelineReady = true;
        Warn(
            "VideoEncoderJpegXsMainConcept: pipeline ready (%ux%u, %s engine, %s, source "
            "format=%d, sRGB re-encode %s).\n",
            m_width,
            m_height,
            m_useGpuEngine ? "GPU" : "CPU",
            m_useGpuEngine ? "YUY2 4:2:2" : "I420 4:2:0",
            (int)m_sourceFormat,
            sourceNeedsSrgbEncode ? "on" : "off"
        );
        return true;
    } catch (const std::exception& e) {
        Error(
            "VideoEncoderJpegXsMainConcept: staging pipeline setup failed, disabling: %s\n",
            e.what()
        );
        m_gpuSetupFailed = true;
        return false;
    }
}

void VideoEncoderJpegXsMainConcept::Transmit(
    ID3D11Texture2D* sourceTexture,
    uint64_t /*presentationTime*/,
    uint64_t targetTimestampNs,
    bool /*insertIDR*/
) {
    if (m_columnEncs.empty() || !m_columnEncs[0] || sourceTexture == nullptr) {
        return;
    }
    if (!EnsureGpuPipeline(sourceTexture)) {
        return;
    }

    auto* context = m_d3dRender->GetContext();

    int writeIndex = m_stagingWriteIndex;
    int readIndex = 1 - m_stagingWriteIndex;

    bool writeSlotReserved;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        writeSlotReserved = m_stagingReserved[writeIndex];
    }
    if (writeSlotReserved) {
        // Encode thread hasn't Map()'d this slot yet -- drop this frame
        // rather than stall the shared render thread (same policy as
        // VideoEncoderJpegXs's own mailbox).
        return;
    }

    if (m_useGpuEngine) {
        RenderYuy2(context);
        context->CopyResource(m_stagingTex[0][writeIndex].Get(), m_yuy2Tex.Get());
    } else {
        RenderI420(context);
        context->CopyResource(m_stagingTex[0][writeIndex].Get(), m_yTex.Get());
        context->CopyResource(m_stagingTex[1][writeIndex].Get(), m_cbTex.Get());
        context->CopyResource(m_stagingTex[2][writeIndex].Get(), m_crTex.Get());
    }
    context->Flush();
    m_stagingTimestampNs[writeIndex] = targetTimestampNs;
    m_stagingHasData[writeIndex] = true;

    if (m_stagingHasData[readIndex]) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_hasPendingFrame) {
            m_pendingStagingIndex = readIndex;
            m_pendingTimestampNs = m_stagingTimestampNs[readIndex];
            m_hasPendingFrame = true;
            m_stagingReserved[readIndex] = true;
            m_cv.notify_one();
        }
    }

    m_stagingWriteIndex = readIndex;
}

void VideoEncoderJpegXsMainConcept::EncodeWorkerLoop() {
    while (true) {
        int stagingIndex = -1;
        uint64_t timestampNs = 0;
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
            m_hasPendingFrame = false;
        }

        {
            auto* context = m_d3dRender->GetContext();
            // One plane on the YUY2 path, three on the 4:2:0 one. Every plane is a
            // per-row memcpy: the shader already produced the exact byte layout the
            // encoder wants, only the staging row pitch differs from a tight one.
            const uint32_t planeCount = m_useGpuEngine ? 1u : 3u;
            for (uint32_t plane = 0; plane < planeCount; plane++) {
                const size_t rowBytes = m_useGpuEngine ? (size_t)m_width * 2
                    : (plane == 0 ? (size_t)m_width : (size_t)m_chromaWidth);
                const uint32_t rows = (!m_useGpuEngine && plane > 0) ? m_chromaHeight : m_height;

                D3D11_MAPPED_SUBRESOURCE mappedPlane;
                HRESULT hrPlane = context->Map(
                    m_stagingTex[plane][stagingIndex].Get(), 0, D3D11_MAP_READ, 0, &mappedPlane
                );
                if (FAILED(hrPlane)) {
                    Error(
                        "VideoEncoderJpegXsMainConcept: staging Map() failed for plane %u "
                        "hr=0x%08lx\n",
                        plane,
                        (unsigned long)hrPlane
                    );
                    continue;
                }
                const uint8_t* srcPlane = reinterpret_cast<const uint8_t*>(mappedPlane.pData);
                for (uint32_t y = 0; y < rows; y++) {
                    std::memcpy(
                        m_scratch[plane].data() + (size_t)y * rowBytes,
                        srcPlane + (size_t)y * mappedPlane.RowPitch,
                        rowBytes
                    );
                }
                context->Unmap(m_stagingTex[plane][stagingIndex].Get(), 0);
            }
        }
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stagingReserved[stagingIndex] = false;
        }

        // Zero-copy column crop: BGRX is interleaved, so column i starts at
        // i * m_columnWidth * 4 bytes into every row and the stride stays at the
        // FULL row width. Each encoder then reads m_columnWidth pixels per row from
        // its own offset -- no memcpy needed to isolate a column, the same trick
        // VideoEncoderJpegXs uses on its planar buffers.
        std::vector<uint32_t> columnSizes(m_columnsNum, 0);
        bool encodeOk = true;
        for (uint32_t i = 0; i < m_columnsNum; i++) {
            mc_frame_t frame {};
            frame.width = m_columnWidth;
            frame.height = m_height;
            frame.four_cc = m_settings.four_cc;
            // Two bytes per pixel, and a column boundary always lands on a whole
            // (Y,Cb,Y,Cr) group because the column width is even.
            if (m_useGpuEngine) {
                frame.plane[0] = m_scratch[0].data() + (size_t)i * m_columnWidth * 2;
                frame.stride[0] = (int32_t)(m_width * 2);
            }
            else {
                // Three planes, each cropped by the same pointer-offset trick:
                // luma by the column width, chroma by half of it, strides left at
                // the full frame so each encoder reads only its own columns.
                frame.plane[0] = m_scratch[0].data() + (size_t)i * m_columnWidth;
                frame.plane[1] = m_scratch[1].data() + (size_t)i * (m_columnWidth / 2);
                frame.plane[2] = m_scratch[2].data() + (size_t)i * (m_columnWidth / 2);
                frame.stride[0] = (int32_t)m_width;
                frame.stride[1] = (int32_t)m_chromaWidth;
                frame.stride[2] = (int32_t)m_chromaWidth;
            }

            uint32_t bytesWritten = (uint32_t)m_columnBitstreamBufs[i].size();
            int32_t err = jxsOutVideoPutFrameV(
                m_columnEncs[i], &frame, m_columnBitstreamBufs[i].data(), &bytesWritten, nullptr
            );
            if (err || bytesWritten == 0) {
                LogPeriod(
                    "VideoEncoderJpegXsMainConcept.Encode",
                    "jxsOutVideoPutFrameV failed for column %u (err=%d, bytes=%u)\n",
                    i,
                    err,
                    bytesWritten
                );
                encodeOk = false;
                break;
            }
            columnSizes[i] = bytesWritten;
        }
        if (!encodeOk) {
            continue;
        }

        // Once, so a black picture can be told apart from an empty one without
        // guessing: these are the bytes that actually leave the encoder.
        static bool reportedFirstFrame = false;
        if (!reportedFirstFrame) {
            reportedFirstFrame = true;
            uint32_t total = 0;
            for (uint32_t i = 0; i < m_columnsNum; i++) {
                total += columnSizes[i];
            }
            Warn("VideoEncoderJpegXsMainConcept: first frame encoded, %u column(s), %u bytes "
                 "total; luma scratch starts %02x %02x %02x %02x, chroma %02x %02x\n",
                 m_columnsNum,
                 total,
                 m_scratch[0].size() > 3 ? m_scratch[0][0] : 0,
                 m_scratch[0].size() > 3 ? m_scratch[0][1] : 0,
                 m_scratch[0].size() > 3 ? m_scratch[0][2] : 0,
                 m_scratch[0].size() > 3 ? m_scratch[0][3] : 0,
                 (!m_useGpuEngine && m_scratch[1].size() > 0) ? m_scratch[1][0] : 0,
                 (!m_useGpuEngine && m_scratch[2].size() > 0) ? m_scratch[2][0] : 0);
        }

        uint8_t* sendBuffer = nullptr;
        uint32_t sendSize = 0;
        if (m_columnsNum <= 1) {
            // No wrapper: a plain single codestream, which the client accepts too.
            sendBuffer = m_columnBitstreamBufs[0].data();
            sendSize = columnSizes[0];
        } else {
            // "ALVS" wrapper: magic + columns_num + per-column length table +
            // concatenated codestreams, byte for byte the layout
            // VideoEncoderJpegXs writes, because the client already parses that one.
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

        if (m_streamDumper) {
            m_streamDumper->WritePacket(sendBuffer, sendSize);
            m_streamDumper->OnFrameEncoded(sendSize, true);
        }

        // The real output. Skipped entirely for a side-channel instance: only the
        // one encoder that is actually the stream may call these, or a side instance
        // running alongside another codec would corrupt its packet sequence.
        if (!m_sideChannelOnly) {
            if (!m_sentConfig) {
                m_sentConfig = true;
                SetVideoConfigNals(nullptr, 0, ALVR_CODEC_JPEGXS);
            }
            VideoSend(timestampNs, sendBuffer, (int)sendSize, /*isIdr=*/true, 0, /*packetsPerFrame=*/1, /*column=*/0);
        }
    }
}

#endif // ALVR_JPEGXS_MAINCONCEPT
