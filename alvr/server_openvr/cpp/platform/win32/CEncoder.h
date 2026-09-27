#pragma once
#include "shared/d3drender.h"

#include "shared/threadtools.h"

#include "FrameRender.h"
#include "VideoEncoder.h"
#include "VideoEncoderAMF.h"
#include "VideoEncoderNVENC.h"
#include "alvr_server/Utils.h"
#include <d3d11.h>
#include <d3d11_1.h>
#include <map>
#include <wincodec.h>
#include <wincodecsdk.h>
#include <wrl.h>
#ifdef ALVR_GPL
#include "VideoEncoderSW.h"
#endif
#include "alvr_server/IDRScheduler.h"
#ifdef ALVR_JPEGXS
#include "VideoEncoderJpegXs.h"
#endif
#ifdef ALVR_JPEGXS_MAINCONCEPT
#include "VideoEncoderJpegXsMainConcept.h"
#endif

using Microsoft::WRL::ComPtr;

//----------------------------------------------------------------------------
// Blocks on reading backbuffer from gpu, so WaitForPresent can return
// as soon as we know rendering made it this frame.  This step of the pipeline
// should run about 3ms per frame.
//----------------------------------------------------------------------------
class CEncoder : public CThread {
public:
    CEncoder();
    ~CEncoder();

    void Initialize(std::shared_ptr<CD3DRender> d3dRender);

    void SetViewsConfig(
        vr::HmdRect2_t projLeft,
        vr::HmdMatrix34_t eyeToHeadLeft,
        vr::HmdRect2_t projRight,
        vr::HmdMatrix34_t eyeToHeadRight
    );

    bool CopyToStaging(
        ID3D11Texture2D* pTexture[][2],
        vr::VRTextureBounds_t bounds[][2],
        vr::HmdMatrix34_t poses[],
        int layerCount,
        bool recentering,
        uint64_t presentationTime,
        uint64_t targetTimestampNs,
        const std::string& message,
        const std::string& debugText
    );

    virtual void Run();

    virtual void Stop();

    void NewFrameReady();

    void WaitForEncode();

    void OnStreamStart();

    void OnPacketLoss();

    void InsertIDR();

    void CaptureFrame();

private:
    CThreadEvent m_newFrameReady, m_encodeFinished;
    std::shared_ptr<VideoEncoder> m_videoEncoder;
    // Additive, dump-only JPEG XS instance -- constructed only when JPEG XS ISN'T the real
    // codec (m_videoEncoder above is a different encoder, e.g. HEVC) but the JPEG XS
    // "Dump stream to disk" setting is on anyway, for testing/comparison without touching
    // the real stream. Stored as the base type (like m_videoEncoder) so this header
    // doesn't need VideoEncoderJpegXs.h / doesn't need to know ALVR_JPEGXS's state --
    // construction (the only place the concrete type matters) happens in the .cpp, which
    // already includes it under that same #ifdef. Null whenever the additive instance
    // wasn't constructed (JPEG XS already the real codec, dump-to-disk off, or ALVR_JPEGXS
    // not compiled in).
    std::shared_ptr<VideoEncoder> m_jpegXsSideEncoder;
    // Second additive, dump-only side channel -- MainConcept's commercial JPEG XS SDK
    // (GPU/CUDA-accelerated), for comparing against SVT-JPEG-XS above and the real stream.
    // Same construction/lifetime pattern as m_jpegXsSideEncoder; see
    // VideoEncoderJpegXsMainConcept.h for why a separate class instead of extending
    // VideoEncoderJpegXs (different SDK, different color path).
    std::shared_ptr<VideoEncoder> m_mainConceptJxsSideEncoder;
    bool m_bExiting;
    uint64_t m_presentationTime;
    uint64_t m_targetTimestampNs;

    std::shared_ptr<FrameRender> m_FrameRender;

    IDRScheduler m_scheduler;
};
