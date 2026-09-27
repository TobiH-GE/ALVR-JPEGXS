#include "CEncoder.h"
#include "alvr_server/Settings.h"

CEncoder::CEncoder()
    : m_bExiting(false)
    , m_targetTimestampNs(0) {
    m_encodeFinished.Set();
}

CEncoder::~CEncoder() {
    if (m_videoEncoder) {
        m_videoEncoder->Shutdown();
        m_videoEncoder.reset();
    }
    if (m_jpegXsSideEncoder) {
        m_jpegXsSideEncoder->Shutdown();
        m_jpegXsSideEncoder.reset();
    }
    if (m_mainConceptJxsSideEncoder) {
        m_mainConceptJxsSideEncoder->Shutdown();
        m_mainConceptJxsSideEncoder.reset();
    }
}

void CEncoder::Initialize(std::shared_ptr<CD3DRender> d3dRender) {
    m_FrameRender = std::make_shared<FrameRender>(d3dRender);
    m_FrameRender->Startup();
    uint32_t encoderWidth, encoderHeight;
    m_FrameRender->GetEncodingResolution(&encoderWidth, &encoderHeight);

#ifdef ALVR_JPEGXS
    // Additive dump-only instance: independent of which encoder ends up being the real,
    // network-streamed one below -- only skipped when JPEG XS already IS that real encoder
    // (constructing a second one would be pure duplicate work for identical output).
    // Best-effort: a failure here logs a warning and leaves m_jpegXsSideEncoder null,
    // rather than affecting the real encoder selection below in any way.
    if (Settings::Instance().m_codec != ALVR_CODEC_JPEGXS && Settings::Instance().m_jpegXsDumpToDisk) {
        try {
            Debug("Try to use VideoEncoderJpegXs (additive side-channel dump).\n");
            auto sideEncoder = std::make_shared<VideoEncoderJpegXs>(
                d3dRender, encoderWidth, encoderHeight, /*sideChannelOnly=*/true
            );
            sideEncoder->Initialize();
            m_jpegXsSideEncoder = sideEncoder;
        } catch (Exception e) {
            Warn(
                "CEncoder: additive VideoEncoderJpegXs side-channel failed to initialize (%s) -- "
                "continuing without the side-channel dump.\n",
                e.what()
            );
        }
    }
#endif

#ifdef ALVR_JPEGXS_MAINCONCEPT
    // Second additive side channel -- see VideoEncoderJpegXsMainConcept.h. Reuses the same
    // "Dump stream to disk" setting as the SVT-JPEG-XS side channel above (both run together
    // when it's on) rather than adding a new Dashboard toggle for what's currently a one-off
    // comparison test. Independent of whether the SVT-JPEG-XS side channel above succeeded.
    // Skipped when MainConcept already IS the real encoder below: a second instance
    // would be duplicate work for identical output, same reasoning as the SVT-JPEG-XS
    // side channel above.
    if (Settings::Instance().m_jpegXsDumpToDisk
        && !(Settings::Instance().m_codec == ALVR_CODEC_JPEGXS
             && Settings::Instance().m_jpegXsUseMainConcept)) {
        try {
            Debug("Try to use VideoEncoderJpegXsMainConcept (additive side-channel dump).\n");
            auto mcSideEncoder = std::make_shared<VideoEncoderJpegXsMainConcept>(
                d3dRender, encoderWidth, encoderHeight
            );
            mcSideEncoder->Initialize();
            m_mainConceptJxsSideEncoder = mcSideEncoder;
        } catch (Exception e) {
            Warn(
                "CEncoder: additive VideoEncoderJpegXsMainConcept side-channel failed to "
                "initialize (%s) -- continuing without it.\n",
                e.what()
            );
        }
    }
#endif

#ifdef ALVR_JPEGXS
    // JPEG XS is a software encoder, always "available" if compiled in --
    // unlike AMF/NVENC below, this is a deterministic selection based on
    // the user's actual codec choice, not a hardware-availability probe.
    // Checked first so it takes priority when selected; falls through to
    // the normal AMF/NVENC(/SW) probing chain below on failure (or if not
    // selected), exactly like those backends' own failures already do.
    if (Settings::Instance().m_codec == ALVR_CODEC_JPEGXS) {
#ifdef ALVR_JPEGXS_MAINCONCEPT
        // MainConcept's encoder in place of SVT-JPEG-XS, when asked for. Same
        // fall-through on failure as everything else here: SVT-JPEG-XS below, then
        // the hardware probing chain. Note this changes the wire format as well as
        // the encoder -- see VideoEncoderJpegXsMainConcept.h.
        if (Settings::Instance().m_jpegXsUseMainConcept) {
            try {
                Debug("Try to use VideoEncoderJpegXsMainConcept (selected codec).\n");
                auto mcEncoder = std::make_shared<VideoEncoderJpegXsMainConcept>(
                    d3dRender, encoderWidth, encoderHeight, /*sideChannelOnly=*/false
                );
                mcEncoder->Initialize();
                m_videoEncoder = mcEncoder;
                return;
            } catch (Exception e) {
                Error(
                    "CEncoder: VideoEncoderJpegXsMainConcept failed to initialize (%s), falling "
                    "back to SVT-JPEG-XS.\n",
                    e.what()
                );
            }
        }
#endif
        try {
            Debug("Try to use VideoEncoderJpegXs (selected codec).\n");
            m_videoEncoder
                = std::make_shared<VideoEncoderJpegXs>(d3dRender, encoderWidth, encoderHeight);
            m_videoEncoder->Initialize();
            return;
        } catch (Exception e) {
            Error(
                "CEncoder: VideoEncoderJpegXs failed to initialize (%s), falling back to "
                "hardware encoder probing.\n",
                e.what()
            );
        }
    }
#else
    if (Settings::Instance().m_codec == ALVR_CODEC_JPEGXS) {
        Warn("CEncoder: JpegXs codec selected but server was built without ALVR_JPEGXS support; "
             "falling back to hardware encoder probing.\n");
    }
#endif

    Exception vceException;
    Exception nvencException;
#ifdef ALVR_GPL
    Exception swException;
    if (Settings::Instance().m_force_sw_encoding) {
        try {
            Debug("Try to use VideoEncoderSW.\n");
            m_videoEncoder
                = std::make_shared<VideoEncoderSW>(d3dRender, encoderWidth, encoderHeight);
            m_videoEncoder->Initialize();
            return;
        } catch (Exception e) {
            swException = e;
        }
    }
#endif

    try {
        Debug("Try to use VideoEncoderAMF.\n");
        m_videoEncoder = std::make_shared<VideoEncoderAMF>(d3dRender, encoderWidth, encoderHeight);
        m_videoEncoder->Initialize();
        return;
    } catch (Exception e) {
        vceException = e;
    }
    try {
        Debug("Try to use VideoEncoderNVENC.\n");
        m_videoEncoder
            = std::make_shared<VideoEncoderNVENC>(d3dRender, encoderWidth, encoderHeight);
        m_videoEncoder->Initialize();
        return;
    } catch (Exception e) {
        nvencException = e;
    }
#ifdef ALVR_GPL
    try {
        Debug("Try to use VideoEncoderSW.\n");
        m_videoEncoder = std::make_shared<VideoEncoderSW>(d3dRender, encoderWidth, encoderHeight);
        m_videoEncoder->Initialize();
        return;
    } catch (Exception e) {
        swException = e;
    }
    throw MakeException(
        "All VideoEncoder are not available. VCE: %s, NVENC: %s, SW: %s",
        vceException.what(),
        nvencException.what(),
        swException.what()
    );
#else
    throw MakeException(
        "All VideoEncoder are not available. VCE: %s, NVENC: %s",
        vceException.what(),
        nvencException.what()
    );
#endif
}

void CEncoder::SetViewsConfig(
    vr::HmdRect2_t projLeft,
    vr::HmdMatrix34_t eyeToHeadLeft,
    vr::HmdRect2_t projRight,
    vr::HmdMatrix34_t eyeToHeadRight
) {
    m_FrameRender->SetViewsConfig(projLeft, eyeToHeadLeft, projRight, eyeToHeadRight);
}

bool CEncoder::CopyToStaging(
    ID3D11Texture2D* pTexture[][2],
    vr::VRTextureBounds_t bounds[][2],
    vr::HmdMatrix34_t poses[],
    int layerCount,
    bool recentering,
    uint64_t presentationTime,
    uint64_t targetTimestampNs,
    const std::string& message,
    const std::string& debugText
) {
    m_presentationTime = presentationTime;
    m_targetTimestampNs = targetTimestampNs;
    m_FrameRender->Startup();

    m_FrameRender->RenderFrame(
        pTexture, bounds, poses, layerCount, recentering, message, debugText
    );
    return true;
}

void CEncoder::Run() {
    Debug("CEncoder: Start thread. Id=%d\n", GetCurrentThreadId());
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_MOST_URGENT);

    while (!m_bExiting) {
        m_newFrameReady.Wait();
        if (m_bExiting)
            break;

        if (m_FrameRender->GetTexture()) {
            bool insertIDR = m_scheduler.CheckIDRInsertion();

            m_videoEncoder->Transmit(
                m_FrameRender->GetTexture().Get(),
                m_presentationTime,
                m_targetTimestampNs,
                insertIDR
            );

            if (m_jpegXsSideEncoder) {
                m_jpegXsSideEncoder->Transmit(
                    m_FrameRender->GetTexture().Get(),
                    m_presentationTime,
                    m_targetTimestampNs,
                    insertIDR
                );
            }

            if (m_mainConceptJxsSideEncoder) {
                m_mainConceptJxsSideEncoder->Transmit(
                    m_FrameRender->GetTexture().Get(),
                    m_presentationTime,
                    m_targetTimestampNs,
                    insertIDR
                );
            }
        }

        m_encodeFinished.Set();
    }
}

void CEncoder::Stop() {
    m_bExiting = true;
    m_newFrameReady.Set();
    Join();
    m_FrameRender.reset();
}

void CEncoder::NewFrameReady() {
    m_encodeFinished.Reset();
    m_newFrameReady.Set();
}

void CEncoder::WaitForEncode() { m_encodeFinished.Wait(); }

void CEncoder::OnStreamStart() { m_scheduler.OnStreamStart(); }

void CEncoder::OnPacketLoss() { m_scheduler.OnPacketLoss(); }

void CEncoder::InsertIDR() { m_scheduler.InsertIDR(); }

void CEncoder::CaptureFrame() { }
