#pragma once
#include "CEncoder.h"
#include "alvr_server/PoseHistory.h"
#include "alvr_server/Utils.h"
#include "alvr_server/openvr_driver_wrap.h"

#include "alvr_server/Settings.h"

#include <mutex>

class OvrDirectModeComponent : public vr::IVRDriverDirectModeComponent {
public:
    OvrDirectModeComponent(
        std::shared_ptr<CD3DRender> pD3DRender, std::shared_ptr<PoseHistory> poseHistory
    );

    void SetEncoder(std::shared_ptr<CEncoder> pEncoder);

    /** Specific to Oculus compositor support, textures supplied must be created using this method.
     */
    virtual void CreateSwapTextureSet(
        uint32_t unPid,
        const SwapTextureSetDesc_t* pSwapTextureSetDesc,
        SwapTextureSet_t* pOutSwapTextureSet
    );

    /** Used to textures created using CreateSwapTextureSet.  Only one of the set's handles needs to
     * be used to destroy the entire set. */
    virtual void DestroySwapTextureSet(vr::SharedTextureHandle_t sharedTextureHandle);

    /** Used to purge all texture sets for a given process. */
    virtual void DestroyAllSwapTextureSets(uint32_t unPid);

    /** After Present returns, calls this to get the next index to use for rendering. */
    virtual void GetNextSwapTextureSetIndex(
        vr::SharedTextureHandle_t sharedTextureHandles[2], uint32_t (*pIndices)[2]
    );

    /** Call once per layer to draw for this frame.  One shared texture handle per eye.  Textures
     * must be created using CreateSwapTextureSet and should be alternated per frame.  Call Present
     * once all layers have been submitted. */
    virtual void SubmitLayer(const SubmitLayerPerEye_t (&perEye)[2]);

    /** Submits queued layers for display. */
    virtual void Present(vr::SharedTextureHandle_t syncTexture);

    /** Called after Present to allow driver to take more time until vsync after they've
     * successfully acquired the sync texture in Present.*/
    virtual void PostPresent();

    /** The compositor's only way to learn whether this driver is keeping up, and the only place it
     * tells the driver what it intends to do about it: it fills m_nReprojectionFlags with
     * VRCompositor_ReprojectionMotion_Enabled / _ForcedOn / _AppThrottled before the call and reads
     * the present counters back out afterwards. ALVR left this unimplemented, so the compositor has
     * never had a single frame statistic from this driver -- which is the most plausible reason
     * SteamVR's Motion Smoothing option is greyed out for it. Filling it in is a prerequisite for
     * finding out, not a guarantee that Motion Smoothing then works. */
    virtual void GetFrameTiming(vr::DriverDirectMode_FrameTiming* pFrameTiming);

    void CopyTexture(uint32_t layerCount);

private:
    std::shared_ptr<CD3DRender> m_pD3DRender;
    std::shared_ptr<CEncoder> m_pEncoder;
    std::shared_ptr<PoseHistory> m_poseHistory;

    // Resource for each process
    struct ProcessResource {
        ComPtr<ID3D11Texture2D> textures[3];
        HANDLE sharedHandles[3];
        uint32_t pid;
    };
    std::map<HANDLE, std::pair<ProcessResource*, int>> m_handleMap;

    static const int MAX_LAYERS = 10;
    int m_submitLayer;
    SubmitLayerPerEye_t m_submitLayers[MAX_LAYERS][2];
    vr::HmdQuaternion_t m_prevFramePoseRotation;
    vr::HmdQuaternion_t m_framePoseRotation;
    uint64_t m_targetTimestampNs;
    uint64_t m_prevTargetTimestampNs;

    // Frame-timing bookkeeping for GetFrameTiming, all touched under m_presentMutex.
    // m_framePresents counts Present() calls since the last GetFrameTiming, which is what
    // "number of times frame was presented" means for a driver that presents exactly once per
    // frame. Dropped frames are counted from the target timestamp not advancing: Present()
    // already detects that case for its own duplicate-frame check.
    uint32_t m_framePresents = 0;
    uint32_t m_frameDropped = 0;
    uint32_t m_lastReprojectionFlags = 0;

    std::mutex m_presentMutex;
};
