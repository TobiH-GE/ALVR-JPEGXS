use crate::{
    bitrate::BitrateManager,
    hand_gestures::HandGestureManager,
    input_mapping::ButtonMappingManager,
    sockets::WelcomeSocket,
    statistics::StatisticsManager,
    tracking::{self, TrackingManager},
    ConnectionContext, ServerCoreEvent, ViewsConfig, FILESYSTEM_LAYOUT, SESSION_MANAGER,
};
use alvr_adb::{WiredConnection, WiredConnectionStatus};
use alvr_common::{
    con_bail, dbg_connection, debug, error,
    glam::{Quat, UVec2, Vec2, Vec3},
    info,
    parking_lot::{Condvar, Mutex, RwLock},
    settings_schema::Switch,
    warn, AnyhowToCon, ConResult, ConnectionError, ConnectionState, LifecycleState, Pose,
    BUTTON_INFO, CONTROLLER_PROFILE_INFO, QUEST_CONTROLLER_PROFILE_PATH,
};
use alvr_events::{AdbEvent, ButtonEvent, EventType};
use alvr_packets::{
    ClientConnectionResult, ClientControlPacket, ClientListAction, ClientStatistics,
    NegotiatedStreamingConfig, RealTimeConfig, ReservedClientControlPacket, ServerControlPacket,
    Tracking, VideoPacketHeader, AUDIO, HAPTICS, STATISTICS, TRACKING, VIDEO,
};
use alvr_session::{
    BodyTrackingBDConfig, BodyTrackingSinkConfig, CodecType, ControllersEmulationMode, FrameSize,
    H264Profile, OpenvrConfig, SessionConfig, SocketProtocol,
};
use alvr_sockets::{
    PeerType, ProtoControlSocket, StreamSocketBuilder, CONTROL_PORT, KEEPALIVE_INTERVAL,
    KEEPALIVE_TIMEOUT, WIRED_CLIENT_HOSTNAME,
};
use std::{
    collections::HashMap,
    net::{IpAddr, Ipv4Addr},
    process::Command,
    sync::{mpsc::RecvTimeoutError, Arc},
    thread,
    time::{Duration, Instant},
};

const RETRY_CONNECT_MIN_INTERVAL: Duration = Duration::from_secs(1);
const HANDSHAKE_ACTION_TIMEOUT: Duration = Duration::from_secs(2);
pub const STREAMING_RECV_TIMEOUT: Duration = Duration::from_millis(500);
const REAL_TIME_UPDATE_INTERVAL: Duration = Duration::from_secs(1);

const MAX_UNREAD_PACKETS: usize = 10; // Applies per stream

// Video packets dropped because the send queue to the socket thread was full. Counted in
// send_video_nal (lib.rs), reported by the send diagnostics below.
pub static VIDEO_QUEUE_DROPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
// Packets currently waiting between the encoder and the send thread, and frames refused as a
// whole because the queue already held about two frames. See send_video_nal in lib.rs.
pub static VIDEO_QUEUE_LEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
// Bumped for every session. The previous session's send thread can still be draining its own
// channel while this one resets the counter, and its decrements would then take the fresh count
// below zero -- which, unsigned, becomes enormous and makes the admission refuse every frame
// from then on. That is exactly what killed the stream on 2026-09-22: the encoder kept running,
// the send thread never received anything again. A thread only counts down for its own session.
pub static VIDEO_QUEUE_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static VIDEO_FRAMES_REFUSED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
// See send_video_nal in lib.rs: frames carrying the same timestamp as the frame before them.
pub static VIDEO_REPEATED_TIMESTAMPS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

// CPU time this thread has used so far (kernel + user). Compared with the wall time spent in
// send it tells computing apart from waiting: a send blocked by a full transmit queue (e.g.
// Ethernet flow control from the receiver) costs wall time but almost no CPU.
// Returns (kernel, user). 2026-09-22 it showed the slow sends (24-31 us a shard) are all CPU
// time, not waiting. Windows charges interrupt/DPC time to whichever thread it interrupted, so
// the split plus the processor number below tell a network-interrupt collision on this core
// apart from an expensive stack or filter driver.
#[cfg(windows)]
fn current_thread_cpu_time() -> (Duration, Duration) {
    use std::ffi::c_void;
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentThread() -> *mut c_void;
        fn GetThreadTimes(
            thread: *mut c_void,
            creation: *mut u64,
            exit: *mut u64,
            kernel: *mut u64,
            user: *mut u64,
        ) -> i32;
    }
    let (mut c, mut e, mut k, mut u) = (0u64, 0u64, 0u64, 0u64);
    unsafe { GetThreadTimes(GetCurrentThread(), &mut c, &mut e, &mut k, &mut u) };
    // 100 ns units
    (Duration::from_nanos(k * 100), Duration::from_nanos(u * 100))
}
#[cfg(not(windows))]
fn current_thread_cpu_time() -> (Duration, Duration) {
    (Duration::ZERO, Duration::ZERO)
}

#[cfg(windows)]
fn current_processor() -> u32 {
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcessorNumber() -> u32;
    }
    unsafe { GetCurrentProcessorNumber() }
}
#[cfg(not(windows))]
fn current_processor() -> u32 {
    0
}


// How busy the machine and this process are, for the per-second send diagnostics: the encoder,
// SteamVR and the game share both the CPU and the GPU, and a run that looks bad is worth nothing
// without knowing whether either was saturated at the time.
#[cfg(windows)]
mod load {
    use std::ffi::c_void;

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct FileTime {
        low: u32,
        high: u32,
    }
    impl FileTime {
        fn as_u64(self) -> u64 {
            (self.high as u64) << 32 | self.low as u64
        }
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GetSystemTimes(idle: *mut FileTime, kernel: *mut FileTime, user: *mut FileTime) -> i32;
        fn GetCurrentProcess() -> *mut c_void;
        fn GetCurrentProcessId() -> u32;
        fn GetProcessTimes(
            process: *mut c_void,
            creation: *mut FileTime,
            exit: *mut FileTime,
            kernel: *mut FileTime,
            user: *mut FileTime,
        ) -> i32;
    }

    #[derive(Default)]
    pub struct Load {
        prev_idle: u64,
        prev_busy: u64,
        prev_process: u64,
        cores: f64,
        gpu: Option<Nvml>,
    }

    impl Load {
        pub fn new() -> Self {
            Load {
                cores: std::thread::available_parallelism().map_or(1.0, |n| n.get() as f64),
                gpu: Nvml::open(),
                ..Default::default()
            }
        }

        // "cpu 34 % of the machine, vrserver 12 % (2.9 of 24 cores) | gpu 61 %, vrserver 22 %"
        pub fn line(&mut self) -> String {
            let (mut idle, mut kernel, mut user) = Default::default();
            let (mut c, mut e, mut pkernel, mut puser) = Default::default();
            unsafe {
                GetSystemTimes(&mut idle, &mut kernel, &mut user);
                GetProcessTimes(GetCurrentProcess(), &mut c, &mut e, &mut pkernel, &mut puser);
            }
            // The kernel total already includes idle, so busy is kernel + user - idle.
            let idle = idle.as_u64();
            let busy = kernel.as_u64() + user.as_u64() - idle;
            let process = pkernel.as_u64() + puser.as_u64();

            let d_idle = idle.saturating_sub(self.prev_idle) as f64;
            let d_busy = busy.saturating_sub(self.prev_busy) as f64;
            let d_process = process.saturating_sub(self.prev_process) as f64;
            self.prev_idle = idle;
            self.prev_busy = busy;
            self.prev_process = process;

            let total = d_idle + d_busy;
            let (machine, mine, cores_used) = if total > 0.0 {
                (
                    100.0 * d_busy / total,
                    100.0 * d_process / total,
                    d_process / total * self.cores,
                )
            } else {
                (0.0, 0.0, 0.0)
            };

            let gpu = match self.gpu.as_mut().map(|g| g.sample()) {
                Some(Some((whole, Some(ours)))) => {
                    format!("gpu {whole} % total, this process {ours} %, everything else {} %",
                            whole.saturating_sub(ours))
                }
                // The whole-GPU number is there but no per-process sample came back: with the
                // encoder on the CPU that is the honest answer, the process really does no
                // compute on the GPU beyond the frame readback.
                Some(Some((whole, None))) => format!("gpu {whole} % total, this process not measurable"),
                _ => "gpu n/a".to_string(),
            };
            format!(
                "cpu {machine:.0} % of the machine, this process {mine:.0} % ({cores_used:.1} of {:.0} cores) | {gpu}",
                self.cores
            )
        }
    }

    // NVML, loaded at runtime so a machine without the NVIDIA driver just reports "n/a".
    type NvmlDevice = *mut c_void;
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct Utilization {
        gpu: u32,
        memory: u32,
    }
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct ProcessSample {
        pid: u32,
        timestamp: u64,
        sm_util: u32,
        mem_util: u32,
        enc_util: u32,
        dec_util: u32,
    }

    pub struct Nvml {
        device: NvmlDevice,
        utilization: unsafe extern "C" fn(NvmlDevice, *mut Utilization) -> i32,
        process_utilization: unsafe extern "C" fn(NvmlDevice, *mut ProcessSample, *mut u32, u64) -> i32,
        pid: u32,
    }

    impl Nvml {
        fn open() -> Option<Self> {
            use std::ffi::CString;
            #[link(name = "kernel32")]
            extern "system" {
                fn LoadLibraryA(name: *const u8) -> *mut c_void;
                fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
            }
            unsafe {
                let name = CString::new("nvml.dll").ok()?;
                let mut module = LoadLibraryA(name.as_ptr() as *const u8);
                if module.is_null() {
                    // Not on PATH on every driver version; this is where the installer puts it.
                    let full = CString::new(
                        "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll",
                    )
                    .ok()?;
                    module = LoadLibraryA(full.as_ptr() as *const u8);
                }
                if module.is_null() {
                    return None;
                }
                let get = |symbol: &str| -> Option<*mut c_void> {
                    let symbol = CString::new(symbol).ok()?;
                    let address = GetProcAddress(module, symbol.as_ptr() as *const u8);
                    (!address.is_null()).then_some(address)
                };
                let init: unsafe extern "C" fn() -> i32 = std::mem::transmute(get("nvmlInit_v2")?);
                if init() != 0 {
                    return None;
                }
                let handle: unsafe extern "C" fn(u32, *mut NvmlDevice) -> i32 =
                    std::mem::transmute(get("nvmlDeviceGetHandleByIndex_v2")?);
                let mut device: NvmlDevice = std::ptr::null_mut();
                if handle(0, &mut device) != 0 {
                    return None;
                }
                Some(Nvml {
                    device,
                    utilization: std::mem::transmute(get("nvmlDeviceGetUtilizationRates")?),
                    process_utilization: std::mem::transmute(get("nvmlDeviceGetProcessUtilization")?),
                    pid: GetCurrentProcessId(),
                })
            }
        }

        // (whole GPU %, this process's share of the SMs % if NVML reported one).
        fn sample(&mut self) -> Option<(u32, Option<u32>)> {
            let mut whole = Utilization::default();
            if unsafe { (self.utilization)(self.device, &mut whole) } != 0 {
                return None;
            }
            let mut samples = [ProcessSample::default(); 128];
            let mut count = samples.len() as u32;
            // NVML wants microseconds since the epoch and returns only samples newer than that.
            // Asking from one second ago keeps it to this window; carrying the newest timestamp
            // forward instead can miss the buffer entirely once a call comes back empty.
            let since = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_micros() as u64)
                .saturating_sub(1_000_000);
            let rc = unsafe {
                (self.process_utilization)(self.device, samples.as_mut_ptr(), &mut count, since)
            };
            let mut ours = None;
            if rc == 0 {
                for sample in samples.iter().take(count as usize) {
                    if sample.pid == self.pid {
                        ours = Some(ours.unwrap_or(0u32).max(sample.sm_util));
                    }
                }
            }
            Some((whole.gpu, ours))
        }
    }
}

#[cfg(not(windows))]
mod load {
    #[derive(Default)]
    pub struct Load;
    impl Load {
        pub fn new() -> Self {
            Load
        }
        pub fn line(&mut self) -> String {
            String::new()
        }
    }
}

// One socket send per 1.4 KB shard is ~135,000 sends a second at 4288x1664. On a hybrid CPU
// (i9-12900KF, Balanced plan) that costs 3.3 us per shard on a performance core but 9.5 us on
// an efficiency core and 23.6 us on a power-throttled one (EcoQoS). Windows moved this thread
// between them on its own, and at the slow levels it managed only 30-70 % of the packets
// (2026-09-22). So: performance cores only, no power throttling, raised priority.
#[cfg(windows)]
fn keep_send_thread_on_performance_cores() -> String {
    use std::ffi::c_void;
    type Handle = *mut c_void;
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentThread() -> Handle;
        fn GetCurrentProcess() -> Handle;
        fn SetThreadPriority(thread: Handle, priority: i32) -> i32;
        fn SetThreadInformation(thread: Handle, class: i32, info: *const c_void, size: u32) -> i32;
        fn GetSystemCpuSetInformation(
            info: *mut u8,
            length: u32,
            returned: *mut u32,
            process: Handle,
            flags: u32,
        ) -> i32;
        fn SetThreadSelectedCpuSets(thread: Handle, ids: *const u32, count: u32) -> i32;
    }
    const THREAD_PRIORITY_HIGHEST: i32 = 2;
    const THREAD_POWER_THROTTLING: i32 = 3;
    const THREAD_POWER_THROTTLING_EXECUTION_SPEED: u32 = 1;
    #[repr(C)]
    struct PowerThrottlingState {
        version: u32,
        control_mask: u32,
        state_mask: u32,
    }

    let mut notes = Vec::new();
    unsafe {
        let thread = GetCurrentThread();

        let ok = SetThreadPriority(thread, THREAD_PRIORITY_HIGHEST) != 0;
        notes.push(format!("priority highest {}", if ok { "ok" } else { "FAILED" }));

        // Control bit set, state bit clear: this thread opts out of EcoQoS.
        let state = PowerThrottlingState {
            version: 1,
            control_mask: THREAD_POWER_THROTTLING_EXECUTION_SPEED,
            state_mask: 0,
        };
        let ok = SetThreadInformation(
            thread,
            THREAD_POWER_THROTTLING,
            &state as *const _ as *const c_void,
            std::mem::size_of::<PowerThrottlingState>() as u32,
        ) != 0;
        notes.push(format!("power throttling off {}", if ok { "ok" } else { "FAILED" }));

        // SYSTEM_CPU_SET_INFORMATION entries: Size u32 @0, Type u32 @4, Id u32 @8,
        // EfficiencyClass u8 @18. A higher class is a faster core.
        let mut length = 0u32;
        GetSystemCpuSetInformation(std::ptr::null_mut(), 0, &mut length, GetCurrentProcess(), 0);
        let mut buffer = vec![0u8; length as usize];
        if length == 0
            || GetSystemCpuSetInformation(
                buffer.as_mut_ptr(),
                length,
                &mut length,
                GetCurrentProcess(),
                0,
            ) == 0
        {
            notes.push("cpu sets unavailable".into());
            return notes.join(", ");
        }
        let mut sets = Vec::new();
        let mut offset = 0usize;
        while offset + 20 <= length as usize {
            let size = u32::from_le_bytes(buffer[offset..offset + 4].try_into().unwrap()) as usize;
            if size < 20 {
                break;
            }
            let kind = u32::from_le_bytes(buffer[offset + 4..offset + 8].try_into().unwrap());
            if kind == 0 {
                let id = u32::from_le_bytes(buffer[offset + 8..offset + 12].try_into().unwrap());
                sets.push((id, buffer[offset + 18]));
            }
            offset += size;
        }
        let fastest = sets.iter().map(|&(_, class)| class).max().unwrap_or(0);
        let slowest = sets.iter().map(|&(_, class)| class).min().unwrap_or(0);
        if fastest == slowest {
            notes.push(format!("{} logical cpus, all one class: no pinning", sets.len()));
        } else {
            let ids: Vec<u32> = sets
                .iter()
                .filter(|&&(_, class)| class == fastest)
                .map(|&(id, _)| id)
                .collect();
            let ok = SetThreadSelectedCpuSets(thread, ids.as_ptr(), ids.len() as u32) != 0;
            notes.push(format!(
                "pinned to {} of {} logical cpus (efficiency class {}) {}",
                ids.len(),
                sets.len(),
                fastest,
                if ok { "ok" } else { "FAILED" }
            ));
        }
    }
    notes.join(", ")
}

pub struct VideoPacket {
    pub header: VideoPacketHeader,
    pub payload: Vec<u8>,
}

fn align32(value: f32) -> u32 {
    ((value / 32.).floor() * 32.) as u32
}

fn is_streaming(client_hostname: &str) -> bool {
    SESSION_MANAGER
        .read()
        .client_list()
        .get(client_hostname)
        .is_some_and(|c| c.connection_state == ConnectionState::Streaming)
}

pub fn contruct_openvr_config(session: &SessionConfig) -> OpenvrConfig {
    let old_config = session.openvr_config.clone();
    let settings = session.to_settings();

    let mut controller_is_tracker = false;
    let mut controller_profile = 0;
    let mut use_separate_hand_trackers = false;
    let controllers_enabled = if let Switch::Enabled(config) = settings.headset.controllers {
        controller_is_tracker =
            matches!(config.emulation_mode, ControllersEmulationMode::ViveTracker);
        // These numbers don't mean anything, they're just for triggering SteamVR resets.
        // Gaps are included in the numbering to make adding other controllers
        // a bit easier though.
        controller_profile = match config.emulation_mode {
            ControllersEmulationMode::RiftSTouch => 0,
            ControllersEmulationMode::Quest2Touch => 1,
            ControllersEmulationMode::Quest3Plus => 2,
            ControllersEmulationMode::QuestPro => 3,
            ControllersEmulationMode::Pico4 => 10,
            ControllersEmulationMode::ValveIndex => 20,
            ControllersEmulationMode::ViveWand => 40,
            ControllersEmulationMode::ViveTracker => 41,
            ControllersEmulationMode::Custom { .. } => 500,
        };
        use_separate_hand_trackers = config
            .hand_skeleton
            .as_option()
            .is_some_and(|c| c.steamvr_input_2_0);

        true
    } else {
        false
    };

    let body_tracking_vive_enabled =
        if let Switch::Enabled(config) = &settings.headset.body_tracking {
            matches!(config.sink, BodyTrackingSinkConfig::FakeViveTracker)
        } else {
            false
        };

    // Should be true if using full body tracking
    let body_tracking_has_legs = settings
        .headset
        .body_tracking
        .as_option()
        .and_then(|c| c.sources.body_tracking_fb.as_option().cloned())
        .map(|c| c.full_body)
        .or_else(|| {
            settings.headset.body_tracking.as_option().map(|c| {
                matches!(
                    c.sources.body_tracking_bd.as_option(),
                    Some(BodyTrackingBDConfig::BodyTracking { .. })
                )
            })
        })
        .unwrap_or(false);

    let mut foveation_center_size_x = 0.0;
    let mut foveation_center_size_y = 0.0;
    let mut foveation_center_shift_x = 0.0;
    let mut foveation_center_shift_y = 0.0;
    let mut foveation_edge_ratio_x = 0.0;
    let mut foveation_edge_ratio_y = 0.0;
    let enable_foveated_encoding = if let Switch::Enabled(config) = settings.video.foveated_encoding
    {
        foveation_center_size_x = config.center_size_x;
        foveation_center_size_y = config.center_size_y;
        foveation_center_shift_x = config.center_shift_x;
        foveation_center_shift_y = config.center_shift_y;
        foveation_edge_ratio_x = config.edge_ratio_x;
        foveation_edge_ratio_y = config.edge_ratio_y;

        true
    } else {
        false
    };

    let mut brightness = 0.0;
    let mut contrast = 0.0;
    let mut saturation = 0.0;
    let mut gamma = 0.0;
    let mut sharpening = 0.0;
    let enable_color_correction = if let Switch::Enabled(config) = settings.video.color_correction {
        brightness = config.brightness;
        contrast = config.contrast;
        saturation = config.saturation;
        gamma = config.gamma;
        sharpening = config.sharpening;
        true
    } else {
        false
    };

    let nvenc_overrides = settings.video.encoder_config.nvenc;
    let amf_controls = settings.video.encoder_config.amf;
    let hdr_controls = settings.video.encoder_config.hdr;
    let jpeg_xs_controls = settings.video.encoder_config.jpeg_xs;

    OpenvrConfig {
        tracking_ref_only: settings.headset.tracking_ref_only,
        enable_vive_tracker_proxy: settings.headset.enable_vive_tracker_proxy,
        minimum_idr_interval_ms: settings.connection.minimum_idr_interval_ms,
        adapter_index: settings.video.adapter_index,
        codec: settings.video.preferred_codec as _,
        h264_profile: settings.video.encoder_config.h264_profile as u32,
        rate_control_mode: settings.video.encoder_config.rate_control_mode as u32,
        filler_data: settings.video.encoder_config.filler_data,
        entropy_coding: settings.video.encoder_config.entropy_coding as u32,
        force_hdr_srgb_correction: hdr_controls.force_hdr_srgb_correction,
        clamp_hdr_extended_range: hdr_controls.clamp_hdr_extended_range,
        enable_amf_pre_analysis: amf_controls.enable_pre_analysis,
        enable_vbaq: settings.video.encoder_config.enable_vbaq,
        enable_amf_hmqb: amf_controls.enable_hmqb,
        use_amf_preproc: amf_controls.use_preproc,
        amf_preproc_sigma: amf_controls.preproc_sigma,
        amf_preproc_tor: amf_controls.preproc_tor,
        nvenc_quality_preset: nvenc_overrides.quality_preset as u32,
        encoder_quality_preset: settings.video.encoder_config.quality_preset as u32,
        force_sw_encoding: settings
            .video
            .encoder_config
            .software
            .force_software_encoding,
        sw_thread_count: settings.video.encoder_config.software.thread_count,
        jpeg_xs_bits_per_pixel: jpeg_xs_controls.bits_per_pixel,
        jpeg_xs_columns_num: jpeg_xs_controls.columns_num,
        jpeg_xs_use_cuda: jpeg_xs_controls.use_cuda,
        jpeg_xs_cuda_entropy_on_gpu: jpeg_xs_controls.cuda_entropy_on_gpu,
        jpeg_xs_cuda_d3d11_input: jpeg_xs_controls.cuda_d3d11_input,
        jpeg_xs_cuda_blocking_sync: jpeg_xs_controls.cuda_blocking_sync,
        jpeg_xs_cuda_gpu_column_min_threads: jpeg_xs_controls.cuda_gpu_column_min_threads,
        jpeg_xs_cuda_hybrid_cpu_left: jpeg_xs_controls.cuda_hybrid_cpu_left,
        jpeg_xs_cuda_hybrid_zero_copy: jpeg_xs_controls.cuda_hybrid_zero_copy,
        jpeg_xs_pipeline_columns: jpeg_xs_controls.pipeline_columns,
        jpeg_xs_slice_packets: jpeg_xs_controls.slice_packets,
        jpeg_xs_use_mainconcept: jpeg_xs_controls.use_mainconcept,
        jpeg_xs_thread_count: jpeg_xs_controls.thread_count,
        jpeg_xs_dump_to_disk: jpeg_xs_controls.dump_to_disk,
        jpeg_xs_dump_reference_stream: jpeg_xs_controls.dump_reference_stream,
        jpeg_xs_output_directory: jpeg_xs_controls.output_directory,
        controllers_enabled,
        controller_is_tracker,
        body_tracking_vive_enabled,
        body_tracking_has_legs,
        enable_foveated_encoding,
        foveation_center_size_x,
        foveation_center_size_y,
        foveation_center_shift_x,
        foveation_center_shift_y,
        foveation_edge_ratio_x,
        foveation_edge_ratio_y,
        enable_color_correction,
        brightness,
        contrast,
        saturation,
        gamma,
        sharpening,
        linux_async_compute: settings.extra.patches.linux_async_compute,
        linux_async_reprojection: settings.extra.patches.linux_async_reprojection,
        nvenc_tuning_preset: nvenc_overrides.tuning_preset as u32,
        nvenc_multi_pass: nvenc_overrides.multi_pass as u32,
        nvenc_adaptive_quantization_mode: nvenc_overrides.adaptive_quantization_mode as u32,
        nvenc_low_delay_key_frame_scale: nvenc_overrides.low_delay_key_frame_scale,
        nvenc_refresh_rate: nvenc_overrides.refresh_rate,
        enable_intra_refresh: nvenc_overrides.enable_intra_refresh,
        intra_refresh_period: nvenc_overrides.intra_refresh_period,
        intra_refresh_count: nvenc_overrides.intra_refresh_count,
        max_num_ref_frames: nvenc_overrides.max_num_ref_frames,
        gop_length: nvenc_overrides.gop_length,
        p_frame_strategy: nvenc_overrides.p_frame_strategy,
        nvenc_rate_control_mode: nvenc_overrides.rate_control_mode,
        rc_buffer_size: nvenc_overrides.rc_buffer_size,
        rc_initial_delay: nvenc_overrides.rc_initial_delay,
        rc_max_bitrate: nvenc_overrides.rc_max_bitrate,
        rc_average_bitrate: nvenc_overrides.rc_average_bitrate,
        nvenc_enable_weighted_prediction: nvenc_overrides.enable_weighted_prediction,
        capture_frame_dir: settings.extra.capture.capture_frame_dir,
        amd_bitrate_corruption_fix: settings.video.bitrate.image_corruption_fix,
        use_separate_hand_trackers,
        _controller_profile: controller_profile,
        _server_impl_debug: settings.extra.logging.debug_groups.server_impl,
        _client_impl_debug: settings.extra.logging.debug_groups.client_impl,
        _server_core_debug: settings.extra.logging.debug_groups.server_core,
        _client_core_debug: settings.extra.logging.debug_groups.client_core,
        _connection_debug: settings.extra.logging.debug_groups.connection,
        _sockets_debug: settings.extra.logging.debug_groups.sockets,
        _server_gfx_debug: settings.extra.logging.debug_groups.server_gfx,
        _client_gfx_debug: settings.extra.logging.debug_groups.client_gfx,
        _encoder_debug: settings.extra.logging.debug_groups.encoder,
        _decoder_debug: settings.extra.logging.debug_groups.decoder,
        ..old_config
    }
}

// Alternate connection trials with manual IPs and clients discovered on the local network
pub fn handshake_loop(ctx: Arc<ConnectionContext>, lifecycle_state: Arc<RwLock<LifecycleState>>) {
    dbg_connection!("handshake_loop: Begin");

    let mut welcome_socket = match WelcomeSocket::new() {
        Ok(socket) => socket,
        Err(e) => {
            error!("Failed to create discovery socket: {e:?}");
            return;
        }
    };

    let mut wired_connection = None;

    while *lifecycle_state.read() != LifecycleState::ShuttingDown {
        dbg_connection!("handshake_loop: Try connect to wired device");

        let mut wired_client_ips = HashMap::new();
        if SESSION_MANAGER
            .read()
            .client_list()
            .iter()
            .any(|(hostname, info)| {
                info.connection_state == ConnectionState::Disconnected
                    && hostname.as_str() == WIRED_CLIENT_HOSTNAME
            })
        {
            // Make sure the wired connection is created once and kept alive
            let wired_connection = if let Some(connection) = &wired_connection {
                connection
            } else {
                let connection = match WiredConnection::new(
                    FILESYSTEM_LAYOUT.get().unwrap(),
                    |downloaded, maybe_total| {
                        if let Some(total) = maybe_total {
                            alvr_events::send_event(EventType::Adb(AdbEvent {
                                download_progress: downloaded as f32 / total as f32,
                            }));
                        };
                    },
                ) {
                    Ok(connection) => connection,
                    Err(e) => {
                        error!("{e:?}");
                        thread::sleep(RETRY_CONNECT_MIN_INTERVAL);
                        continue;
                    }
                };

                wired_connection = Some(connection);

                wired_connection.as_ref().unwrap()
            };

            let stream_port;
            let client_type;
            let client_autolaunch;
            {
                let session_manager_lock = SESSION_MANAGER.read();
                let connection = &session_manager_lock.settings().connection;
                stream_port = connection.stream_port;
                client_type = connection.wired_client_type.clone();
                client_autolaunch = connection.wired_client_autolaunch;
            }

            let status = match wired_connection.setup(
                CONTROL_PORT,
                stream_port,
                &client_type,
                client_autolaunch,
            ) {
                Ok(status) => status,
                Err(e) => {
                    error!("{e:?}");
                    thread::sleep(RETRY_CONNECT_MIN_INTERVAL);
                    continue;
                }
            };

            #[cfg_attr(not(debug_assertions), expect(unused_variables))]
            if let WiredConnectionStatus::NotReady(s) = status {
                dbg_connection!("handshake_loop: Wired connection not ready: {s}");
                thread::sleep(RETRY_CONNECT_MIN_INTERVAL);
                continue;
            }

            let client_ip = IpAddr::V4(Ipv4Addr::LOCALHOST);
            wired_client_ips.insert(client_ip, WIRED_CLIENT_HOSTNAME.to_owned());
        }

        if !wired_client_ips.is_empty()
            && try_connect(
                Arc::clone(&ctx),
                Arc::clone(&lifecycle_state),
                wired_client_ips,
            )
            .is_ok()
        {
            thread::sleep(RETRY_CONNECT_MIN_INTERVAL);
            continue;
        }

        dbg_connection!("handshake_loop: Try connect to manual IPs");

        let available_manual_client_ips = {
            let mut manual_client_ips = HashMap::new();
            for (hostname, connection_info) in
                SESSION_MANAGER
                    .read()
                    .client_list()
                    .iter()
                    .filter(|(hostname, info)| {
                        info.connection_state == ConnectionState::Disconnected
                            && hostname.as_str() != WIRED_CLIENT_HOSTNAME
                    })
            {
                for ip in &connection_info.manual_ips {
                    manual_client_ips.insert(*ip, hostname.clone());
                }
            }
            manual_client_ips
        };

        if !available_manual_client_ips.is_empty()
            && try_connect(
                Arc::clone(&ctx),
                Arc::clone(&lifecycle_state),
                available_manual_client_ips,
            )
            .is_ok()
        {
            thread::sleep(RETRY_CONNECT_MIN_INTERVAL);
            continue;
        }

        let discovery_config = SESSION_MANAGER
            .read()
            .settings()
            .connection
            .client_discovery
            .clone();
        if let Switch::Enabled(config) = discovery_config {
            dbg_connection!("handshake_loop: Discovering clients");

            let clients = match welcome_socket.recv_all() {
                Ok(clients) => clients,
                Err(e) => {
                    warn!("UDP handshake listening error: {e:?}");

                    thread::sleep(RETRY_CONNECT_MIN_INTERVAL);
                    continue;
                }
            };

            if clients.is_empty() {
                thread::sleep(RETRY_CONNECT_MIN_INTERVAL);
                continue;
            }

            for (client_hostname, client_ip) in clients {
                let trusted = {
                    let mut session_manager = SESSION_MANAGER.write();

                    session_manager.update_client_list(
                        client_hostname.clone(),
                        ClientListAction::AddIfMissing {
                            trusted: false,
                            manual_ips: vec![],
                        },
                    );

                    if config.auto_trust_clients {
                        session_manager
                            .update_client_list(client_hostname.clone(), ClientListAction::Trust);
                    }

                    session_manager
                        .client_list()
                        .get(&client_hostname)
                        .is_some_and(|c| c.trusted)
                };

                // do not attempt connection if the client is already connected
                if trusted
                    && SESSION_MANAGER
                        .read()
                        .client_list()
                        .get(&client_hostname)
                        .is_some_and(|c| c.connection_state == ConnectionState::Disconnected)
                {
                    if let Err(e) = try_connect(
                        Arc::clone(&ctx),
                        Arc::clone(&lifecycle_state),
                        [(client_ip, client_hostname.clone())].into_iter().collect(),
                    ) {
                        error!("Could not initiate connection for {client_hostname}: {e}");
                    }
                }

                thread::sleep(RETRY_CONNECT_MIN_INTERVAL);
            }
        } else {
            thread::sleep(RETRY_CONNECT_MIN_INTERVAL);
        }
    }

    alvr_common::dbg_connection!("handshake_loop: Joining connection threads");

    // At this point, LIFECYCLE_STATE == ShuttingDown, so all threads are already terminating
    for thread in ctx.connection_threads.lock().drain(..) {
        thread.join().ok();
    }

    alvr_common::dbg_connection!("handshake_loop: End");
}

fn try_connect(
    ctx: Arc<ConnectionContext>,
    lifecycle_state: Arc<RwLock<LifecycleState>>,
    mut client_ips: HashMap<IpAddr, String>,
) -> ConResult {
    dbg_connection!("try_connect: Finding client and creating control socket");

    let (proto_socket, client_ip) = ProtoControlSocket::connect_to(
        Duration::from_secs(1),
        PeerType::AnyClient(client_ips.keys().cloned().collect()),
    )?;

    let Some(client_hostname) = client_ips.remove(&client_ip) else {
        con_bail!("unreachable");
    };

    dbg_connection!("try_connect: Pushing new client connection thread");

    ctx.connection_threads.lock().push(thread::spawn({
        let ctx = Arc::clone(&ctx);
        move || {
            if let Err(e) = connection_pipeline(
                Arc::clone(&ctx),
                lifecycle_state,
                proto_socket,
                client_hostname.clone(),
                client_ip,
            ) {
                error!("Handshake error for {client_hostname}: {e}");
            }

            let mut clients_to_be_removed = ctx.clients_to_be_removed.lock();

            let action = if clients_to_be_removed.contains(&client_hostname) {
                clients_to_be_removed.remove(&client_hostname);

                ClientListAction::RemoveEntry
            } else {
                ClientListAction::SetConnectionState(ConnectionState::Disconnected)
            };
            SESSION_MANAGER
                .write()
                .update_client_list(client_hostname, action);
        }
    }));

    Ok(())
}

fn connection_pipeline(
    ctx: Arc<ConnectionContext>,
    lifecycle_state: Arc<RwLock<LifecycleState>>,
    mut proto_socket: ProtoControlSocket,
    client_hostname: String,
    client_ip: IpAddr,
) -> ConResult {
    dbg_connection!("connection_pipeline: Begin");

    // This session lock will make sure settings and client list cannot be changed while connecting
    // to thos client, no other client can connect until handshake is finished. It will then be
    // temporarily relocked while shutting down the threads.
    let mut session_manager_lock = SESSION_MANAGER.write();

    dbg_connection!("connection_pipeline: Setting client state in session");
    session_manager_lock.update_client_list(
        client_hostname.clone(),
        ClientListAction::SetConnectionState(ConnectionState::Connecting),
    );
    session_manager_lock.update_client_list(
        client_hostname.clone(),
        ClientListAction::UpdateCurrentIp(Some(client_ip)),
    );

    let disconnect_notif = Arc::new(Condvar::new());

    dbg_connection!("connection_pipeline: Getting client status packet");
    let connection_result = match proto_socket.recv(HANDSHAKE_ACTION_TIMEOUT) {
        Ok(r) => r,
        Err(ConnectionError::TryAgain(e)) => {
            debug!("Failed to recive client connection packet. This is normal for USB connection.\n{e}");

            return Ok(());
        }
        Err(e) => return Err(e),
    };

    let maybe_streaming_caps = if let ClientConnectionResult::ConnectionAccepted {
        client_protocol_id,
        display_name,
        streaming_capabilities,
        ..
    } = connection_result
    {
        session_manager_lock.update_client_list(
            client_hostname.clone(),
            ClientListAction::SetDisplayName(display_name),
        );

        if client_protocol_id != alvr_common::protocol_id_u64() {
            warn!(
                "Trusted client is incompatible! Expected protocol ID: {}, found: {}",
                alvr_common::protocol_id_u64(),
                client_protocol_id,
            );

            return Ok(());
        }

        streaming_capabilities
    } else {
        debug!("Found client in standby. Retrying");
        return Ok(());
    };

    let streaming_caps = if let Some(streaming_caps) = maybe_streaming_caps {
        alvr_packets::decode_video_streaming_capabilities(&streaming_caps).to_con()?
    } else {
        con_bail!("Only streaming clients are supported for now");
    };

    dbg_connection!("connection_pipeline: setting up negotiated streaming config");

    let initial_settings = session_manager_lock.settings().clone();

    fn get_view_res(config: FrameSize, default_res: UVec2) -> UVec2 {
        let res = match config {
            FrameSize::Scale(scale) => default_res.as_vec2() * scale,
            FrameSize::Absolute { width, height } => {
                let width = width as f32;
                Vec2::new(
                    width,
                    height.map_or_else(
                        || {
                            let default_res = default_res.as_vec2();
                            width * default_res.y / default_res.x
                        },
                        |h| h as f32,
                    ),
                )
            }
        };

        UVec2::new(align32(res.x), align32(res.y))
    }

    let stream_view_resolution = get_view_res(
        initial_settings.video.transcoding_view_resolution.clone(),
        streaming_caps.default_view_resolution,
    );

    let target_view_resolution = get_view_res(
        initial_settings
            .video
            .emulated_headset_view_resolution
            .clone(),
        streaming_caps.default_view_resolution,
    );

    let fps = {
        let mut best_match = 0_f32;
        let mut min_diff = f32::MAX;
        for rate in &streaming_caps.supported_refresh_rates {
            let diff = (*rate - initial_settings.video.preferred_fps).abs();
            if diff < min_diff {
                best_match = *rate;
                min_diff = diff;
            }
        }
        best_match
    };

    if !streaming_caps
        .supported_refresh_rates
        .contains(&initial_settings.video.preferred_fps)
    {
        warn!("Chosen refresh rate not supported. Using {fps}Hz");
    }

    let enable_foveated_encoding =
        if let Switch::Enabled(config) = &initial_settings.video.foveated_encoding {
            let enable = streaming_caps.supports_foveated_encoding || config.force_enable;

            if !enable {
                warn!("Foveated encoding is not supported by the client.");
            }

            enable
        } else {
            false
        };

    let encoder_profile = if initial_settings.video.encoder_config.h264_profile == H264Profile::High
    {
        let profile = if streaming_caps.encoder_high_profile {
            H264Profile::High
        } else {
            H264Profile::Main
        };

        if profile != H264Profile::High {
            warn!("High profile encoding is not supported by the client.");
        }

        profile
    } else {
        initial_settings.video.encoder_config.h264_profile
    };

    let mut enable_10_bits_encoding = if initial_settings
        .video
        .encoder_config
        .server_overrides_use_10bit
    {
        initial_settings.video.encoder_config.use_10bit
    } else {
        streaming_caps.prefer_10bit
    };

    if enable_10_bits_encoding && !streaming_caps.encoder_10_bits {
        warn!("10 bits encoding is not supported by the client.");
        enable_10_bits_encoding = false
    }

    let use_full_range = if initial_settings
        .video
        .encoder_config
        .server_overrides_use_full_range
    {
        initial_settings.video.encoder_config.use_full_range
    } else {
        streaming_caps.prefer_full_range
    };

    let enable_hdr = if initial_settings
        .video
        .encoder_config
        .hdr
        .server_overrides_enable_hdr
    {
        initial_settings.video.encoder_config.hdr.enable_hdr
    } else {
        streaming_caps.prefer_hdr
    };

    let encoding_gamma = if initial_settings
        .video
        .encoder_config
        .server_overrides_encoding_gamma
    {
        initial_settings.video.encoder_config.encoding_gamma
    } else {
        streaming_caps.preferred_encoding_gamma
    };

    let codec = if initial_settings.video.preferred_codec == CodecType::AV1 {
        let codec = if streaming_caps.encoder_av1 {
            CodecType::AV1
        } else {
            CodecType::Hevc
        };

        if codec != CodecType::AV1 {
            warn!("AV1 encoding is not supported by the client.");
        }

        codec
    } else if initial_settings.video.preferred_codec == CodecType::JpegXs {
        let codec = if streaming_caps.encoder_jpegxs {
            CodecType::JpegXs
        } else {
            CodecType::Hevc
        };

        if codec != CodecType::JpegXs {
            warn!("JPEG XS encoding is not supported by the client.");
        }

        codec
    } else {
        initial_settings.video.preferred_codec
    };

    #[cfg_attr(target_os = "linux", allow(unused_variables))]
    let game_audio_sample_rate = if let Switch::Enabled(game_audio_config) =
        &initial_settings.audio.game_audio
    {
        #[cfg(not(target_os = "linux"))]
        {
            let game_audio_device =
                alvr_audio::AudioDevice::new_output(game_audio_config.device.as_ref()).to_con()?;
            if let Switch::Enabled(microphone_config) = &initial_settings.audio.microphone {
                let (sink, source) = alvr_audio::AudioDevice::new_virtual_microphone_pair(
                    microphone_config.devices.clone(),
                )
                .to_con()?;
                if matches!(
                    microphone_config.devices,
                    alvr_session::MicrophoneDevicesConfig::VAC
                        | alvr_session::MicrophoneDevicesConfig::VBCable
                ) {
                    // VoiceMeeter and Custom devices may have arbitrary internal routing.
                    // Therefore, we cannot detect the loopback issue without knowing the routing.
                    if alvr_audio::is_same_device(&game_audio_device, &sink)
                        || alvr_audio::is_same_device(&game_audio_device, &source)
                    {
                        con_bail!("Game audio and microphone cannot point to the same device!");
                    }
                }
                // else:
                // Stream played via VA-CABLE-X will be directly routed to VA-CABLE-X's virtual microphone.
                // Game audio will loop back to the game microphone if they are set to the same VA-CABLE-X device.
            }

            game_audio_device.input_sample_rate().to_con()?
        }
        #[cfg(target_os = "linux")]
        44100
    } else {
        0
    };

    let wired = client_ip.is_loopback();

    dbg_connection!("connection_pipeline: send streaming config");
    let stream_config_packet = alvr_packets::encode_stream_config(
        session_manager_lock.session(),
        &NegotiatedStreamingConfig {
            view_resolution: stream_view_resolution,
            refresh_rate_hint: fps,
            game_audio_sample_rate,
            enable_foveated_encoding,
            use_multimodal_protocol: streaming_caps.multimodal_protocol,
            use_full_range,
            encoding_gamma,
            enable_hdr,
            wired,
        },
    )
    .to_con()?;
    proto_socket.send(&stream_config_packet).to_con()?;

    let (mut control_sender, mut control_receiver) =
        proto_socket.split(STREAMING_RECV_TIMEOUT).to_con()?;

    let mut new_openvr_config = contruct_openvr_config(session_manager_lock.session());
    new_openvr_config.eye_resolution_width = stream_view_resolution.x;
    new_openvr_config.eye_resolution_height = stream_view_resolution.y;
    new_openvr_config.target_eye_resolution_width = target_view_resolution.x;
    new_openvr_config.target_eye_resolution_height = target_view_resolution.y;
    new_openvr_config.refresh_rate = fps as _;
    new_openvr_config.enable_foveated_encoding = enable_foveated_encoding;
    new_openvr_config.h264_profile = encoder_profile as _;
    new_openvr_config.use_10bit_encoder = enable_10_bits_encoding;
    new_openvr_config.use_full_range_encoding = use_full_range;
    new_openvr_config.enable_hdr = enable_hdr;
    new_openvr_config.encoding_gamma = encoding_gamma;
    new_openvr_config.codec = codec as _;

    if session_manager_lock.session().openvr_config != new_openvr_config {
        session_manager_lock.session_mut().openvr_config = new_openvr_config;

        control_sender.send(&ServerControlPacket::Restarting).ok();

        crate::notify_restart_driver();
    }

    dbg_connection!("connection_pipeline: Send StartStream packet");
    control_sender
        .send(&ServerControlPacket::StartStream)
        .to_con()?;

    let signal = control_receiver.recv(HANDSHAKE_ACTION_TIMEOUT)?;
    if !matches!(signal, ClientControlPacket::StreamReady) {
        con_bail!("Got unexpected packet waiting for stream ack");
    }
    dbg_connection!("connection_pipeline: Got StreamReady packet");

    *ctx.statistics_manager.write() = Some(StatisticsManager::new(
        initial_settings.connection.statistics_history_size,
        Duration::from_secs_f32(1.0 / fps),
        if let Switch::Enabled(config) = &initial_settings.headset.controllers {
            config.steamvr_pipeline_frames
        } else {
            0.0
        },
    ));

    *ctx.bitrate_manager.lock() =
        BitrateManager::new(initial_settings.video.bitrate.history_size, fps);

    let stream_protocol = if wired {
        SocketProtocol::Tcp
    } else {
        initial_settings.connection.stream_protocol
    };

    dbg_connection!("connection_pipeline: StreamSocket connect_to_client");
    let mut stream_socket = StreamSocketBuilder::connect_to_client(
        HANDSHAKE_ACTION_TIMEOUT,
        client_ip,
        initial_settings.connection.stream_port,
        stream_protocol,
        initial_settings.connection.dscp,
        initial_settings.connection.server_send_buffer_bytes,
        initial_settings.connection.server_recv_buffer_bytes,
        initial_settings.connection.packet_size as _,
    )?;

    alvr_sockets::set_udp_send_segmentation(initial_settings.connection.udp_send_segmentation);
    let mut video_sender = stream_socket.request_stream(VIDEO);
    let game_audio_sender: alvr_sockets::StreamSender<()> = stream_socket.request_stream(AUDIO);
    let mut microphone_receiver: alvr_sockets::StreamReceiver<()> =
        stream_socket.subscribe_to_stream(AUDIO, MAX_UNREAD_PACKETS);
    let tracking_receiver =
        stream_socket.subscribe_to_stream::<Tracking>(TRACKING, MAX_UNREAD_PACKETS);
    let haptics_sender = stream_socket.request_stream(HAPTICS);
    let mut statics_receiver =
        stream_socket.subscribe_to_stream::<ClientStatistics>(STATISTICS, MAX_UNREAD_PACKETS);

    let (video_channel_sender, video_channel_receiver) =
        std::sync::mpsc::sync_channel(initial_settings.connection.max_queued_server_video_frames);
    // A new, empty queue: whatever the last session left queued went away with its channel.
    let video_queue_gen = VIDEO_QUEUE_GEN.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
    VIDEO_QUEUE_LEN.store(0, std::sync::atomic::Ordering::SeqCst);
    *ctx.video_channel_sender.lock() = Some(video_channel_sender);
    *ctx.haptics_sender.lock() = Some(haptics_sender);

    // Send diagnostics, one line a second into C:\Temp\alvr_send_diag.log. The question they
    // answer: when the client receives far fewer packets than the encoder produces while losing
    // none on the way (2026-09-22: ~5,400 of 18,900 a second at 4288x1664), is this thread the
    // bottleneck? "busy" is the share of wall time spent inside the socket send; near 100 %
    // together with queue drops means it cannot keep up with one send call per shard.
    let shard_payload = (initial_settings.connection.packet_size.max(64) as usize).saturating_sub(16);
    let video_send_thread = thread::spawn({
        let client_hostname = client_hostname.clone();
        move || {
            #[cfg(windows)]
            {
                let note = keep_send_thread_on_performance_cores();
                // Which address the stream goes to, so a session over Wi-Fi is never mistaken
                // for a slow one over the cable again (2026-09-22).
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("C:\\Temp\\alvr_send_diag.log")
                {
                    use std::io::Write;
                    // The socket buffers the stack actually settled on, per session -- the
                    // leading suspect for the two send states. See stream_socket_buffer_sizes().
                    let (snd, rcv) = alvr_sockets::stream_socket_buffer_sizes();
                    f.write_all(
                        format!(
                            "send thread setup: streaming to {client_ip}, {note}, \
                             socket buffers send {snd} B recv {rcv} B\n"
                        )
                        .as_bytes(),
                    )
                    .ok();
                }
            }

            let mut window_start = Instant::now();
            let mut window_cpu_start = current_thread_cpu_time();
            let mut machine_load = load::Load::new();
            let mut sent_packets: u64 = 0;
            let mut sent_bytes: u64 = 0;
            let mut send_time = Duration::ZERO;
            let mut wait_time = Duration::ZERO;
            // Which logical CPU each send ran on (sampled after the send).
            let mut cpu_hist = [0u32; 64];
            while is_streaming(&client_hostname) {
                let t_wait = Instant::now();
                let VideoPacket { header, payload } =
                    match video_channel_receiver.recv_timeout(STREAMING_RECV_TIMEOUT) {
                        Ok(packet) => packet,
                        Err(RecvTimeoutError::Timeout) => continue,
                        Err(RecvTimeoutError::Disconnected) => return,
                    };
                if VIDEO_QUEUE_GEN.load(std::sync::atomic::Ordering::Relaxed) == video_queue_gen {
                    // Saturating: an increment this thread never saw (a packet pushed before the
                    // reset) must not wrap the counter either.
                    VIDEO_QUEUE_LEN
                        .fetch_update(
                            std::sync::atomic::Ordering::Relaxed,
                            std::sync::atomic::Ordering::Relaxed,
                            |n| Some(n.saturating_sub(1)),
                        )
                        .ok();
                }
                wait_time += t_wait.elapsed();

                let t_send = Instant::now();
                let mut buffer = video_sender.get_buffer(&header).unwrap();
                // todo: make encoder write to socket buffers directly to avoid copy
                buffer
                    .get_range_mut(0, payload.len())
                    .copy_from_slice(&payload);
                video_sender.send(buffer).ok();
                send_time += t_send.elapsed();
                cpu_hist[(current_processor() as usize).min(63)] += 1;
                sent_packets += 1;
                sent_bytes += payload.len() as u64;

                let window = window_start.elapsed();
                if window >= Duration::from_secs(1) {
                    let shards = sent_bytes / shard_payload.max(1) as u64 + sent_packets;
                    let drops = VIDEO_QUEUE_DROPS.swap(0, std::sync::atomic::Ordering::Relaxed);
                    let refused = VIDEO_FRAMES_REFUSED.swap(0, std::sync::atomic::Ordering::Relaxed);
                    // Includes the little time spent waiting for work and logging; the send
                    // itself dominates whenever the thread is busy.
                    let cpu_now = current_thread_cpu_time();
                    let kernel = cpu_now.0.saturating_sub(window_cpu_start.0);
                    let user = cpu_now.1.saturating_sub(window_cpu_start.1);
                    window_cpu_start = cpu_now;
                    let mut ranked: Vec<(usize, u32)> =
                        cpu_hist.iter().copied().enumerate().filter(|&(_, n)| n > 0).collect();
                    ranked.sort_by(|a, b| b.1.cmp(&a.1));
                    let total: u32 = ranked.iter().map(|&(_, n)| n).sum();
                    let on_cpus = ranked
                        .iter()
                        .take(3)
                        .map(|&(c, n)| format!("cpu{c} {:.0}%", 100.0 * n as f64 / total.max(1) as f64))
                        .collect::<Vec<_>>()
                        .join(", ");
                    cpu_hist = [0; 64];
                    let line = format!(
                        "send thread: {} packets, ~{} shards, {:.0} Mbit/s | busy {:.0}% (send {:.1} us/shard, kernel {:.1} + user {:.1} us/shard, on {}), waiting for work {:.0}% | queue drops {} ({:.1}% of packets, {} whole frames refused, queue now {}, {} repeated timestamps) | udp segmentation {} | {}\n",
                        sent_packets,
                        shards,
                        sent_bytes as f64 * 8.0 / window.as_secs_f64() / 1e6,
                        100.0 * send_time.as_secs_f64() / window.as_secs_f64(),
                        send_time.as_secs_f64() * 1e6 / shards.max(1) as f64,
                        kernel.as_secs_f64() * 1e6 / shards.max(1) as f64,
                        user.as_secs_f64() * 1e6 / shards.max(1) as f64,
                        on_cpus,
                        100.0 * wait_time.as_secs_f64() / window.as_secs_f64(),
                        drops,
                        100.0 * drops as f64 / (sent_packets + drops).max(1) as f64,
                        refused,
                        VIDEO_QUEUE_LEN.load(std::sync::atomic::Ordering::Relaxed),
                        VIDEO_REPEATED_TIMESTAMPS.swap(0, std::sync::atomic::Ordering::Relaxed),
                        alvr_sockets::udp_segmentation_state(),
                        machine_load.line(),
                    );
                    if let Ok(mut f) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open("C:\\Temp\\alvr_send_diag.log")
                    {
                        use std::io::Write;
                        f.write_all(line.as_bytes()).ok();
                    }
                    window_start = Instant::now();
                    sent_packets = 0;
                    sent_bytes = 0;
                    send_time = Duration::ZERO;
                    wait_time = Duration::ZERO;
                }
            }
        }
    });

    #[cfg_attr(target_os = "linux", allow(unused_variables))]
    let game_audio_thread = if let Switch::Enabled(config) =
        initial_settings.audio.game_audio.clone()
    {
        #[cfg(windows)]
        let ctx = Arc::clone(&ctx);

        let client_hostname = client_hostname.clone();
        thread::spawn(move || {
            while is_streaming(&client_hostname) {
                #[cfg(target_os = "linux")]
                if let Err(e) = alvr_audio::linux::record_audio_blocking_pipewire(
                    Arc::new({
                        let client_hostname = client_hostname.clone();
                        move || is_streaming(&client_hostname)
                    }),
                    game_audio_sender.clone(),
                    2,
                    game_audio_sample_rate,
                ) {
                    error!("Audio record error: {e:?}");
                }

                #[cfg(not(target_os = "linux"))]
                {
                    let device = match alvr_audio::AudioDevice::new_output(config.device.as_ref()) {
                        Ok(data) => data,
                        Err(e) => {
                            warn!("New audio device failed: {e:?}");
                            thread::sleep(RETRY_CONNECT_MIN_INTERVAL);
                            continue;
                        }
                    };

                    #[cfg(windows)]
                    if let Ok(id) = alvr_audio::get_windows_device_id(&device) {
                        let prop = alvr_session::OpenvrProperty {
                            key: alvr_session::OpenvrPropKey::AudioDefaultPlaybackDeviceIdString,
                            value: id,
                        };
                        ctx.events_sender
                            .send(ServerCoreEvent::SetOpenvrProperty {
                                device_id: *alvr_common::HEAD_ID,
                                prop,
                            })
                            .ok();
                    } else {
                        continue;
                    };

                    if let Err(e) = alvr_audio::record_audio_blocking(
                        Arc::new({
                            let client_hostname = client_hostname.clone();
                            move || is_streaming(&client_hostname)
                        }),
                        game_audio_sender.clone(),
                        &device,
                        2,
                        config.mute_when_streaming,
                    ) {
                        error!("Audio record error: {e:?}");
                    }

                    #[cfg(windows)]
                    if let Ok(id) = alvr_audio::AudioDevice::new_output(None)
                        .and_then(|d| alvr_audio::get_windows_device_id(&d))
                    {
                        let prop = alvr_session::OpenvrProperty {
                            key: alvr_session::OpenvrPropKey::AudioDefaultPlaybackDeviceIdString,
                            value: id,
                        };
                        ctx.events_sender
                            .send(ServerCoreEvent::SetOpenvrProperty {
                                device_id: *alvr_common::HEAD_ID,
                                prop,
                            })
                            .ok();
                    }
                }
            }
        })
    } else {
        thread::spawn(|| ())
    };

    let microphone_thread =
        if let Switch::Enabled(config) = initial_settings.audio.microphone.clone() {
            #[cfg(not(target_os = "linux"))]
            #[allow(unused_variables)]
            let (sink, source) =
                alvr_audio::AudioDevice::new_virtual_microphone_pair(config.devices).to_con()?;

            #[cfg(windows)]
            if let Ok(id) = alvr_audio::get_windows_device_id(&source) {
                ctx.events_sender
                    .send(ServerCoreEvent::SetOpenvrProperty {
                        device_id: *alvr_common::HEAD_ID,
                        prop: alvr_session::OpenvrProperty {
                            key: alvr_session::OpenvrPropKey::AudioDefaultRecordingDeviceIdString,
                            value: id,
                        },
                    })
                    .ok();
            }

            let client_hostname = client_hostname.clone();
            thread::spawn(move || {
                #[cfg(not(target_os = "linux"))]
                alvr_common::show_err(alvr_audio::play_audio_loop(
                    {
                        let client_hostname = client_hostname.clone();
                        move || is_streaming(&client_hostname)
                    },
                    &sink,
                    1,
                    streaming_caps.microphone_sample_rate,
                    config.buffering,
                    &mut microphone_receiver,
                ));
                #[cfg(target_os = "linux")]
                alvr_common::show_err(alvr_audio::linux::play_microphone_loop_pipewire(
                    {
                        let client_hostname = client_hostname.clone();
                        move || is_streaming(&client_hostname)
                    },
                    1,
                    streaming_caps.microphone_sample_rate,
                    config.buffering,
                    &mut microphone_receiver,
                ));
            })
        } else {
            thread::spawn(|| ())
        };

    *ctx.tracking_manager.write() =
        TrackingManager::new(initial_settings.connection.statistics_history_size);
    let hand_gesture_manager = Arc::new(Mutex::new(HandGestureManager::new()));

    let tracking_receive_thread = thread::spawn({
        let ctx = Arc::clone(&ctx);
        let hand_gesture_manager = Arc::clone(&hand_gesture_manager);
        let initial_settings = initial_settings.clone();
        let client_hostname = client_hostname.clone();
        move || {
            tracking::tracking_loop(
                &ctx,
                initial_settings,
                streaming_caps.multimodal_protocol,
                hand_gesture_manager,
                tracking_receiver,
                || is_streaming(&client_hostname),
            );
        }
    });

    let statistics_thread = thread::spawn({
        let ctx = Arc::clone(&ctx);
        let client_hostname = client_hostname.clone();
        // Read once, like the tracking loop does with phase_lock_frame_pacing: settings() clones
        // the whole tree, which is not something to do per statistics packet.
        let track_pickup_phase = initial_settings.video.phase_lock_frame_pacing
            && initial_settings.video.phase_lock_track_pickup;
        move || {
            while is_streaming(&client_hostname) {
                let data = match statics_receiver.recv(STREAMING_RECV_TIMEOUT) {
                    Ok(stats) => stats,
                    Err(ConnectionError::TryAgain(_)) => continue,
                    Err(ConnectionError::Other(_)) => return,
                };
                let Ok(client_stats) = data.get_header() else {
                    return;
                };

                if let Some(stats) = &mut *ctx.statistics_manager.write() {
                    let timestamp = client_stats.target_timestamp;
                    let decoder_latency = client_stats.video_decode;
                    // Closed-loop phase control, see adapt_phase_lead_to_queue: the headset's own
                    // queue time is the only direct measure of how close a frame lands to the
                    // renderer's pickup. Read before report_statistics consumes client_stats.
                    let decoder_queue = client_stats.video_decoder_queue;
                    let (network_latency, game_latency) = stats.report_statistics(client_stats);
                    if track_pickup_phase {
                        stats.adapt_phase_lead_to_queue(decoder_queue);
                    }

                    ctx.events_sender
                        .send(ServerCoreEvent::GameRenderLatencyFeedback(game_latency))
                        .ok();

                    let session_manager_lock = SESSION_MANAGER.read();
                    ctx.bitrate_manager.lock().report_frame_latencies(
                        &session_manager_lock.settings().video.bitrate.mode,
                        timestamp,
                        network_latency,
                        decoder_latency,
                    );
                }
            }
        }
    });

    let control_sender = Arc::new(Mutex::new(control_sender));

    let real_time_update_thread = thread::spawn({
        let control_sender = Arc::clone(&control_sender);
        let client_hostname = client_hostname.clone();
        move || {
            while is_streaming(&client_hostname) {
                let config = {
                    let session_manager_lock = SESSION_MANAGER.read();
                    let settings = session_manager_lock.settings();

                    RealTimeConfig::from_settings(settings)
                };

                if let Ok(config) = config.encode() {
                    control_sender.lock().send(&config).ok();
                }

                thread::sleep(REAL_TIME_UPDATE_INTERVAL);
            }
        }
    });

    let keepalive_thread = thread::spawn({
        let control_sender = Arc::clone(&control_sender);
        let disconnect_notif = Arc::clone(&disconnect_notif);
        let client_hostname = client_hostname.clone();
        move || {
            while is_streaming(&client_hostname) {
                if let Err(e) = control_sender.lock().send(&ServerControlPacket::KeepAlive) {
                    info!("Client disconnected. Cause: {e:?}");

                    disconnect_notif.notify_one();

                    return;
                }

                thread::sleep(KEEPALIVE_INTERVAL);
            }
        }
    });

    let control_receive_thread = thread::spawn({
        let ctx = Arc::clone(&ctx);

        let controllers_config = session_manager_lock
            .settings()
            .headset
            .controllers
            .as_option();
        let mut controller_button_mapping_manager = controllers_config.map(|config| {
            if let Some(mappings) = &config.button_mappings {
                ButtonMappingManager::new_manual(mappings)
            } else {
                ButtonMappingManager::new_automatic(
                    &CONTROLLER_PROFILE_INFO
                        .get(&alvr_common::hash_string(QUEST_CONTROLLER_PROFILE_PATH))
                        .unwrap()
                        .button_set,
                    &config.emulation_mode,
                    &config.button_mapping_config,
                )
            }
        });
        let controllers_emulation_mode =
            controllers_config.map(|config| config.emulation_mode.clone());

        let disconnect_notif = Arc::clone(&disconnect_notif);
        let control_sender = Arc::clone(&control_sender);
        let client_hostname = client_hostname.clone();
        move || {
            let mut disconnection_deadline = Instant::now() + KEEPALIVE_TIMEOUT;
            while is_streaming(&client_hostname) {
                let packet = match control_receiver.recv(STREAMING_RECV_TIMEOUT) {
                    Ok(packet) => packet,
                    Err(ConnectionError::TryAgain(_)) => {
                        if Instant::now() > disconnection_deadline {
                            info!("Client disconnected. Timeout");
                            break;
                        } else {
                            continue;
                        }
                    }
                    Err(e) => {
                        info!("Client disconnected. Cause: {e}");
                        break;
                    }
                };

                match packet {
                    ClientControlPacket::PlayspaceSync(packet) => {
                        if !initial_settings.headset.tracking_ref_only {
                            let session_manager_lock = SESSION_MANAGER.read();
                            let config = &session_manager_lock.settings().headset;
                            ctx.tracking_manager.write().recenter(
                                config.position_recentering_mode,
                                config.rotation_recentering_mode,
                            );

                            let area = packet.unwrap_or(Vec2::new(2.0, 2.0));
                            let wh = area.x * area.y;
                            if wh.is_finite() && wh > 0.0 {
                                info!("Received new playspace with size: {}", area);
                                ctx.events_sender
                                    .send(ServerCoreEvent::PlayspaceSync(area))
                                    .ok();
                            } else {
                                warn!("Received invalid playspace size: {}", area);
                                ctx.events_sender
                                    .send(ServerCoreEvent::PlayspaceSync(Vec2::new(2.0, 2.0)))
                                    .ok();
                            }
                        }
                    }
                    ClientControlPacket::RequestIdr => {
                        if let Some(config) = ctx.decoder_config.lock().clone() {
                            control_sender
                                .lock()
                                .send(&ServerControlPacket::DecoderConfig(config))
                                .ok();
                        }
                        ctx.events_sender.send(ServerCoreEvent::RequestIDR).ok();
                    }
                    ClientControlPacket::VideoErrorReport => {
                        // legacy endpoint. todo: remove
                        if let Some(stats) = &mut *ctx.statistics_manager.write() {
                            stats.report_packet_loss();
                        }
                        ctx.events_sender.send(ServerCoreEvent::RequestIDR).ok();
                    }
                    ClientControlPacket::ViewsConfig(config) => {
                        ctx.events_sender
                            .send(ServerCoreEvent::ViewsConfig(ViewsConfig {
                                local_view_transforms: [
                                    Pose {
                                        position: Vec3::new(-config.ipd_m / 2., 0., 0.),
                                        orientation: Quat::IDENTITY,
                                    },
                                    Pose {
                                        position: Vec3::new(config.ipd_m / 2., 0., 0.),
                                        orientation: Quat::IDENTITY,
                                    },
                                ],
                                fov: config.fov,
                            }))
                            .ok();
                    }
                    ClientControlPacket::Battery(packet) => {
                        ctx.events_sender
                            .send(ServerCoreEvent::Battery(packet.clone()))
                            .ok();

                        if let Some(stats) = &mut *ctx.statistics_manager.write() {
                            stats.report_battery(
                                packet.device_id,
                                packet.gauge_value,
                                packet.is_plugged,
                            );
                        }
                    }
                    ClientControlPacket::Buttons(entries) => {
                        {
                            let session_manager_lock = SESSION_MANAGER.read();
                            if session_manager_lock
                                .settings()
                                .extra
                                .logging
                                .log_button_presses
                            {
                                alvr_events::send_event(EventType::Buttons(
                                    entries
                                        .iter()
                                        .map(|e| ButtonEvent {
                                            path: BUTTON_INFO.get(&e.path_id).map_or_else(
                                                || format!("Unknown (ID: {:#16x})", e.path_id),
                                                |info| info.path.to_owned(),
                                            ),
                                            value: e.value,
                                        })
                                        .collect(),
                                ));
                            }
                        }

                        if let Some(manager) = &mut controller_button_mapping_manager {
                            let button_entries = entries
                                .iter()
                                .flat_map(|entry| manager.map_button(entry))
                                .collect::<Vec<_>>();

                            if !button_entries.is_empty() {
                                ctx.events_sender
                                    .send(ServerCoreEvent::Buttons(button_entries))
                                    .ok();
                            }
                        };
                    }
                    ClientControlPacket::ActiveInteractionProfile { profile_id, .. } => {
                        controller_button_mapping_manager = if let Switch::Enabled(config) =
                            &SESSION_MANAGER.read().settings().headset.controllers
                        {
                            if let Some(mappings) = &config.button_mappings {
                                Some(ButtonMappingManager::new_manual(mappings))
                            } else if let (Some(profile_info), Some(emulation_mode)) = (
                                CONTROLLER_PROFILE_INFO.get(&profile_id),
                                &controllers_emulation_mode,
                            ) {
                                Some(ButtonMappingManager::new_automatic(
                                    &profile_info.button_set,
                                    emulation_mode,
                                    &config.button_mapping_config,
                                ))
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                    }
                    ClientControlPacket::Log { level, message } => {
                        info!("Client {client_hostname}: [{level:?}] {message}")
                    }
                    ClientControlPacket::Reserved(json_string) => {
                        let reserved: ReservedClientControlPacket =
                            match serde_json::from_str(&json_string) {
                                Ok(reserved) => reserved,
                                Err(e) => {
                                    info!(
                                    "Failed to parse reserved packet: {e}. Packet: {json_string}"
                                );
                                    continue;
                                }
                            };

                        match reserved {
                            ReservedClientControlPacket::CustomInteractionProfile {
                                input_ids,
                                ..
                            } => {
                                controller_button_mapping_manager = if let Switch::Enabled(config) =
                                    &SESSION_MANAGER.read().settings().headset.controllers
                                {
                                    if let Some(mappings) = &config.button_mappings {
                                        Some(ButtonMappingManager::new_manual(mappings))
                                    } else {
                                        controllers_emulation_mode.as_ref().map(|emulation_mode| {
                                            ButtonMappingManager::new_automatic(
                                                &input_ids,
                                                emulation_mode,
                                                &config.button_mapping_config,
                                            )
                                        })
                                    }
                                } else {
                                    None
                                };
                            }
                        }
                    }
                    _ => (),
                }

                disconnection_deadline = Instant::now() + KEEPALIVE_TIMEOUT;
            }

            disconnect_notif.notify_one()
        }
    });

    let stream_receive_thread = thread::spawn({
        let disconnect_notif = Arc::clone(&disconnect_notif);
        let client_hostname = client_hostname.clone();
        move || {
            while is_streaming(&client_hostname) {
                match stream_socket.recv() {
                    Ok(()) => (),
                    Err(ConnectionError::TryAgain(_)) => continue,
                    Err(e) => {
                        info!("Client disconnected. Cause: {e}");

                        disconnect_notif.notify_one();

                        return;
                    }
                }
            }
        }
    });

    let lifecycle_check_thread = thread::spawn({
        let disconnect_notif = Arc::clone(&disconnect_notif);
        let client_hostname = client_hostname.clone();
        move || {
            while SESSION_MANAGER
                .read()
                .client_list()
                .get(&client_hostname)
                .is_some_and(|c| c.connection_state == ConnectionState::Streaming)
                && *lifecycle_state.read() == LifecycleState::Resumed
            {
                thread::sleep(STREAMING_RECV_TIMEOUT);
            }

            disconnect_notif.notify_one()
        }
    });

    {
        if initial_settings.connection.enable_on_connect_script {
            let on_connect_script = FILESYSTEM_LAYOUT.get().map(|l| l.connect_script()).unwrap();
            info!(
                "Running on connect script (connect): {}",
                on_connect_script.display()
            );
            if let Err(e) = Command::new(&on_connect_script)
                .env("ACTION", "connect")
                .spawn()
            {
                warn!("Failed to run connect script: {e}");
            }
        }
    }

    if initial_settings.extra.capture.startup_video_recording {
        info!("Creating recording file");
        crate::create_recording_file(&ctx, session_manager_lock.settings());
    }

    session_manager_lock.update_client_list(
        client_hostname.clone(),
        ClientListAction::SetConnectionState(ConnectionState::Streaming),
    );

    ctx.events_sender
        .send(ServerCoreEvent::ClientConnected)
        .ok();

    dbg_connection!("connection_pipeline: handshake finished; unlocking streams");
    alvr_common::wait_rwlock(&disconnect_notif, &mut session_manager_lock);
    dbg_connection!("connection_pipeline: Begin connection shutdown");

    // This requests shutdown from threads
    *ctx.video_channel_sender.lock() = None;
    *ctx.haptics_sender.lock() = None;

    *ctx.video_recording_file.lock() = None;

    session_manager_lock.update_client_list(
        client_hostname,
        ClientListAction::SetConnectionState(ConnectionState::Disconnecting),
    );

    let enable_on_disconnect_script = session_manager_lock
        .settings()
        .connection
        .enable_on_disconnect_script;
    if enable_on_disconnect_script {
        let on_disconnect_script = FILESYSTEM_LAYOUT
            .get()
            .map(|l| l.disconnect_script())
            .unwrap();
        info!(
            "Running on disconnect script (disconnect): {}",
            on_disconnect_script.display()
        );
        if let Err(e) = Command::new(&on_disconnect_script)
            .env("ACTION", "disconnect")
            .spawn()
        {
            warn!("Failed to run disconnect script: {e}");
        }
    }

    // Allow threads to shutdown correctly
    drop(session_manager_lock);

    // Ensure shutdown of threads
    dbg_connection!("connection_pipeline: Shutdown threads");
    video_send_thread.join().ok();
    game_audio_thread.join().ok();
    microphone_thread.join().ok();
    tracking_receive_thread.join().ok();
    statistics_thread.join().ok();
    real_time_update_thread.join().ok();
    control_receive_thread.join().ok();
    stream_receive_thread.join().ok();
    keepalive_thread.join().ok();
    lifecycle_check_thread.join().ok();

    ctx.events_sender
        .send(ServerCoreEvent::ClientDisconnected)
        .ok();

    dbg_connection!("connection_pipeline: End");

    Ok(())
}
