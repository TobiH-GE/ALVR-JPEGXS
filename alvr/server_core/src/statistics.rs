use alvr_common::{SlidingWindowAverage, HEAD_ID};
use alvr_events::{BitrateDirectives, EventType, GraphStatistics, StatisticsSummary};
use alvr_packets::ClientStatistics;
use std::{
    collections::{HashMap, VecDeque},
    fs::OpenOptions,
    io::{BufWriter, Write},
    time::{Duration, Instant},
};

const FULL_REPORT_INTERVAL: Duration = Duration::from_millis(500);
const EPS_INTERVAL: Duration = Duration::from_micros(1);

pub struct HistoryFrame {
    target_timestamp: Duration,
    tracking_received: Instant,
    frame_present: Instant,
    frame_composed: Instant,
    frame_encoded: Instant,
    video_packet_bytes: usize,
    total_pipeline_latency: Duration,
}

impl Default for HistoryFrame {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            target_timestamp: Duration::ZERO,
            tracking_received: now,
            frame_present: now,
            frame_composed: now,
            frame_encoded: now,
            video_packet_bytes: 0,
            total_pipeline_latency: Duration::ZERO,
        }
    }
}

#[derive(Default, Clone)]
struct BatteryData {
    gauge_value: f32,
    is_plugged: bool,
}

pub struct StatisticsManager {
    history_buffer: VecDeque<HistoryFrame>,
    max_history_size: usize,
    last_full_report_instant: Instant,
    last_frame_present_instant: Instant,
    last_frame_present_interval: Duration,
    video_packets_total: usize,
    video_packets_partial_sum: usize,
    video_bytes_total: usize,
    video_bytes_partial_sum: usize,
    packets_lost_total: usize,
    packets_lost_partial_sum: usize,
    battery_gauges: HashMap<u64, BatteryData>,
    steamvr_pipeline_latency: Duration,
    motion_to_photon_latency_average: SlidingWindowAverage<Duration>,
    last_vsync_time: Instant,
    // Tracking arrival -> virtual vsync. Constant 1 ms until the queue controller above moves it.
    phase_lead_s: f64,
    phase_queue_samples: Vec<f64>,
    frame_interval: Duration,
    last_throughput_directives: BitrateDirectives,
    // 2026-09-05: the Dashboard's live graph is the only place these per-frame numbers
    // (the exact same GraphStatistics sent below) were ever visible, making it impossible
    // to look back at a rough patch after the fact or grep/plot it. Buffered (not raw
    // File::write_all) since this fires up to ~90 times/sec -- an unbuffered write would
    // be a syscall per video frame, the same class of self-inflicted stall the client-side
    // per-call log() open/close bug turned out to be. None if the file couldn't be
    // created; every write site below is a no-op in that case rather than panicking.
    graph_stats_log: Option<BufWriter<std::fs::File>>,
}

impl StatisticsManager {
    // history size used to calculate average total pipeline latency
    pub fn new(
        max_history_size: usize,
        nominal_server_frame_interval: Duration,
        steamvr_pipeline_frames: f32,
    ) -> Self {
        Self {
            history_buffer: VecDeque::new(),
            max_history_size,
            last_full_report_instant: Instant::now(),
            last_frame_present_instant: Instant::now(),
            last_frame_present_interval: Duration::ZERO,
            video_packets_total: 0,
            video_packets_partial_sum: 0,
            video_bytes_total: 0,
            video_bytes_partial_sum: 0,
            packets_lost_total: 0,
            packets_lost_partial_sum: 0,
            battery_gauges: HashMap::new(),
            steamvr_pipeline_latency: Duration::from_secs_f32(
                steamvr_pipeline_frames * nominal_server_frame_interval.as_secs_f32(),
            ),
            motion_to_photon_latency_average: SlidingWindowAverage::new(
                Duration::ZERO,
                max_history_size,
            ),
            last_vsync_time: Instant::now(),
            phase_lead_s: 0.001,
            phase_queue_samples: Vec::new(),
            frame_interval: nominal_server_frame_interval,
            last_throughput_directives: BitrateDirectives::default(),
            graph_stats_log: {
                let file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("C:\\Temp\\alvr_graph_stats.csv")
                    .ok();
                file.map(|mut f| {
                    // Appends across restarts, so only write the header on a fresh (empty)
                    // file -- otherwise a re-run would interleave a second header mid-file.
                    if f.metadata().map(|m| m.len()).unwrap_or(0) == 0 {
                        let _ = writeln!(
                            f,
                            "wall_clock_ms,total_pipeline_latency_ms,game_time_ms,server_compositor_ms,\
                             encoder_ms,network_ms,decoder_ms,decoder_queue_ms,client_compositor_ms,\
                             vsync_queue_ms,client_fps,server_fps,throughput_mbps,bitrate_mbps"
                        );
                    }
                    BufWriter::new(f)
                })
            },
        }
    }

    pub fn report_tracking_received(&mut self, target_timestamp: Duration, phase_lock_vsync: bool) {
        if !self
            .history_buffer
            .iter()
            .any(|frame| frame.target_timestamp == target_timestamp)
        {
            let now = Instant::now();
            self.history_buffer.push_front(HistoryFrame {
                target_timestamp,
                tracking_received: now,
                ..Default::default()
            });
            if phase_lock_vsync {
                self.phase_lock_vsync_to_tracking(now);
            }
        }

        if self.history_buffer.len() > self.max_history_size {
            self.history_buffer.pop_back();
        }
    }

    // Phase-locks the virtual vsync that paces SteamVR (duration_until_next_vsync, slept on in
    // server_openvr's wait_for_vsync when enforce_server_frame_pacing is on) to the arrival of
    // the headset's tracking. The virtual vsync used to start at connection time and just step by
    // the nominal frame interval, so its phase against the tracking arrival was random per
    // session: measured 2026-09-13 as game_time (tracking received -> SteamVR present) of
    // 3.1-3.3 ms in some sessions and 9.6-11.8 ms in others, i.e. one whole vsync of total
    // latency (50.5 vs 61.6 ms). Target: the vsync falls phase_lead_s after the tracking
    // arrives, so SteamVR starts its frame right after the newest pose was submitted. Each packet
    // moves the vsync by only a fraction of the phase error, capped per step, so Wi-Fi jitter of
    // single packets averages out and SteamVR's pacing never jumps; the continuous correction
    // also absorbs small clock-rate differences between headset and PC.
    //
    // ---
    //
    // Closed loop on the headset's own decoder_queue, i.e. the time a decoded frame waits for the
    // renderer to pick it up. Total latency is a step function of that wait: land just before the
    // pickup and the frame is shown in the next display cycle (~47 ms measured), land just after
    // and it waits a full 11.1 ms display period (~58 ms). The phase between the two free-running
    // 90 Hz clocks is drawn per session and drifts, so it cannot be set once.
    //
    // The lead (tracking arrival -> virtual vsync) is the knob: a larger lead produces the frame
    // later, so it arrives later and waits less. The loop creeps towards the pickup and retreats
    // fast, and treats a window with too many queue times above half a period as "already over
    // the cliff" rather than "far too early" -- a missed frame waits nearly a whole period, which
    // would otherwise read as the opposite of what it is.
    // The cost is deeply asymmetric: sitting half a millisecond further from the pickup costs
    // half a millisecond, overshooting it costs a whole display period (11.1 ms). So the loop
    // creeps towards the pickup and retreats fast, with a dead zone in between -- a symmetric
    // controller aimed at a single target oscillated across the cliff instead, in a ~4 s limit
    // cycle: 2.5 ms, 2.7, then 7.7 (missed), pull back, repeat (measured 2026-09-19).
    // Target with real margin against the arrival jitter (network p50 9.2 vs p95 11.3 ms): the
    // frame should be ready a few ms before the pickup, not a few tenths.
    const PHASE_QUEUE_TARGET_S: f64 = 0.0035;
    const PHASE_QUEUE_DEAD_ZONE_S: f64 = 0.001;
    const PHASE_QUEUE_STEP_S: f64 = 0.0005;
    // A frame that missed its pickup waits nearly a whole period, so a queue time above half a
    // period is a miss, not lateness. More than this share of them in a window and the loop
    // retreats at once instead of reading the inflated median as "far too early".
    const PHASE_QUEUE_WINDOW: usize = 90;

    pub fn adapt_phase_lead_to_queue(&mut self, decoder_queue: Duration) {
        let interval_s = self.frame_interval.as_secs_f64();
        if interval_s <= 0.0 {
            return;
        }
        self.phase_queue_samples.push(decoder_queue.as_secs_f64());
        if self.phase_queue_samples.len() < Self::PHASE_QUEUE_WINDOW {
            return;
        }

        // Median, not mean: a single frame that missed its pickup waits a whole period, and that
        // outlier would drag a mean far enough to move the lead the wrong way.
        self.phase_queue_samples
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let q = self.phase_queue_samples[self.phase_queue_samples.len() / 2];
        let missed = self
            .phase_queue_samples
            .iter()
            .filter(|s| **s > interval_s / 2.0)
            .count() as f64
            / self.phase_queue_samples.len() as f64;
        self.phase_queue_samples.clear();

        // The queue time IS the latency to be minimised, and there is no separate "missed"
        // case to detect: a frame that just missed the pickup and a frame that was simply ready
        // far too early both show up as a long wait, and the answer to both is the same -- make
        // the frame later. Treating a long wait as a miss and retreating was measured on device
        // as the wrong direction in 24 of 30 windows, ending at 58.2 ms with 18 % of frames on
        // the fast step; the wrap below already turns "just missed" into the short way round,
        // which is a small retreat.
        //
        // What the earlier symmetric version got wrong was not the direction but the step: at a
        // 2 ms target it crept into the cliff and then jumped 2 ms back, a ~4 s limit cycle. Hence
        // a target with margin against the jitter, a dead zone, and half-millisecond steps.
        let mut err_s = q - Self::PHASE_QUEUE_TARGET_S;
        err_s -= interval_s * (err_s / interval_s).round();
        let (step_s, reason) = if err_s.abs() < Self::PHASE_QUEUE_DEAD_ZONE_S {
            (0.0, "hold")
        } else {
            (
                err_s.clamp(-Self::PHASE_QUEUE_STEP_S, Self::PHASE_QUEUE_STEP_S),
                if err_s > 0.0 { "later" } else { "earlier" },
            )
        };
        let old_lead_s = self.phase_lead_s;
        self.phase_lead_s = (self.phase_lead_s + step_s).rem_euclid(interval_s);

        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("C:\\Temp\\alvr_phase_lock.log")
        {
            use std::io::Write;
            let _ = writeln!(
                file,
                "{} phase queue: median {:.2} ms | missed {:.0}% | target {:.2} | {} | lead {:.2} -> {:.2} ms",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis())
                    .unwrap_or(0),
                q * 1000.0,
                missed * 100.0,
                Self::PHASE_QUEUE_TARGET_S * 1000.0,
                reason,
                old_lead_s * 1000.0,
                self.phase_lead_s * 1000.0
            );
        }
    }

    fn phase_lock_vsync_to_tracking(&mut self, arrival: Instant) {
        const PHASE_LOCK_GAIN: f64 = 0.1;
        const PHASE_LOCK_MAX_STEP_S: f64 = 0.0005;

        let interval = self.frame_interval;
        let interval_s = interval.as_secs_f64();
        if interval_s <= 0.0 {
            return;
        }

        // Most recent vsync not after the arrival.
        while self.last_vsync_time + interval <= arrival {
            self.last_vsync_time += interval;
        }

        let target = arrival + Duration::from_secs_f64(self.phase_lead_s);
        let next_vsync = self.last_vsync_time + interval;
        let mut error_s = if target >= next_vsync {
            (target - next_vsync).as_secs_f64()
        } else {
            -(next_vsync - target).as_secs_f64()
        };
        // Wrap to [-T/2, T/2]: the vsync is periodic, so correct toward the nearest one.
        error_s -= interval_s * (error_s / interval_s).round();

        let step_s = (error_s * PHASE_LOCK_GAIN).clamp(-PHASE_LOCK_MAX_STEP_S, PHASE_LOCK_MAX_STEP_S);
        let step = Duration::from_secs_f64(step_s.abs());
        if step_s >= 0.0 {
            self.last_vsync_time += step;
        } else if let Some(earlier) = self.last_vsync_time.checked_sub(step) {
            self.last_vsync_time = earlier;
        }
    }

    pub fn report_frame_present(&mut self, target_timestamp: Duration, offset: Duration) {
        if let Some(frame) = self
            .history_buffer
            .iter_mut()
            .find(|frame| frame.target_timestamp == target_timestamp)
        {
            let now = Instant::now() - offset;

            self.last_frame_present_interval =
                now.saturating_duration_since(self.last_frame_present_instant);
            self.last_frame_present_instant = now;

            frame.frame_present = now;
        }
    }

    pub fn report_frame_composed(&mut self, target_timestamp: Duration, offset: Duration) {
        if let Some(frame) = self
            .history_buffer
            .iter_mut()
            .find(|frame| frame.target_timestamp == target_timestamp)
        {
            frame.frame_composed = Instant::now() - offset;
        }
    }

    // returns encoding interval
    pub fn report_frame_encoded(
        &mut self,
        target_timestamp: Duration,
        bytes_count: usize,
    ) -> Duration {
        self.video_packets_total += 1;
        self.video_packets_partial_sum += 1;
        self.video_bytes_total += bytes_count;
        self.video_bytes_partial_sum += bytes_count;

        if let Some(frame) = self
            .history_buffer
            .iter_mut()
            .find(|frame| frame.target_timestamp == target_timestamp)
        {
            frame.frame_encoded = Instant::now();

            frame.video_packet_bytes = bytes_count;

            frame
                .frame_encoded
                .saturating_duration_since(frame.frame_composed)
        } else {
            Duration::ZERO
        }
    }

    pub fn report_packet_loss(&mut self) {
        self.packets_lost_total += 1;
        self.packets_lost_partial_sum += 1;
    }

    pub fn report_battery(&mut self, device_id: u64, gauge_value: f32, is_plugged: bool) {
        *self.battery_gauges.entry(device_id).or_default() = BatteryData {
            gauge_value,
            is_plugged,
        };
    }

    pub fn report_throughput_stats(&mut self, stats: BitrateDirectives) {
        self.last_throughput_directives = stats;
    }

    // Called every frame. Some statistics are reported once every frame
    // Returns (network latency, game time latency)
    pub fn report_statistics(&mut self, client_stats: ClientStatistics) -> (Duration, Duration) {
        self.motion_to_photon_latency_average
            .submit_sample(client_stats.total_pipeline_latency);

        if let Some(frame) = self
            .history_buffer
            .iter_mut()
            .find(|frame| frame.target_timestamp == client_stats.target_timestamp)
        {
            frame.total_pipeline_latency = client_stats.total_pipeline_latency;

            let game_time_latency = frame
                .frame_present
                .saturating_duration_since(frame.tracking_received);

            let server_compositor_latency = frame
                .frame_composed
                .saturating_duration_since(frame.frame_present);

            let encoder_latency = frame
                .frame_encoded
                .saturating_duration_since(frame.frame_composed);

            // The network latency cannot be estiamed directly. It is what's left of the total
            // latency after subtracting all other latency intervals. In particular it contains the
            // transport latency of the tracking packet and the interval between the first video
            // packet is sent and the last video packet is received for a specific frame.
            // For safety, use saturating_sub to avoid a crash if for some reason the network
            // latency is miscalculated as negative.
            let network_latency = frame.total_pipeline_latency.saturating_sub(
                game_time_latency
                    + server_compositor_latency
                    + encoder_latency
                    + client_stats.video_decode
                    + client_stats.video_decoder_queue
                    + client_stats.rendering
                    + client_stats.vsync_queue,
            );

            let client_fps =
                1.0 / Duration::max(client_stats.frame_interval, EPS_INTERVAL).as_secs_f32();
            let server_fps =
                1.0 / Duration::max(self.last_frame_present_interval, EPS_INTERVAL).as_secs_f32();

            if self.last_full_report_instant + FULL_REPORT_INTERVAL < Instant::now() {
                self.last_full_report_instant += FULL_REPORT_INTERVAL;

                let interval_secs = FULL_REPORT_INTERVAL.as_secs_f32();

                alvr_events::send_event(EventType::StatisticsSummary(StatisticsSummary {
                    video_packets_total: self.video_packets_total,
                    video_packets_per_sec: (self.video_packets_partial_sum as f32 / interval_secs)
                        as _,
                    video_mbytes_total: (self.video_bytes_total as f32 / 1e6) as usize,
                    video_mbits_per_sec: self.video_bytes_partial_sum as f32 * 8.
                        / 1e6
                        / interval_secs,
                    total_latency_ms: client_stats.total_pipeline_latency.as_secs_f32() * 1000.,
                    network_latency_ms: network_latency.as_secs_f32() * 1000.,
                    encode_latency_ms: encoder_latency.as_secs_f32() * 1000.,
                    decode_latency_ms: client_stats.video_decode.as_secs_f32() * 1000.,
                    packets_lost_total: self.packets_lost_total,
                    packets_lost_per_sec: (self.packets_lost_partial_sum as f32 / interval_secs)
                        as _,
                    client_fps: client_fps as _,
                    server_fps: server_fps as _,
                    battery_hmd: (self
                        .battery_gauges
                        .get(&HEAD_ID)
                        .cloned()
                        .unwrap_or_default()
                        .gauge_value
                        * 100.) as u32,
                    hmd_plugged: self
                        .battery_gauges
                        .get(&HEAD_ID)
                        .cloned()
                        .unwrap_or_default()
                        .is_plugged,
                }));

                self.video_packets_partial_sum = 0;
                self.video_bytes_partial_sum = 0;
                self.packets_lost_partial_sum = 0;

                if let Some(log) = &mut self.graph_stats_log {
                    let _ = log.flush();
                }
            }

            let packet_bits = frame.video_packet_bytes as f32 * 8.0;
            let throughput_bps =
                packet_bits / Duration::max(network_latency, EPS_INTERVAL).as_secs_f32();
            let bitrate_bps = packet_bits
                / Duration::max(self.last_frame_present_interval, EPS_INTERVAL).as_secs_f32();

            // todo: use target timestamp in nanoseconds. the dashboard needs to use the first
            // timestamp as the graph time origin.
            alvr_events::send_event(EventType::GraphStatistics(GraphStatistics {
                total_pipeline_latency_s: client_stats.total_pipeline_latency.as_secs_f32(),
                game_time_s: game_time_latency.as_secs_f32(),
                server_compositor_s: server_compositor_latency.as_secs_f32(),
                encoder_s: encoder_latency.as_secs_f32(),
                network_s: network_latency.as_secs_f32(),
                decoder_s: client_stats.video_decode.as_secs_f32(),
                decoder_queue_s: client_stats.video_decoder_queue.as_secs_f32(),
                client_compositor_s: client_stats.rendering.as_secs_f32(),
                vsync_queue_s: client_stats.vsync_queue.as_secs_f32(),
                client_fps,
                server_fps,
                bitrate_directives: self.last_throughput_directives.clone(),
                throughput_bps,
                bitrate_bps,
            }));

            if let Some(log) = &mut self.graph_stats_log {
                let wall_clock_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis())
                    .unwrap_or(0);
                let _ = writeln!(
                    log,
                    "{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{},{},{:.2},{:.2}",
                    wall_clock_ms,
                    client_stats.total_pipeline_latency.as_secs_f32() * 1000.,
                    game_time_latency.as_secs_f32() * 1000.,
                    server_compositor_latency.as_secs_f32() * 1000.,
                    encoder_latency.as_secs_f32() * 1000.,
                    network_latency.as_secs_f32() * 1000.,
                    client_stats.video_decode.as_secs_f32() * 1000.,
                    client_stats.video_decoder_queue.as_secs_f32() * 1000.,
                    client_stats.rendering.as_secs_f32() * 1000.,
                    client_stats.vsync_queue.as_secs_f32() * 1000.,
                    client_fps,
                    server_fps,
                    throughput_bps / 1e6,
                    bitrate_bps / 1e6,
                );
                // Deliberately NOT flushed here -- this fires up to ~90 times/sec, and a
                // flush()-per-row would be a write() syscall per video frame on this same
                // hot path, exactly the class of self-inflicted periodic stall the
                // client-side log() open/close bug turned out to be. BufWriter's own
                // buffer absorbs that; the periodic flush below (same cadence as the
                // half-second StatisticsSummary report) bounds how much a crash could lose
                // without paying a syscall on every frame.
            }

            (network_latency, game_time_latency)
        } else {
            (Duration::ZERO, Duration::ZERO)
        }
    }

    pub fn motion_to_photon_latency_average(&self) -> Duration {
        self.motion_to_photon_latency_average.get_average()
    }

    pub fn tracker_pose_time_offset(&self) -> Duration {
        // This is the opposite of the client's StatisticsManager::tracker_prediction_offset().
        self.steamvr_pipeline_latency
    }

    // NB: this call is non-blocking, waiting should be done externally
    pub fn duration_until_next_vsync(&mut self) -> Duration {
        let now = Instant::now();

        // update the last vsync if it's too old
        while self.last_vsync_time + self.frame_interval < now {
            self.last_vsync_time += self.frame_interval;
        }

        (self.last_vsync_time + self.frame_interval).saturating_duration_since(now)
    }
}
