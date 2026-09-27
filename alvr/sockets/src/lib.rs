mod backend;
mod control_socket;
mod stream_socket;

use alvr_common::{anyhow::Result, info};
use alvr_session::{DscpTos, SocketBufferSize};
use socket2::Socket;
use std::{
    net::{IpAddr, Ipv4Addr},
    time::Duration,
};

pub use control_socket::*;
pub use stream_socket::*;

pub const LOCAL_IP: IpAddr = IpAddr::V4(Ipv4Addr::UNSPECIFIED);
pub const CONTROL_PORT: u16 = 9943;
pub const HANDSHAKE_PACKET_SIZE_BYTES: usize = 56; // this may change in future protocols
pub const KEEPALIVE_INTERVAL: Duration = Duration::from_millis(500);
pub const KEEPALIVE_TIMEOUT: Duration = Duration::from_secs(2);

pub const MDNS_SERVICE_TYPE: &str = "_alvr._tcp.local.";
pub const MDNS_PROTOCOL_KEY: &str = "protocol";
pub const MDNS_DEVICE_ID_KEY: &str = "device_id";

pub const WIRED_CLIENT_HOSTNAME: &str = "client.wired";

// UDP segmentation offload for the stream sender (Windows): one send call hands the stack many
// shards, which it cuts into datagrams itself. Process-wide on purpose -- there is one stream
// socket, and the setting is fixed for a session. See StreamSender::send.
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
const USO_OFF: u8 = 0;
const USO_WANTED: u8 = 1;
const USO_ON: u8 = 2;
const USO_UNSUPPORTED: u8 = 3;
// Datagrams larger than one Ethernet frame (packet_size above ~1470 B): Windows' segmentation cuts
// at the segment size and does not fragment, so it cannot carry them. Sent one call per datagram,
// and the IP layer fragments each into MTU-sized frames -- the receiver then reads one datagram per
// several frames (2026-09-25, the fragmentation experiment).
const USO_ABOVE_MTU: u8 = 4;
// Largest UDP payload that fits a 1500-byte Ethernet frame without IP fragmentation (IPv4).
pub const UDP_PAYLOAD_PER_FRAME: usize = 1472;
static USO_STATE: AtomicU8 = AtomicU8::new(USO_OFF);
static USO_SEGMENT: AtomicUsize = AtomicUsize::new(0);

// What the stack ACTUALLY gave us for this session's stream socket, read back after setting it --
// not what was asked for. The send thread has two stable states, 1.0 us/shard and 17 us/shard,
// decided per session and constant within it, with zero errors at every level and an identical
// NIC (route, link speed, offload settings and error counters are the same in both -- measured
// 2026-09-24). A blocking sendto on a small send buffer would look exactly like the slow one:
// 100 % busy, 0 % waiting for work, all of it kernel time. The buffer is set per session, which
// would also explain why the state is fixed for a session. Printed in the send thread's setup
// line so the next slow session says so itself.
static SOCK_SEND_BUF: AtomicUsize = AtomicUsize::new(0);
static SOCK_RECV_BUF: AtomicUsize = AtomicUsize::new(0);

/// The stream socket's send and receive buffer sizes as the stack reports them.
pub fn stream_socket_buffer_sizes() -> (usize, usize) {
    (
        SOCK_SEND_BUF.load(Ordering::Relaxed),
        SOCK_RECV_BUF.load(Ordering::Relaxed),
    )
}

pub fn set_udp_send_segmentation(enabled: bool) {
    USO_STATE.store(
        if enabled && cfg!(windows) { USO_WANTED } else { USO_OFF },
        Ordering::Relaxed,
    );
    USO_SEGMENT.store(0, Ordering::Relaxed);
    USO_FALLBACK_CALLS.store(0, Ordering::Relaxed);
    USO_RETRIED_OK.store(0, Ordering::Relaxed);
    USO_LAST_ERROR.store(0, Ordering::Relaxed);
    USO_UNSUPPORTED_STREAK.store(0, Ordering::Relaxed);
}

fn udp_send_segmentation_wanted() -> bool {
    matches!(USO_STATE.load(Ordering::Relaxed), USO_WANTED | USO_ON)
}

// Segmented sends that failed and went out one shard at a time instead, since the session started,
// and the Windows error code of the last one. Before 2026-09-25 ANY failed send -- a momentary
// WSAENOBUFS, or the WSAECONNRESET Windows reports after an ICMP "port unreachable" from a client
// that was not listening yet -- switched segmentation off for good. Every shard then cost a send
// call of its own (17-30 us against 1-1.5 us), the send thread fell to about 30 % of what the
// encoder produced, and the queue refused the rest: the "slow send state" that came and went
// between sessions. Now only an error that means "not supported" switches it off.
static USO_FALLBACK_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static USO_RETRIED_OK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static USO_LAST_ERROR: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
static USO_UNSUPPORTED_STREAK: AtomicUsize = AtomicUsize::new(0);

pub fn udp_segmentation_state() -> String {
    let state = match USO_STATE.load(Ordering::Relaxed) {
        USO_OFF => "off",
        USO_WANTED => "on, not used yet",
        USO_ON => "on",
        USO_ABOVE_MTU => "off: packets above the MTU, the IP layer fragments them",
        _ => "REFUSED by Windows, one call per packet",
    };
    let fallbacks = USO_FALLBACK_CALLS.load(Ordering::Relaxed);
    let retried = USO_RETRIED_OK.load(Ordering::Relaxed);
    if fallbacks == 0 && retried == 0 {
        state.to_string()
    } else {
        format!(
            "{state} ({fallbacks} calls sent shard by shard, {retried} succeeded on retry, last error {})",
            USO_LAST_ERROR.load(Ordering::Relaxed)
        )
    }
}

fn set_socket_buffers(
    socket: &socket2::Socket,
    send_buffer_bytes: SocketBufferSize,
    recv_buffer_bytes: SocketBufferSize,
) -> Result<()> {
    info!(
        "Initial socket buffer size: send: {}B, recv: {}B",
        socket.send_buffer_size()?,
        socket.recv_buffer_size()?
    );

    {
        let maybe_size = match send_buffer_bytes {
            SocketBufferSize::Default => None,
            SocketBufferSize::Maximum => Some(u32::MAX),
            SocketBufferSize::Custom(size) => Some(size),
        };

        if let Some(size) = maybe_size {
            if let Err(e) = socket.set_send_buffer_size(size as usize) {
                info!("Error setting socket send buffer: {e}");
            } else {
                info!(
                    "Set socket send buffer succeeded: {}",
                    socket.send_buffer_size()?
                );
            }
        }
    }

    {
        let maybe_size = match recv_buffer_bytes {
            SocketBufferSize::Default => None,
            SocketBufferSize::Maximum => Some(u32::MAX),
            SocketBufferSize::Custom(size) => Some(size),
        };

        if let Some(size) = maybe_size {
            if let Err(e) = socket.set_recv_buffer_size(size as usize) {
                info!("Error setting socket recv buffer: {e}");
            } else {
                info!(
                    "Set socket recv buffer succeeded: {}",
                    socket.recv_buffer_size()?
                );
            }
        }
    }

    // Read back, so what gets reported is what the stack settled on rather than what was wanted.
    SOCK_SEND_BUF.store(socket.send_buffer_size().unwrap_or(0), Ordering::Relaxed);
    SOCK_RECV_BUF.store(socket.recv_buffer_size().unwrap_or(0), Ordering::Relaxed);

    Ok(())
}

fn set_dscp(socket: &Socket, dscp: Option<DscpTos>) {
    // https://en.wikipedia.org/wiki/Differentiated_services
    if let Some(dscp) = dscp {
        let tos = match dscp {
            DscpTos::BestEffort => 0,
            DscpTos::ClassSelector(precedence) => precedence << 3,
            DscpTos::AssuredForwarding {
                class,
                drop_probability,
            } => (class << 3) | drop_probability as u8,
            DscpTos::ExpeditedForwarding => 0b101110,
        };

        socket.set_tos((tos << 2) as u32).ok();
    }
}
