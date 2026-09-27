use crate::LOCAL_IP;

use super::{SocketReader, SocketWriter};
use alvr_common::{anyhow::Result, ConResult, HandleTryAgain};
use alvr_session::{DscpTos, SocketBufferSize};
use socket2::{MaybeUninitSlice, Socket};
use std::{
    ffi::c_int,
    mem::MaybeUninit,
    net::{IpAddr, UdpSocket},
    ptr,
    time::Duration,
};

// Create tokio socket, convert to socket2, apply settings, convert back to tokio. This is done to
// let tokio set all the internal parameters it needs from the start.
pub fn bind(
    port: u16,
    dscp: Option<DscpTos>,
    send_buffer_bytes: SocketBufferSize,
    recv_buffer_bytes: SocketBufferSize,
) -> Result<UdpSocket> {
    let socket = UdpSocket::bind((LOCAL_IP, port))?;
    #[cfg(windows)]
    disable_udp_connreset(&socket);
    let socket: Socket = socket.into();

    crate::set_socket_buffers(&socket, send_buffer_bytes, recv_buffer_bytes).ok();

    crate::set_dscp(&socket, dscp);

    Ok(socket.into())
}

// Windows reports an ICMP "port unreachable" for an earlier datagram as WSAECONNRESET on the
// NEXT send or receive of the socket. A client that is not listening yet when the stream starts,
// or restarts its socket, then makes an unrelated send fail. SIO_UDP_CONNRESET = FALSE turns that
// off; UDP has no connection to reset anyway.
#[cfg(windows)]
fn disable_udp_connreset(socket: &UdpSocket) {
    use std::{ffi::c_void, os::windows::io::AsRawSocket};

    #[link(name = "ws2_32")]
    extern "system" {
        fn WSAIoctl(
            s: usize,
            code: u32,
            in_buf: *const c_void,
            in_len: u32,
            out_buf: *mut c_void,
            out_len: u32,
            bytes_returned: *mut u32,
            overlapped: *mut c_void,
            completion: *mut c_void,
        ) -> i32;
    }
    const SIO_UDP_CONNRESET: u32 = 0x9800_000C;

    let report: u32 = 0; // FALSE
    let mut returned: u32 = 0;
    unsafe {
        WSAIoctl(
            socket.as_raw_socket() as usize,
            SIO_UDP_CONNRESET,
            &report as *const u32 as *const c_void,
            4,
            ptr::null_mut(),
            0,
            &mut returned,
            ptr::null_mut(),
            ptr::null_mut(),
        );
    }
}

pub fn connect(
    socket: &UdpSocket,
    peer_ip: IpAddr,
    port: u16,
    timeout: Duration,
) -> Result<(UdpSocket, Socket)> {
    socket.connect((peer_ip, port))?;
    socket.set_read_timeout(Some(timeout))?;

    Ok((socket.try_clone()?, socket.try_clone()?.into()))
}

impl SocketWriter for UdpSocket {
    fn send(&mut self, buffer: &[u8]) -> Result<()> {
        UdpSocket::send(self, buffer)?;

        Ok(())
    }

    #[cfg(windows)]
    fn send_segmented(&mut self, buffer: &[u8], segment_size: usize) -> Result<bool> {
        use crate::{
            UDP_PAYLOAD_PER_FRAME, USO_ABOVE_MTU, USO_FALLBACK_CALLS, USO_LAST_ERROR, USO_OFF,
            USO_ON, USO_RETRIED_OK, USO_SEGMENT, USO_STATE, USO_UNSUPPORTED,
            USO_UNSUPPORTED_STREAK,
        };
        use std::{os::windows::io::AsRawSocket, sync::atomic::Ordering};

        #[link(name = "ws2_32")]
        extern "system" {
            fn setsockopt(s: usize, level: i32, name: i32, value: *const u8, len: i32) -> i32;
        }
        const IPPROTO_UDP: i32 = 17;
        const UDP_SEND_MSG_SIZE: i32 = 2;

        match USO_STATE.load(Ordering::Relaxed) {
            USO_OFF | USO_UNSUPPORTED | USO_ABOVE_MTU => return Ok(false),
            _ if segment_size > UDP_PAYLOAD_PER_FRAME => {
                USO_STATE.store(USO_ABOVE_MTU, Ordering::Relaxed);
                return Ok(false);
            }
            USO_ON if USO_SEGMENT.load(Ordering::Relaxed) == segment_size => (),
            _ => {
                // First use, or a different segment size: set it on the socket. It stays set,
                // and a send no larger than one segment still goes out as a single datagram.
                let size = segment_size as u32;
                let rc = unsafe {
                    setsockopt(
                        self.as_raw_socket() as usize,
                        IPPROTO_UDP,
                        UDP_SEND_MSG_SIZE,
                        &size as *const u32 as *const u8,
                        4,
                    )
                };
                if rc != 0 {
                    USO_STATE.store(USO_UNSUPPORTED, Ordering::Relaxed);
                    return Ok(false);
                }
                USO_SEGMENT.store(segment_size, Ordering::Relaxed);
                USO_STATE.store(USO_ON, Ordering::Relaxed);
            }
        }

        // Errors that say the stack does not support segmented sends at all. Everything else
        // (WSAENOBUFS under a burst, WSAEWOULDBLOCK, WSAEINTR, a WSAECONNRESET left behind by an
        // ICMP "port unreachable") is momentary and must not switch segmentation off for the rest
        // of the session -- that is what put the streamer into its slow state (see lib.rs).
        const WSAEINVAL: i32 = 10022;
        const WSAEMSGSIZE: i32 = 10040;
        const WSAENOPROTOOPT: i32 = 10042;
        const WSAEOPNOTSUPP: i32 = 10045;
        const WSAEINTR: i32 = 10004;
        const WSAEWOULDBLOCK: i32 = 10035;
        const WSAENOBUFS: i32 = 10055;
        const WSAECONNRESET: i32 = 10054;

        let mut last_code = 0;
        for attempt in 0..3 {
            match UdpSocket::send(self, buffer) {
                Ok(_) => {
                    USO_UNSUPPORTED_STREAK.store(0, Ordering::Relaxed);
                    if attempt > 0 {
                        USO_RETRIED_OK.fetch_add(1, Ordering::Relaxed);
                    }
                    return Ok(true);
                }
                Err(e) => {
                    last_code = e.raw_os_error().unwrap_or(-1);
                    USO_LAST_ERROR.store(last_code, Ordering::Relaxed);
                    match last_code {
                        // Nothing went out; a moment later there is room again.
                        WSAENOBUFS | WSAEWOULDBLOCK | WSAEINTR => {
                            std::thread::sleep(std::time::Duration::from_micros(50));
                        }
                        // Reported for an earlier datagram, not this one: send again right away.
                        WSAECONNRESET => (),
                        _ => break,
                    }
                }
            }
        }

        // Nothing of this call went out, so the caller sends these shards one by one. Only three
        // "not supported" answers in a row switch segmentation off.
        USO_FALLBACK_CALLS.fetch_add(1, Ordering::Relaxed);
        if matches!(
            last_code,
            WSAEINVAL | WSAEMSGSIZE | WSAENOPROTOOPT | WSAEOPNOTSUPP
        ) && USO_UNSUPPORTED_STREAK.fetch_add(1, Ordering::Relaxed) + 1 >= 3
        {
            USO_STATE.store(USO_UNSUPPORTED, Ordering::Relaxed);
        }
        Ok(false)
    }
}

impl SocketReader for Socket {
    fn recv(&mut self, buffer: &mut [u8]) -> ConResult<usize> {
        Socket::recv(self, unsafe {
            &mut *(ptr::from_mut(buffer) as *mut [MaybeUninit<u8>])
        })
        .handle_try_again()
    }

    fn peek(&self, buffer: &mut [u8]) -> ConResult<usize> {
        #[cfg(windows)]
        const FLAGS: c_int = 0x02 | 0x8000; // MSG_PEEK | MSG_PARTIAL
        #[cfg(not(windows))]
        const FLAGS: c_int = 0x02 | 0x20; // MSG_PEEK | MSG_TRUNC

        let buffer = MaybeUninitSlice::new(unsafe {
            &mut *(ptr::from_mut(buffer) as *mut [MaybeUninit<u8>])
        });
        Ok(self
            .recv_vectored_with_flags(&mut [buffer], FLAGS)
            .handle_try_again()?
            .0)
    }
}
