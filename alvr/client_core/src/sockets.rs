use alvr_common::{anyhow::Result, ALVR_NAME};
use alvr_sockets::CONTROL_PORT;
use std::net::{IpAddr, Ipv4Addr, UdpSocket};

pub struct AnnouncerSocket {
    socket: UdpSocket,
    packet: [u8; 56],
}

impl AnnouncerSocket {
    pub fn new(hostname: &str) -> Result<Self> {
        // Bind to the machine's actual LAN address rather than the
        // wildcard 0.0.0.0 (alvr_sockets::LOCAL_IP) -- on a host with
        // multiple network interfaces/routes (observed on macOS with
        // active VPN/utun interfaces alongside a real Wi-Fi/Ethernet
        // link), binding the wildcard address leaves the OS to guess
        // which interface a subsequent send to the global broadcast
        // address (255.255.255.255) should go out on, which can fail
        // outright with "No route to host" rather than just picking a
        // (possibly wrong) default. Binding the real local address up
        // front removes that ambiguity. Falls back to the wildcard
        // address (previous behavior) if it can't be determined.
        let local_addr = alvr_system_info::local_ip();
        let bind_addr = if matches!(local_addr, IpAddr::V4(_)) {
            local_addr
        } else {
            IpAddr::V4(Ipv4Addr::UNSPECIFIED)
        };
        let socket = UdpSocket::bind((bind_addr, CONTROL_PORT))?;
        socket.set_broadcast(true)?;

        let mut packet = [0; 56];
        packet[0..ALVR_NAME.len()].copy_from_slice(ALVR_NAME.as_bytes());
        packet[16..24].copy_from_slice(&alvr_common::protocol_id_u64().to_le_bytes());
        packet[24..24 + hostname.len()].copy_from_slice(hostname.as_bytes());

        Ok(Self { socket, packet })
    }

    pub fn announce_broadcast(&self) -> Result<()> {
        self.socket
            .send_to(&self.packet, (Ipv4Addr::BROADCAST, CONTROL_PORT))?;

        Ok(())
    }
}
