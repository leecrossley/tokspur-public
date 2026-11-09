//! IPv4 loopback listener calls for `native/src/http.cpp`. Address reuse and the bounded backlog are explicit so Rust and C++ apply the same restart and admission rules.

use std::io;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd};

use crate::ffi;

/// Makes closing this socket abort the TCP connection rather than drain it.
pub fn abortive_close_on_drop(fd: BorrowedFd<'_>) -> io::Result<()> {
    let value = ffi::Linger {
        enabled: 1,
        seconds: 0,
    };
    // SAFETY: the descriptor is borrowed and value is an initialised Darwin linger structure of the supplied size.
    let result = unsafe {
        ffi::setsockopt(
            fd.as_raw_fd(),
            ffi::SOL_SOCKET,
            ffi::SO_LINGER,
            std::ptr::from_ref(&value).cast(),
            size_of::<ffi::Linger>() as ffi::SocklenT,
        )
    };
    if result < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

/// `socket(AF_INET, SOCK_STREAM, 0)`.
pub fn tcp_socket() -> io::Result<OwnedFd> {
    // SAFETY: socket has no memory preconditions.
    let fd = unsafe { ffi::socket(ffi::AF_INET, ffi::SOCK_STREAM, 0) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: a fresh descriptor nothing else owns.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

fn loopback(port: u16) -> ffi::SockaddrIn {
    ffi::SockaddrIn {
        sin_len: size_of::<ffi::SockaddrIn>() as u8,
        sin_family: ffi::AF_INET as u8,
        sin_port: port.to_be(),
        sin_addr: ffi::InAddr {
            s_addr: ffi::INADDR_LOOPBACK.to_be(),
        },
        sin_zero: [0; 8],
    }
}

/// `bind` to 127.0.0.1:`port`.
pub fn bind_loopback(fd: BorrowedFd<'_>, port: u16) -> io::Result<()> {
    let address = loopback(port);
    // SAFETY: the address points to one initialised sockaddr_in of the size passed.
    let result = unsafe {
        ffi::bind(
            fd.as_raw_fd(),
            &address,
            size_of::<ffi::SockaddrIn>() as ffi::SocklenT,
        )
    };
    if result < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

/// `listen(fd, backlog)`.
pub fn listen(fd: BorrowedFd<'_>, backlog: i32) -> io::Result<()> {
    // SAFETY: listen has no memory preconditions.
    if unsafe { ffi::listen(fd.as_raw_fd(), backlog) } < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

/// An IPv4 socket address as the kernel reported it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ipv4Address {
    /// The `socklen_t` the kernel wrote.
    pub length: usize,
    pub family: u8,
    /// Host byte order.
    pub address: u32,
    /// Host byte order.
    pub port: u16,
}

impl Ipv4Address {
    pub const SOCKADDR_IN_BYTES: usize = size_of::<ffi::SockaddrIn>();
    pub const FAMILY_INET: u8 = ffi::AF_INET as u8;
    pub const LOOPBACK: u32 = ffi::INADDR_LOOPBACK;
}

fn decode(address: &ffi::SockaddrIn, length: ffi::SocklenT) -> Ipv4Address {
    Ipv4Address {
        length: length as usize,
        family: address.sin_family,
        address: u32::from_be(address.sin_addr.s_addr),
        port: u16::from_be(address.sin_port),
    }
}

/// `getsockname`.
pub fn local_address(fd: BorrowedFd<'_>) -> io::Result<Ipv4Address> {
    let mut address = loopback(0);
    let mut length = size_of::<ffi::SockaddrIn>() as ffi::SocklenT;
    // SAFETY: address/length point to writable storage of the stated size.
    if unsafe { ffi::getsockname(fd.as_raw_fd(), &mut address, &mut length) } < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(decode(&address, length))
}

/// `getpeername`.
pub fn peer_address(fd: BorrowedFd<'_>) -> io::Result<Ipv4Address> {
    let mut address = loopback(0);
    let mut length = size_of::<ffi::SockaddrIn>() as ffi::SocklenT;
    // SAFETY: address/length point to writable storage of the stated size.
    if unsafe { ffi::getpeername(fd.as_raw_fd(), &mut address, &mut length) } < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(decode(&address, length))
}

/// `bind` to `0.0.0.0:port`.
pub fn bind_any(fd: BorrowedFd<'_>, port: u16) -> io::Result<()> {
    let address = ffi::SockaddrIn {
        sin_len: size_of::<ffi::SockaddrIn>() as u8,
        sin_family: ffi::AF_INET as u8,
        sin_port: port.to_be(),
        sin_addr: ffi::InAddr {
            s_addr: ffi::INADDR_ANY.to_be(),
        },
        sin_zero: [0; 8],
    };
    // SAFETY: `address` points to one initialised `sockaddr_in` of the stated size.
    let result = unsafe {
        ffi::bind(
            fd.as_raw_fd(),
            &address,
            size_of::<ffi::SockaddrIn>() as ffi::SocklenT,
        )
    };
    if result < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

/// One `accept`, returning the connection and its peer address.
pub fn accept(fd: BorrowedFd<'_>) -> io::Result<(OwnedFd, Ipv4Address)> {
    let mut peer = loopback(0);
    let mut length = size_of::<ffi::SockaddrIn>() as ffi::SocklenT;
    // SAFETY: peer/length point to writable storage of the stated size.
    let accepted = unsafe { ffi::accept(fd.as_raw_fd(), &mut peer, &mut length) };
    if accepted < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: a fresh descriptor nothing else owns.
    let connection = unsafe { OwnedFd::from_raw_fd(accepted) };
    Ok((connection, decode(&peer, length)))
}

pub const SOCK_STREAM: i32 = ffi::SOCK_STREAM;

/// `setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, 1)`.
pub fn set_reuseaddr(fd: BorrowedFd<'_>) -> io::Result<()> {
    let yes: std::ffi::c_int = 1;
    // SAFETY: option value points to one readable int of the stated size.
    let rc = unsafe {
        ffi::setsockopt(
            fd.as_raw_fd(),
            ffi::SOL_SOCKET,
            ffi::SO_REUSEADDR,
            std::ptr::from_ref(&yes).cast(),
            size_of::<std::ffi::c_int>() as ffi::SocklenT,
        )
    };
    if rc < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn socket_error(fd: BorrowedFd<'_>) -> io::Result<i32> {
    let mut err: std::ffi::c_int = 0;
    let mut length = size_of::<std::ffi::c_int>() as ffi::SocklenT;
    // SAFETY: `err` and `length` point to writable storage of the stated size.
    let rc = unsafe {
        ffi::getsockopt(
            fd.as_raw_fd(),
            ffi::SOL_SOCKET,
            ffi::SO_ERROR,
            std::ptr::from_mut(&mut err).cast(),
            &mut length,
        )
    };
    if rc < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(err)
    }
}

/// Matches Python `tokspur.local.port_available(port)`: probes `127.0.0.1:port` with `SO_REUSEADDR` bind, then verifies no wildcard/dual-stack listener answers a 500 ms loopback `connect`.
pub fn port_available(port: u16) -> bool {
    if port == 0 {
        return false;
    }
    {
        let Ok(probe) = tcp_socket() else {
            return false;
        };
        let Ok(df) = crate::fd::descriptor_flags(probe.as_fd()) else {
            return false;
        };
        if crate::fd::set_descriptor_cloexec(probe.as_fd(), df).is_err() {
            return false;
        }
        if set_reuseaddr(probe.as_fd()).is_err() {
            return false;
        }
        if bind_loopback(probe.as_fd(), port).is_err() {
            return false;
        }
    }
    let Ok(client) = tcp_socket() else {
        return false;
    };
    let Ok(df) = crate::fd::descriptor_flags(client.as_fd()) else {
        return false;
    };
    if crate::fd::set_descriptor_cloexec(client.as_fd(), df).is_err() {
        return false;
    }
    if crate::fd::make_nonblocking(client.as_fd()).is_err() {
        return false;
    }
    let address = loopback(port);
    // SAFETY: `address` points to one initialised `sockaddr_in` of the stated size.
    let rc = unsafe {
        ffi::connect(
            client.as_raw_fd(),
            &address,
            size_of::<ffi::SockaddrIn>() as ffi::SocklenT,
        )
    };
    let is_live_peer = |fd: BorrowedFd<'_>| -> bool {
        let Ok(peer) = peer_address(fd) else {
            return false;
        };
        if let Ok(local) = local_address(fd)
            && local.address == peer.address
            && local.port == peer.port
        {
            return false;
        }
        true
    };
    if rc == 0 {
        return !is_live_peer(client.as_fd());
    }
    let err = ffi::errno();
    if err != ffi::EINPROGRESS {
        return true;
    }
    let mut fds = [ffi::PollFd {
        fd: client.as_raw_fd(),
        events: ffi::POLLOUT,
        revents: 0,
    }];
    match crate::fd::poll(&mut fds, 500) {
        Ok(0) | Err(_) => true,
        Ok(_) => {
            let revents = fds[0].revents;
            if (revents & (ffi::POLLERR | ffi::POLLHUP | ffi::POLLNVAL)) != 0
                || (revents & ffi::POLLOUT) == 0
            {
                return true;
            }
            match socket_error(client.as_fd()) {
                Ok(0) => !is_live_peer(client.as_fd()),
                Ok(_) | Err(_) => true,
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn loopback_port_availability_tracks_listener_lifecycle() {
        for iter in 0..50 {
            let listener = tcp_socket().unwrap();
            bind_loopback(listener.as_fd(), 0).unwrap();
            listen(listener.as_fd(), 1).unwrap();
            let port = local_address(listener.as_fd()).unwrap().port;
            assert!(!port_available(port), "iter={iter} occupied port {port}");
            drop(listener);
            if !port_available(port) {
                let p = tcp_socket().unwrap();
                let r_reuse = set_reuseaddr(p.as_fd());
                let r_bind = bind_loopback(p.as_fd(), port);
                panic!("iter={iter} port={port} reuse={r_reuse:?} bind={r_bind:?}");
            }
        }
    }
}
