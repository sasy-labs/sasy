//! The evaluator's syscall deny-list, built here so bubblewrap can install it
//! *before* the evaluator's first instruction.
//!
//! The evaluator binary links caller-supplied functor C++. Static
//! initializers and `__attribute__((constructor))` functions in that C++ run
//! before `main`, so a filter the evaluator installs itself (the shim's
//! `apply_seccomp_denylist`, still there as defence in depth) arrives too late
//! for them: they run inside bwrap's namespaces with the whole syscall
//! surface, including the escape primitives the list exists to block. bwrap
//! takes the program on a file descriptor (`--seccomp FD`) and installs it
//! between the namespace setup and the `execve`, which closes that window.
//!
//! The program is the same deny-list the shim builds: an architecture check
//! that kills the process outright on a foreign architecture, then one
//! comparison per denied syscall returning `EPERM`, then allow. `EPERM`
//! rather than a kill so an unexpected match degrades instead of crashing a
//! live evaluator.
//!
//! Linux-only. Everywhere else [`deny_list_program`] returns `None` and the
//! caller must use its platform-specific isolation path.

#[cfg(unix)]
use std::os::unix::io::RawFd;
#[cfg(not(unix))]
type RawFd = i32;

/// The syscalls the evaluator may never make, by name.
///
/// Kept as names, on every platform, so the parity test against the shim's
/// `denied[]` initializer can run on any host — the numbers behind them exist
/// only on Linux. The list is the shim's, verbatim; the two must not drift,
/// because the in-process copy is what confines the evaluator on a host whose
/// bubblewrap is too old to take the program before `execve`.
pub const DENIED_SYSCALL_NAMES: &[&str] = &[
    "ptrace",
    "mount",
    "pivot_root",
    "setns",
    "unshare",
    "bpf",
    "perf_event_open",
    "keyctl",
    "add_key",
    "request_key",
    "reboot",
    "kexec_load",
    "kexec_file_load",
    "umount2",
    "init_module",
    "finit_module",
    "delete_module",
    "process_vm_readv",
    "process_vm_writev",
    "open_by_handle_at",
    "pidfd_getfd",
];

#[cfg(target_os = "linux")]
mod imp {
    use super::DENIED_SYSCALL_NAMES;

    // Classic-BPF opcode pieces. `libc` exposes the seccomp structs and return
    // values but not these, so they are spelled out here with the values from
    // `linux/bpf_common.h`.
    pub(super) const BPF_LD: u16 = 0x00;
    pub(super) const BPF_W: u16 = 0x00;
    pub(super) const BPF_ABS: u16 = 0x20;
    pub(super) const BPF_JMP: u16 = 0x05;
    pub(super) const BPF_JEQ: u16 = 0x10;
    pub(super) const BPF_K: u16 = 0x00;
    pub(super) const BPF_RET: u16 = 0x06;

    /// Offsets into `struct seccomp_data`: the syscall number first, then the
    /// architecture.
    pub(super) const SECCOMP_DATA_NR_OFFSET: u32 = 0;
    pub(super) const SECCOMP_DATA_ARCH_OFFSET: u32 = 4;

    pub(super) const SECCOMP_RET_KILL_PROCESS: u32 = 0x8000_0000;
    pub(super) const SECCOMP_RET_ERRNO: u32 = 0x0005_0000;
    pub(super) const SECCOMP_RET_ALLOW: u32 = 0x7fff_0000;
    pub(super) const SECCOMP_RET_DATA: u32 = 0x0000_ffff;

    /// `AUDIT_ARCH_*` for the architecture this binary is built for, from
    /// `linux/audit.h`. `None` on any other architecture: a deny-list keyed to
    /// the wrong `AUDIT_ARCH` would kill every evaluator, so the caller falls
    /// back to the in-process filter instead.
    pub(super) fn audit_arch() -> Option<u32> {
        #[cfg(target_arch = "x86_64")]
        {
            Some(0xc000_003e)
        }
        #[cfg(target_arch = "aarch64")]
        {
            Some(0xc000_00b7)
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            None
        }
    }

    /// The denied syscalls as (name, number) for this target.
    ///
    /// A name the `libc` crate does not define for the target is skipped
    /// rather than failing the build: the pre-exec filter is best-effort per
    /// syscall, and the shim's own list skips the same way with `#ifdef`.
    pub(super) fn denied_syscalls() -> Vec<(&'static str, libc::c_long)> {
        let mut out = Vec::with_capacity(DENIED_SYSCALL_NAMES.len());
        for name in DENIED_SYSCALL_NAMES {
            if let Some(nr) = syscall_number(name) {
                out.push((*name, nr));
            }
        }
        out
    }

    fn syscall_number(name: &str) -> Option<libc::c_long> {
        let nr = match name {
            "ptrace" => libc::SYS_ptrace,
            "mount" => libc::SYS_mount,
            "pivot_root" => libc::SYS_pivot_root,
            "setns" => libc::SYS_setns,
            "unshare" => libc::SYS_unshare,
            "bpf" => libc::SYS_bpf,
            "perf_event_open" => libc::SYS_perf_event_open,
            "keyctl" => libc::SYS_keyctl,
            "add_key" => libc::SYS_add_key,
            "request_key" => libc::SYS_request_key,
            "reboot" => libc::SYS_reboot,
            "kexec_load" => libc::SYS_kexec_load,
            "kexec_file_load" => libc::SYS_kexec_file_load,
            "umount2" => libc::SYS_umount2,
            "init_module" => libc::SYS_init_module,
            "finit_module" => libc::SYS_finit_module,
            "delete_module" => libc::SYS_delete_module,
            "process_vm_readv" => libc::SYS_process_vm_readv,
            "process_vm_writev" => libc::SYS_process_vm_writev,
            "open_by_handle_at" => libc::SYS_open_by_handle_at,
            "pidfd_getfd" => libc::SYS_pidfd_getfd,
            _ => return None,
        };
        Some(nr)
    }

    fn stmt(code: u16, k: u32) -> libc::sock_filter {
        libc::sock_filter {
            code,
            jt: 0,
            jf: 0,
            k,
        }
    }

    fn jump(code: u16, k: u32, jt: u8, jf: u8) -> libc::sock_filter {
        libc::sock_filter { code, jt, jf, k }
    }

    /// The deny-list as classic-BPF instructions, in the shim's shape: load
    /// `arch`, kill on a foreign one, load `nr`, one equality jump per denied
    /// syscall to the trailing `EPERM` return, otherwise allow.
    pub(super) fn deny_list_filter() -> Option<Vec<libc::sock_filter>> {
        let arch = audit_arch()?;
        let denied = denied_syscalls();
        let n = denied.len();
        let mut bpf = Vec::with_capacity(6 + n);
        bpf.push(stmt(BPF_LD | BPF_W | BPF_ABS, SECCOMP_DATA_ARCH_OFFSET));
        bpf.push(jump(BPF_JMP | BPF_JEQ | BPF_K, arch, 1, 0));
        bpf.push(stmt(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS));
        bpf.push(stmt(BPF_LD | BPF_W | BPF_ABS, SECCOMP_DATA_NR_OFFSET));
        for (i, (_, nr)) in denied.iter().enumerate() {
            // Skip the remaining comparisons and the ALLOW that follows them,
            // landing on the ERRNO return at the end.
            let jt = u8::try_from(n - i).ok()?;
            bpf.push(jump(BPF_JMP | BPF_JEQ | BPF_K, *nr as u32, jt, 0));
        }
        bpf.push(stmt(BPF_RET | BPF_K, SECCOMP_RET_ALLOW));
        bpf.push(stmt(
            BPF_RET | BPF_K,
            SECCOMP_RET_ERRNO | (libc::EPERM as u32 & SECCOMP_RET_DATA),
        ));
        Some(bpf)
    }

    /// The filter as the bytes bubblewrap expects on `--seccomp FD`: the
    /// `sock_filter` array exactly as it sits in memory, host endianness,
    /// which is what the kernel is handed.
    pub(super) fn deny_list_program() -> Option<Vec<u8>> {
        let filter = deny_list_filter()?;
        let len = std::mem::size_of_val(&filter[..]);
        // SAFETY: `sock_filter` is a `repr(C)` struct of plain integers with no
        // padding, so its array is a valid byte sequence of exactly this length.
        let bytes = unsafe { std::slice::from_raw_parts(filter.as_ptr() as *const u8, len) };
        Some(bytes.to_vec())
    }

    /// Fill a descriptor with the program: an anonymous file where `memfd` is
    /// available, a pipe otherwise. Every step is checked, and a failure at
    /// any of them closes what was opened and answers `None`.
    ///
    /// `FD_CLOEXEC` is left set. It is cleared in the child of the one spawn
    /// this descriptor was built for, from that spawn's `pre_exec` hook, so
    /// no other fork anywhere in the process ever sees the number.
    pub(super) fn deny_list_program_fd() -> Option<super::ProgramFd> {
        program_fd(true)
    }

    /// Test-only: the pipe fallback on a host whose kernel does have
    /// `memfd_create`, so the fallback's own descriptor can be examined
    /// without an `LD_PRELOAD` that breaks `memfd_create`.
    #[cfg(test)]
    pub(super) fn deny_list_program_fd_on_the_pipe_fallback() -> Option<super::ProgramFd> {
        program_fd(false)
    }

    fn program_fd(may_use_memfd: bool) -> Option<super::ProgramFd> {
        let bytes = deny_list_program()?;
        let from_memfd = if may_use_memfd { memfd(&bytes) } else { None };
        let fd = match from_memfd {
            Some(fd) => fd,
            None => pipe_with(&bytes)?,
        };
        Some(super::ProgramFd { fd })
    }

    fn memfd(bytes: &[u8]) -> Option<libc::c_int> {
        // An empty name: the name is only ever seen in /proc, and an empty one
        // is what the kernel allows.
        let name = c"";
        // SAFETY: `name` is a valid NUL-terminated string; the call returns a
        // descriptor or -1 and touches nothing else. `MFD_ALLOW_SEALING` is
        // what lets `seal` freeze the program below.
        let fd = unsafe {
            libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING)
        };
        if fd < 0 {
            return None;
        }
        if write_all(fd, bytes).is_none() {
            // SAFETY: `fd` was just created here and is not shared.
            unsafe { libc::close(fd) };
            return None;
        }
        if seal(fd).is_none() {
            // SAFETY: as above.
            unsafe { libc::close(fd) };
            return None;
        }
        // SAFETY: `fd` is open and owned here; seeking a memfd to its start is
        // what leaves bwrap's read at the first instruction.
        if unsafe { libc::lseek(fd, 0, libc::SEEK_SET) } != 0 {
            // SAFETY: as above.
            unsafe { libc::close(fd) };
            return None;
        }
        Some(fd)
    }

    /// Freeze the anonymous file's contents and its size, for good. Seals
    /// cannot be lifted, so from here on no descriptor on this file — this
    /// one, or one another process was handed — can change a byte of the
    /// program bwrap is about to install; `F_SEAL_SEAL` stops the set of
    /// seals being widened later.
    fn seal(fd: libc::c_int) -> Option<()> {
        let seals =
            libc::F_SEAL_WRITE | libc::F_SEAL_SHRINK | libc::F_SEAL_GROW | libc::F_SEAL_SEAL;
        // SAFETY: adding seals to an owned memfd created with
        // `MFD_ALLOW_SEALING`; the call touches nothing else.
        if unsafe { libc::fcntl(fd, libc::F_ADD_SEALS, seals) } < 0 {
            return None;
        }
        Some(())
    }

    /// The fallback: the program in a pipe, write end closed so bwrap's read
    /// reaches end-of-file. Sized for a program of a few hundred bytes, well
    /// under a pipe buffer, so the write completes with no reader.
    ///
    /// Both ends are created close-on-exec, in the one call that creates
    /// them: a plain `pipe(2)` leaves the flag clear, and any fork+exec
    /// racing this one — a concurrent evaluator spawn, anything else the
    /// binary starts — would then inherit the program the memfd path is
    /// careful never to leak. `pipe2` is the whole point; the `fcntl` pair is
    /// there for a kernel too old to have it, where the window between the
    /// two calls is all this fallback can do.
    fn pipe_with(bytes: &[u8]) -> Option<libc::c_int> {
        let mut fds = [0 as libc::c_int; 2];
        // SAFETY: `fds` is a two-element array, which is what pipe2(2) fills.
        if unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
            // SAFETY: as above; `pipe2` may be missing (ENOSYS) on an old
            // kernel, and then the flag is set on each end below.
            if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
                return None;
            }
            for fd in fds {
                // SAFETY: reading and setting the flags of a descriptor
                // created just above and owned here.
                let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
                if flags < 0
                    || unsafe { libc::fcntl(fd, libc::F_SETFD, flags | libc::FD_CLOEXEC) } < 0
                {
                    // SAFETY: both ends are owned here and abandoned on
                    // failure; a descriptor that cannot be made
                    // close-on-exec is not one to hand out.
                    unsafe {
                        libc::close(fds[0]);
                        libc::close(fds[1]);
                    }
                    return None;
                }
            }
        }
        let (read_fd, write_fd) = (fds[0], fds[1]);
        let wrote = write_all(write_fd, bytes);
        // SAFETY: the write end is owned here and finished with either way.
        unsafe { libc::close(write_fd) };
        if wrote.is_none() {
            // SAFETY: the read end is owned here and abandoned on failure.
            unsafe { libc::close(read_fd) };
            return None;
        }
        Some(read_fd)
    }

    fn write_all(fd: libc::c_int, bytes: &[u8]) -> Option<()> {
        let mut written = 0usize;
        while written < bytes.len() {
            // SAFETY: writing `bytes.len() - written` bytes from inside the
            // slice to a descriptor owned by the caller.
            let n = unsafe {
                libc::write(
                    fd,
                    bytes[written..].as_ptr() as *const libc::c_void,
                    bytes.len() - written,
                )
            };
            if n <= 0 {
                return None;
            }
            written += n as usize;
        }
        Some(())
    }

    /// Called after `fork` and before `execve`, in the child, so only that
    /// child inherits the descriptor. Nothing here allocates or takes a lock:
    /// two `fcntl` calls, which is async-signal-safe.
    pub(super) fn clear_cloexec(fd: libc::c_int) -> Option<()> {
        // SAFETY: reading and setting the descriptor flags of an owned fd.
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
        if flags < 0 {
            return None;
        }
        // SAFETY: as above.
        if unsafe { libc::fcntl(fd, libc::F_SETFD, flags & !libc::FD_CLOEXEC) } < 0 {
            return None;
        }
        Some(())
    }
}

/// The deny-list as the raw bytes of a `sock_filter` array, ready to hand to
/// `bwrap --seccomp FD`.
///
/// `None` when there is nothing to install before `execve` — a non-Linux host,
/// or a Linux architecture this module has no `AUDIT_ARCH` for — and the
/// caller then spawns without the flag and leaves the confinement to the
/// evaluator's own in-process filter.
pub fn deny_list_program() -> Option<Vec<u8>> {
    #[cfg(target_os = "linux")]
    {
        imp::deny_list_program()
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// A descriptor the deny-list has been written to, positioned at its start,
/// for `bwrap --seccomp FD`.
///
/// It carries `FD_CLOEXEC` while it lives in the parent, so no other spawn
/// inherits it; the spawn it was built for clears that flag in its own child
/// with [`inherit_in_child`], from a `pre_exec` hook, which is
/// what lets it survive bwrap's `execve`.
///
/// Owned: the parent closes it when this value is dropped, which the spawn
/// site does as soon as the child has been started. Nothing else may close
/// the number, so the type does not hand out ownership of it.
pub struct ProgramFd {
    fd: RawFd,
}

impl ProgramFd {
    /// The descriptor number to name in bwrap's argv. Valid only while this
    /// value is alive.
    pub fn raw(&self) -> RawFd {
        self.fd
    }

    /// Test-only: take ownership of a descriptor made elsewhere, so a test
    /// can hand the sandbox probe one bwrap cannot read. The value closes it
    /// on drop, like any other. Linux-only, because the probe it feeds needs
    /// a bwrap to refuse the descriptor.
    #[cfg(all(test, target_os = "linux"))]
    pub(crate) fn from_raw_for_test(fd: RawFd) -> Self {
        Self { fd }
    }
}

/// Clear `FD_CLOEXEC` on a [`ProgramFd`]'s descriptor, in the forked child and
/// before its `execve`, so that this one child — and no other fork in the
/// process — inherits the program. Answers whether the flag was cleared; a
/// `false` means bwrap will not find the descriptor, and the spawn site fails
/// the child rather than exec an evaluator with no filter.
///
/// # Safety
///
/// To be called only between `fork` and `execve`, on a descriptor the parent
/// still holds open: it does no allocation and takes no lock, which is what
/// makes it safe there, and it changes a descriptor flag the parent is
/// relying on if called anywhere else.
pub unsafe fn inherit_in_child(fd: RawFd) -> bool {
    #[cfg(target_os = "linux")]
    {
        imp::clear_cloexec(fd).is_some()
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = fd;
        false
    }
}

impl Drop for ProgramFd {
    fn drop(&mut self) {
        #[cfg(unix)]
        // SAFETY: this type owns the descriptor — it is created in
        // `deny_list_program_fd` and handed to no one else — and is closed
        // exactly once, here.
        unsafe {
            libc::close(self.fd);
        }
    }
}

/// The deny-list, written to a descriptor bwrap can read before it `execve`s
/// the evaluator.
///
/// An anonymous in-memory file (`memfd_create`) where the kernel has one: it
/// needs no path, no directory the sandbox has to be given, and nothing on
/// disk for another process to swap. A pipe is the fallback where `memfd` is
/// unavailable — the program is a few hundred bytes, far inside a pipe's
/// buffer, so the write cannot block on a reader that does not exist yet.
///
/// The program has to survive bwrap's `execve`, and two things keep that from
/// widening into a way for one confined evaluator to rewrite another's filter.
/// The descriptor keeps `FD_CLOEXEC` here in the parent, so a fork racing this
/// spawn does not inherit it at all; only the child of the spawn it was built
/// for clears the flag, from that spawn's `pre_exec` hook
/// ([`inherit_in_child`]). And the memfd is sealed
/// (`F_SEAL_WRITE | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL`) before it can
/// be inherited by anything, so even a descriptor that did leak is read-only
/// on contents the kernel will not let anyone change again — the seals cannot
/// be lifted. The pipe fallback hands out a read end, which cannot be written
/// either.
///
/// `None` on any host with no program to install ([`deny_list_program`]) and
/// on any failure to create or fill the descriptor: the caller then spawns
/// without the flag and the evaluator installs its own copy from `main`.
pub fn deny_list_program_fd() -> Option<ProgramFd> {
    #[cfg(target_os = "linux")]
    {
        imp::deny_list_program_fd()
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The names inside a `static const int denied[] = { … };` initializer,
    /// `SYS_` prefix stripped.
    ///
    /// Only array elements count. A line whose first non-space characters
    /// are `#` or `//` names no syscall the shim blocks: `#ifdef SYS_umount2`
    /// guards an element and `// SYS_bpf,` is one that was taken out, so a
    /// scraper that read either would still see the name after the element
    /// itself was gone — which is the whole failure this comparison exists to
    /// catch.
    fn denied_names_in(src: &str) -> BTreeSet<String> {
        let start = src
            .find("static const int denied[] = {")
            .expect("the shim's denied[] initializer");
        let rest = &src[start..];
        let end = rest
            .find("};")
            .expect("the end of the denied[] initializer");
        rest[..end]
            .lines()
            .filter(|line| {
                let l = line.trim_start();
                !(l.starts_with('#') || l.starts_with("//"))
            })
            .flat_map(|line| line.split(|c: char| !(c.is_alphanumeric() || c == '_')))
            .filter_map(|tok| tok.strip_prefix("SYS_"))
            .map(str::to_string)
            .collect()
    }

    /// The names inside the shim's own `denied[]` initializer.
    fn shim_denied_names() -> BTreeSet<String> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../souffle/evaluator_shim.cpp");
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
        denied_names_in(&src)
    }

    /// The scraper on a list with one element deleted from between its
    /// `#ifdef` guards and one commented out: neither name survives, so the
    /// comparison against the Rust list fails, which is what it must do.
    #[test]
    fn a_deleted_entry_and_a_commented_out_one_both_disappear_from_the_scrape() {
        let intact = concat!(
            "static const int denied[] = {\n",
            "    SYS_ptrace,\n",
            "#ifdef SYS_umount2\n",
            "    SYS_umount2,\n",
            "#endif\n",
            "    SYS_bpf,\n",
            "};\n",
        );
        let names = |s: &[&str]| {
            s.iter()
                .map(|n| n.to_string())
                .collect::<BTreeSet<String>>()
        };
        assert_eq!(
            denied_names_in(intact),
            names(&["ptrace", "umount2", "bpf"])
        );

        let cut = concat!(
            "static const int denied[] = {\n",
            "    SYS_ptrace,\n",
            "#ifdef SYS_umount2\n",
            "#endif\n",
            "    // SYS_bpf,\n",
            "};\n",
        );
        assert_eq!(
            denied_names_in(cut),
            names(&["ptrace"]),
            "a guard line and a commented-out line are not entries"
        );
    }

    #[test]
    fn the_rust_deny_list_names_the_same_syscalls_as_the_shim() {
        let ours: BTreeSet<String> = DENIED_SYSCALL_NAMES.iter().map(|s| s.to_string()).collect();
        assert_eq!(
            ours,
            shim_denied_names(),
            "the pre-exec deny-list and the shim's in-process deny-list must block the same syscalls"
        );
    }

    #[test]
    fn no_syscall_is_named_twice() {
        let unique: BTreeSet<&&str> = DENIED_SYSCALL_NAMES.iter().collect();
        assert_eq!(unique.len(), DENIED_SYSCALL_NAMES.len());
    }

    #[cfg(target_os = "linux")]
    mod linux {
        use super::super::imp::*;
        use std::collections::BTreeSet;

        /// Read the program back out of the bytes and describe what it does:
        /// the architecture it is keyed to, the syscall numbers it refuses,
        /// and the action it takes on everything else.
        fn decode(bytes: &[u8]) -> (u32, BTreeSet<u32>, u32) {
            assert_eq!(bytes.len() % 8, 0, "a sock_filter is 8 bytes wide");
            let insns: Vec<(u16, u8, u8, u32)> = bytes
                .chunks_exact(8)
                .map(|c| {
                    (
                        u16::from_ne_bytes([c[0], c[1]]),
                        c[2],
                        c[3],
                        u32::from_ne_bytes([c[4], c[5], c[6], c[7]]),
                    )
                })
                .collect();
            assert_eq!(insns[0].0, BPF_LD | BPF_W | BPF_ABS);
            assert_eq!(insns[0].3, SECCOMP_DATA_ARCH_OFFSET);
            let arch = insns[1].3;
            assert_eq!(insns[2].0, BPF_RET | BPF_K);
            assert_eq!(
                insns[2].3, SECCOMP_RET_KILL_PROCESS,
                "a foreign architecture is killed, not merely refused"
            );
            assert_eq!(insns[3].3, SECCOMP_DATA_NR_OFFSET);
            let mut denied = BTreeSet::new();
            for (i, insn) in insns.iter().enumerate().skip(4) {
                if insn.0 == BPF_JMP | BPF_JEQ | BPF_K {
                    denied.insert(insn.3);
                    let target = i + 1 + insn.1 as usize;
                    assert_eq!(
                        insns[target].3,
                        SECCOMP_RET_ERRNO | (libc::EPERM as u32 & SECCOMP_RET_DATA),
                        "a match jumps to the EPERM return"
                    );
                }
            }
            let default = insns[insns.len() - 2].3;
            (arch, denied, default)
        }

        #[test]
        fn the_program_refuses_every_denied_syscall_and_allows_the_rest() {
            let bytes = super::super::deny_list_program().expect("a program on this architecture");
            let (arch, denied, default) = decode(&bytes);
            assert_eq!(arch, audit_arch().unwrap());
            let expected: BTreeSet<u32> =
                denied_syscalls().iter().map(|(_, n)| *n as u32).collect();
            assert_eq!(denied, expected);
            assert_eq!(default, SECCOMP_RET_ALLOW);
        }

        /// What bwrap reads off the descriptor must be the program, from its
        /// first instruction.
        #[test]
        fn the_descriptor_holds_the_program() {
            let program = super::super::deny_list_program().expect("a program");
            let handle = super::super::deny_list_program_fd().expect("a descriptor");
            let mut read_back = vec![0u8; program.len() + 8];
            let n = unsafe {
                libc::read(
                    handle.raw(),
                    read_back.as_mut_ptr() as *mut libc::c_void,
                    read_back.len(),
                )
            };
            assert!(n > 0, "reading the descriptor: errno {}", unsafe {
                *libc::__errno_location()
            });
            read_back.truncate(n as usize);
            assert_eq!(read_back, program);
        }

        /// The parent must not leak the descriptor into forks it did not make
        /// it for: it carries `FD_CLOEXEC` until the child of its own spawn
        /// takes it off, which is the only thing that hands it to bwrap.
        #[test]
        fn only_the_child_it_was_made_for_inherits_the_descriptor() {
            for (path, handle) in constructed_both_ways() {
                let handle = handle.unwrap_or_else(|| panic!("a descriptor on the {path} path"));
                let flags = unsafe { libc::fcntl(handle.raw(), libc::F_GETFD) };
                assert_ne!(
                    flags & libc::FD_CLOEXEC,
                    0,
                    "on the {path} path the parent's descriptor must not survive \
                     anyone else's execve"
                );
                assert!(unsafe { super::super::inherit_in_child(handle.raw()) });
                let flags = unsafe { libc::fcntl(handle.raw(), libc::F_GETFD) };
                assert_eq!(
                    flags & libc::FD_CLOEXEC,
                    0,
                    "on the {path} path the spawn's own child must inherit it, or bwrap \
                     finds no program"
                );
            }
        }

        /// The descriptor as both constructors make it: the memfd the kernel
        /// gives us here, and the pipe fallback a kernel without
        /// `memfd_create` leaves. Both are asked the same questions, because
        /// a host that takes the fallback gets the same promises.
        fn constructed_both_ways() -> [(&'static str, Option<super::super::ProgramFd>); 2] {
            [
                ("memfd", super::super::deny_list_program_fd()),
                (
                    "pipe",
                    super::super::imp::deny_list_program_fd_on_the_pipe_fallback(),
                ),
            ]
        }

        /// What `/proc/self/fd/N` says the descriptor is: `memfd:` for the
        /// anonymous file, `pipe:[inode]` for the fallback. The inode is what
        /// tells this process's pipe apart from the pipes a child's own
        /// stdout and stderr are.
        fn fd_target(fd: libc::c_int) -> String {
            std::fs::read_link(format!("/proc/self/fd/{fd}"))
                .unwrap_or_else(|e| panic!("reading /proc/self/fd/{fd}: {e}"))
                .to_string_lossy()
                .into_owned()
        }

        /// The class the flag exists to close: while a program descriptor is
        /// alive, a child this process starts for some other reason must not
        /// find it among its own descriptors.
        #[test]
        fn an_unrelated_spawn_does_not_inherit_the_program() {
            for (path, handle) in constructed_both_ways() {
                let handle = handle.unwrap_or_else(|| panic!("a descriptor on the {path} path"));
                // The exact `/proc` target, not the kind: a child's own
                // stdout and stderr are pipes too, so only this pipe's inode
                // number distinguishes a leak from an ordinary descriptor.
                let target = fd_target(handle.raw());
                // `/proc` names an anonymous file `/memfd:<name> (deleted)` on
                // current kernels and `memfd:<name>` on older ones; a pipe is
                // `pipe:[inode]` everywhere. The kind is checked by content,
                // the leak below by the full target string.
                let expected = if path == "memfd" { "memfd:" } else { "pipe:" };
                assert!(target.contains(expected), "the {path} path made a {target}");
                let out = std::process::Command::new("/bin/sh")
                    .args(["-c", "ls -l /proc/self/fd"])
                    .output()
                    .expect("list a child's descriptors");
                let listing = String::from_utf8_lossy(&out.stdout);
                assert!(
                    !listing.contains(&target),
                    "a child that has nothing to do with the spawn inherited the \
                     {path} program ({target}): {listing}"
                );
            }
        }

        /// Sealed before anything else can hold it: even a process that did
        /// get the descriptor cannot rewrite the program bwrap installs.
        #[test]
        fn the_program_cannot_be_rewritten_through_the_descriptor() {
            let handle = super::super::deny_list_program_fd().expect("a descriptor");
            let seals = unsafe { libc::fcntl(handle.raw(), libc::F_GET_SEALS) };
            if seals < 0 {
                // The pipe fallback: a read end, which is unwritable anyway.
                return;
            }
            for seal in [
                libc::F_SEAL_WRITE,
                libc::F_SEAL_SHRINK,
                libc::F_SEAL_GROW,
                libc::F_SEAL_SEAL,
            ] {
                assert_ne!(seals & seal, 0, "missing seal {seal:#x}");
            }
            let allow_everything = [0u8; 8];
            let n = unsafe {
                libc::pwrite(
                    handle.raw(),
                    allow_everything.as_ptr() as *const libc::c_void,
                    allow_everything.len(),
                    0,
                )
            };
            assert_eq!(n, -1, "the sealed program accepted a write");
            assert_eq!(unsafe { *libc::__errno_location() }, libc::EPERM);
        }

        #[test]
        fn every_denied_name_has_a_syscall_number_on_this_target() {
            let named: BTreeSet<&str> = denied_syscalls().iter().map(|(n, _)| *n).collect();
            let all: BTreeSet<&str> = super::super::DENIED_SYSCALL_NAMES.iter().copied().collect();
            assert_eq!(
                named, all,
                "a name libc does not define here is skipped, and that hole should be seen"
            );
        }
    }
}
