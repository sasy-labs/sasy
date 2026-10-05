//! OS-level sandbox for evaluator child processes.
//!
//! All sandboxing is applied by the C++ evaluator_shim after the binary
//! loads and the Soufflé program is initialized. The shim applies:
//!
//! 1. `unshare(CLONE_NEWNET)` — network namespace isolation (no sockets)
//! 2. `chroot` to empty directory (best-effort, needs CAP_SYS_CHROOT)
//! 3. seccomp BPF filter — syscall allow-list (stdio + memory + threading)
//!
//! These must run AFTER exec (shared library loading needs openat).
//! The pre_exec hook is NOT used — unshare(CLONE_NEWNET) in pre_exec
//! breaks IPC pipe bootstrap on some kernel/container configurations.
//!
//! On non-Linux platforms (macOS), sandboxing is a no-op in the shim
//! (all code is behind `#ifdef __linux__`).
