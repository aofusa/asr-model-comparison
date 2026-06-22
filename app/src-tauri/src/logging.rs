use tracing_subscriber::EnvFilter;

const DEFAULT_LOG_FILTER: &str =
    "amcp_tauri=info,amcp_server=info,amcp_desktop=info,tower_http=info";

pub fn init() {
    init_cli();
}

pub fn init_cli() {
    attach_parent_console();
    init_subscriber();
}

pub fn init_desktop() {
    detach_console();
}

fn init_subscriber() {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(DEFAULT_LOG_FILTER));

    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .try_init();
}

#[cfg(target_os = "windows")]
fn detach_console() {
    #[link(name = "kernel32")]
    extern "system" {
        fn FreeConsole() -> i32;
    }

    unsafe {
        let _ = FreeConsole();
    }
}

#[cfg(not(target_os = "windows"))]
fn detach_console() {}

#[cfg(target_os = "windows")]
fn attach_parent_console() {
    const ATTACH_PARENT_PROCESS: u32 = u32::MAX;
    const GENERIC_WRITE: u32 = 0x4000_0000;
    const FILE_SHARE_WRITE: u32 = 0x0000_0002;
    const OPEN_EXISTING: u32 = 3;
    const STD_OUTPUT_HANDLE: u32 = u32::MAX - 10;
    const STD_ERROR_HANDLE: u32 = u32::MAX - 11;
    const INVALID_HANDLE_VALUE: isize = -1;
    type Handle = *mut std::ffi::c_void;

    #[link(name = "kernel32")]
    extern "system" {
        fn AttachConsole(dwProcessId: u32) -> i32;
        fn CreateFileW(
            lpFileName: *const u16,
            dwDesiredAccess: u32,
            dwShareMode: u32,
            lpSecurityAttributes: *mut std::ffi::c_void,
            dwCreationDisposition: u32,
            dwFlagsAndAttributes: u32,
            hTemplateFile: Handle,
        ) -> Handle;
        fn SetStdHandle(nStdHandle: u32, hHandle: Handle) -> i32;
    }

    unsafe {
        if AttachConsole(ATTACH_PARENT_PROCESS) == 0 {
            return;
        }

        // A Windows-subsystem Tauri binary has no stdio by default. Rebind the
        // process handles so tracing can write to the launching console.
        let conout = "CONOUT$\0".encode_utf16().collect::<Vec<_>>();
        let stdout = CreateFileW(
            conout.as_ptr(),
            GENERIC_WRITE,
            FILE_SHARE_WRITE,
            std::ptr::null_mut(),
            OPEN_EXISTING,
            0,
            std::ptr::null_mut(),
        );
        if stdout as isize != INVALID_HANDLE_VALUE {
            let _ = SetStdHandle(STD_OUTPUT_HANDLE, stdout);
        }

        let stderr = CreateFileW(
            conout.as_ptr(),
            GENERIC_WRITE,
            FILE_SHARE_WRITE,
            std::ptr::null_mut(),
            OPEN_EXISTING,
            0,
            std::ptr::null_mut(),
        );
        if stderr as isize != INVALID_HANDLE_VALUE {
            let _ = SetStdHandle(STD_ERROR_HANDLE, stderr);
        }
    }
}

#[cfg(not(target_os = "windows"))]
fn attach_parent_console() {}
