fn main() {
    if std::env::var_os("CARGO_FEATURE_DESKTOP").is_some() {
        tauri_build::build();
    }
    if std::env::var_os("CARGO_FEATURE_VOXTRAL_LLAMACPP_REALTIME_PATCHED").is_some() {
        build_voxtral_realtime_bridge();
    }
}

fn build_voxtral_realtime_bridge() {
    use std::env;
    use std::path::PathBuf;

    let source_dir = env_path("AMCP_VOXTRAL_PATCHED_LLAMA_DIR", "patched llama.cpp source root");
    let lib_dir = env_path(
        "AMCP_VOXTRAL_PATCHED_LLAMA_LIB_DIR",
        "patched llama.cpp library directory",
    );
    let bin_dir = env::var("AMCP_VOXTRAL_PATCHED_LLAMA_BIN_DIR")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from);

    let include_dir = source_dir.join("include");
    let mtmd_dir = source_dir.join("tools").join("mtmd");
    let ggml_include_dir = source_dir.join("ggml").join("include");

    for path in [&include_dir, &mtmd_dir, &ggml_include_dir] {
        if !path.is_dir() {
            panic!(
                "voxtral-llamacpp-realtime-patched requires {}, but it does not exist",
                path.display()
            );
        }
    }

    println!("cargo:rerun-if-env-changed=AMCP_VOXTRAL_PATCHED_LLAMA_DIR");
    println!("cargo:rerun-if-env-changed=AMCP_VOXTRAL_PATCHED_LLAMA_LIB_DIR");
    println!("cargo:rerun-if-env-changed=AMCP_VOXTRAL_PATCHED_LLAMA_BIN_DIR");
    println!("cargo:rerun-if-env-changed=AMCP_VOXTRAL_PATCHED_LLAMA_LINK_VULKAN");
    println!("cargo:rerun-if-changed=src/native/voxtral_realtime_bridge.cpp");

    cc::Build::new()
        .cpp(true)
        .file("src/native/voxtral_realtime_bridge.cpp")
        .include(include_dir)
        .include(mtmd_dir)
        .include(ggml_include_dir)
        .flag_if_supported("/std:c++17")
        .flag_if_supported("-std=c++17")
        .compile("amcp_voxtral_realtime_bridge");

    for search_dir in [
        lib_dir.clone(),
        lib_dir.join("src"),
        lib_dir.join("tools").join("mtmd"),
        lib_dir.join("ggml").join("src"),
        lib_dir.join("bin"),
    ] {
        if search_dir.is_dir() {
            println!("cargo:rustc-link-search=native={}", search_dir.display());
        }
    }
    if let Some(bin_dir) = &bin_dir {
        println!("cargo:rustc-link-search=native={}", bin_dir.display());
    }
    for lib in ["mtmd", "llama", "ggml", "ggml-base", "ggml-cpu"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
    let link_vulkan = env::var("AMCP_VOXTRAL_PATCHED_LLAMA_LINK_VULKAN")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or_else(|_| lib_dir.join("ggml-vulkan.lib").exists());
    if link_vulkan {
        println!("cargo:rustc-link-lib=dylib=ggml-vulkan");
    }

    if cfg!(target_os = "windows") {
        if let Some(bin_dir) = bin_dir {
            copy_voxtral_realtime_dlls(&bin_dir, link_vulkan);
        }
    }
}

fn env_path(name: &str, description: &str) -> std::path::PathBuf {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            panic!("voxtral-llamacpp-realtime-patched requires {name} ({description})")
        })
}

fn copy_voxtral_realtime_dlls(bin_dir: &std::path::Path, link_vulkan: bool) {
    let out_dir = env_path("OUT_DIR", "Cargo build output directory");
    let profile_dir = out_dir
        .ancestors()
        .nth(3)
        .unwrap_or_else(|| panic!("failed to resolve Cargo profile dir from {}", out_dir.display()));

    let mut dlls = vec![
        "llama.dll",
        "mtmd.dll",
        "ggml.dll",
        "ggml-base.dll",
        "ggml-cpu.dll",
    ];
    if link_vulkan {
        dlls.push("ggml-vulkan.dll");
    }

    for dll in dlls {
        let source = bin_dir.join(dll);
        if !source.is_file() {
            panic!(
                "voxtral-llamacpp-realtime-patched requires {}, but it does not exist",
                source.display()
            );
        }
        let destination = profile_dir.join(dll);
        std::fs::copy(&source, &destination).unwrap_or_else(|error| {
            panic!(
                "failed to copy {} to {}: {error}",
                source.display(),
                destination.display()
            )
        });
        println!("cargo:rerun-if-changed={}", source.display());
    }
}
