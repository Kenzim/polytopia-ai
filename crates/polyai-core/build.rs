//! Stores the commit of the engine checkout in `ENGINE_COMMIT` so game records say what produced them.
use std::process::Command;

fn main() {
    let engine = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../engine");
    let git = |args: &[&str]| -> Option<String> {
        let out = Command::new("git").arg("-C").arg(&engine).args(args).output().ok()?;
        out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
    };
    let commit = git(&["rev-parse", "--short=12", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let dirty = git(&["status", "--porcelain", "--untracked-files=no", "--", "src", "data", "Cargo.toml"]).is_some_and(|s| !s.is_empty());
    println!("cargo:rustc-env=ENGINE_COMMIT={}{}", commit, if dirty { "-dirty" } else { "" });
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", engine.join(".git/HEAD").display());
}
