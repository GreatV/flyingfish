//! Application composition: the CLI's workflows over the runtime and model
//! crates. Runtime and model implementations live behind real Cargo boundaries.

/// The three dependency layers, each reachable by exactly one path.
///
/// `runtime` is the model-independent core; model crates are the adapters.
/// Their modules are deliberately not re-exported flat here: an application
/// module that reaches into an adapter should say so at the use site.
pub use ff_core as runtime;
pub(crate) use ff_core::required_option;
pub use ff_dsv41 as dsv41;
pub use ff_edge0 as edge0;
pub use ff_glm as glm;
pub use ff_h3 as h3;
pub use ff_minicpm as minicpm;
pub use ff_music as music;
pub use ff_qwen35 as qwen35;
pub use ff_trellis as trellis;

pub mod calibration;
pub mod checkpoint_layout;
pub mod collective_benchmark;
pub mod durable_fs;
pub mod host_profile;
pub mod interconnect_benchmark;
pub mod models;
pub mod recovery;
pub mod resource_policy;

/// Bind the application's identity rather than the collecting dependency's.
pub fn collect_binary_identity(
    path: &std::path::Path,
) -> anyhow::Result<runtime::identity::BinaryIdentity> {
    let mut features = Vec::new();
    if cfg!(feature = "cuda") {
        features.push("cuda");
    }
    if cfg!(feature = "flash-attn") {
        features.push("flash-attn");
    }
    if cfg!(feature = "metal") {
        features.push("metal");
    }
    runtime::identity::BinaryIdentity::collect(
        path,
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        &features,
    )
}
