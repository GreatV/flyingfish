//! CPU-only admission receipt for external harnesses; uses the Engine's pure policy.
use anyhow::{Context, Result, ensure};
use flyingfish::{
    gdn_plan::{GdnPolicy, GdnWorkspace},
    op_contract::GdnSpec,
};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 3,
        "usage: q38_gdn_budget <sm> <SM-count> <physical-rows>"
    );
    let arch = args[0]
        .parse::<u32>()
        .context("invalid CUDA architecture")?;
    let sms = args[1].parse::<usize>().context("invalid SM count")?;
    let rows = args[2].parse::<u32>().context("invalid row capacity")?;
    let policy = GdnPolicy::for_device(arch, sms);
    let spec = GdnSpec::q38();
    let legacy = GdnWorkspace::q38(&spec, rows)?;
    let workspace = policy.workspace(&spec, rows)?;
    let buckets = policy
        .buckets(rows)
        .into_iter()
        .map(|bound| {
            let req = policy.workspace(&spec, bound)?;
            Ok(serde_json::json!({"bound_rows":bound,"workspace_bytes":req.bytes}))
        })
        .collect::<Result<Vec<_>>>()?;
    println!(
        "{}",
        serde_json::json!({
            "gdn_plan":policy.identity(),"cuda_sm":arch,"sms":sms,"physical_rows":rows,
            "legacy_workspace_bytes":legacy.bytes,"workspace_bytes":workspace.bytes,
            "workspace_alignment":workspace.alignment,"scan_buckets":buckets,
            "accounting":"maximum complete plan; one arena reused across layers",
        })
    );
    Ok(())
}
