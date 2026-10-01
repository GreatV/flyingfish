use anyhow::Result;
use std::{net::SocketAddr, num::NonZeroUsize, path::PathBuf};

#[cfg(feature = "cuda")]
mod engine;
#[cfg(any(feature = "cuda", test))]
mod http;

#[derive(Debug, clap::Args)]
pub(in crate::cli) struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    host_profile: Option<PathBuf>,
    #[arg(long, default_value = "cuda:0")]
    device: String,
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: SocketAddr,
    #[arg(long, default_value_t = NonZeroUsize::new(4096).unwrap())]
    max_context: NonZeroUsize,
}

#[cfg(not(feature = "cuda"))]
pub(super) fn run(_args: Args) -> Result<()> {
    anyhow::bail!("Qwen serving requires a binary built with --features cuda")
}

#[cfg(feature = "cuda")]
pub(super) fn run(args: Args) -> Result<()> {
    use super::super::text_runtime::{TextDevice, resolve_text_device};
    use anyhow::Context;
    let (TextDevice::Cuda(devices), _) = resolve_text_device(&args.device)? else {
        anyhow::bail!("Qwen serving requires CUDA");
    };
    let address = args.listen;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind(address)
            .await
            .context("bind HTTP listener")?;
        let state = http::Worker::start(move || engine::Engine::load(args, devices)).await?;
        eprintln!("Qwen HTTP ready: http://{}", listener.local_addr()?);
        axum::serve(listener, http::router(state))
            .await
            .context("serve HTTP")
    })
}
