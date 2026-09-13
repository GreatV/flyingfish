mod cli;

/// The stack the command tree's parser needs.
///
/// Windows gives a process's main thread 1 MiB by default, where Linux gives
/// 8 MiB. This binary's clap tree is large enough that parsing it overflows
/// the smaller of the two before any subcommand runs, so the work moves to a
/// thread that asks for the room outright rather than depending on whichever
/// default the platform happens to provide.
const MAIN_STACK_BYTES: usize = 8 * 1024 * 1024;

fn main() -> anyhow::Result<()> {
    std::thread::Builder::new()
        .name("ff-main".to_owned())
        .stack_size(MAIN_STACK_BYTES)
        .spawn(cli::run)
        .map_err(|error| anyhow::anyhow!("failed to start the main worker thread: {error}"))?
        .join()
        .map_err(|_| anyhow::anyhow!("the main worker thread panicked"))?
}
