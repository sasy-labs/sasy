use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    suite: PathBuf,
    #[arg(long = "suite-sha256")]
    suite_sha256: String,
    #[arg(long = "probe-functors")]
    probe_functors: PathBuf,
    #[arg(long = "probe-functors-sha256")]
    probe_functors_sha256: String,
    /// Closed, versioned manifest of every runtime asset the assured path may
    /// use. Nothing outside it is discovered.
    #[arg(long = "input-manifest")]
    input_manifest: PathBuf,
    #[arg(long = "input-manifest-sha256")]
    input_manifest_sha256: String,
    #[arg(long = "work-root")]
    work_root: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

#[cfg(feature = "compiler")]
#[tokio::main]
async fn main() -> ExitCode {
    let args = Args::parse();
    let report = match sasy_policy::runtime_conformance::execute_souffle_report(
        &args.suite,
        &args.suite_sha256,
        &args.probe_functors,
        &args.probe_functors_sha256,
        &args.input_manifest,
        &args.input_manifest_sha256,
        &args.work_root,
    )
    .await
    {
        Ok(report) => report,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(error) =
        sasy_policy::runtime_conformance::write_execution_report_create_new(&report, &args.output)
    {
        eprintln!("{error}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

#[cfg(not(feature = "compiler"))]
fn main() -> ExitCode {
    let _ = Args::parse();
    eprintln!("runtime conformance requires the compiler feature");
    ExitCode::FAILURE
}
