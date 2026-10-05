//! Standalone analyzer CLI: reads a desugared `.dl`
//! source from stdin or a file argument, runs the four
//! static analyses, prints findings as text or JSON.
//!
//! Usage:
//!   policy-analyze              (reads desugared from stdin)
//!   policy-analyze path/to.dl   (reads from file)
//!   policy-analyze --json …
//!
//! The desugared input is expected to have already gone
//! through `sugar.py` (the production preprocessor) and
//! had `common_policy.dl` prepended; the Makefile target
//! handles that. When run on a raw user policy, results
//! will be incomplete or wrong.

use std::io::Read;
use std::process::ExitCode;

use sasy_policy::analysis::analyze_desugared;

fn main() -> ExitCode {
    let args = std::env::args().skip(1).peekable();
    let mut json = false;
    let mut path: Option<String> = None;

    for arg in args {
        match arg.as_str() {
            "--json" => json = true,
            "--help" | "-h" => {
                eprintln!("usage: policy-analyze [--json] [PATH]");
                eprintln!("  --json     emit a JSON-formatted report instead of text");
                eprintln!("  PATH       path to a desugared .dl file (default: stdin)");
                return ExitCode::SUCCESS;
            }
            other if path.is_none() => path = Some(other.to_string()),
            other => {
                eprintln!("unknown argument: {}", other);
                return ExitCode::from(2);
            }
        }
    }

    let (source, file_label) = match &path {
        Some(p) => match std::fs::read_to_string(p) {
            Ok(s) => (s, p.clone()),
            Err(e) => {
                eprintln!("read {}: {}", p, e);
                return ExitCode::FAILURE;
            }
        },
        None => {
            let mut buf = String::new();
            if let Err(e) = std::io::stdin().read_to_string(&mut buf) {
                eprintln!("read stdin: {}", e);
                return ExitCode::FAILURE;
            }
            (buf, "<stdin>".to_string())
        }
    };

    let report = match analyze_desugared(&source, &file_label, &[]) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("parse: {}", e);
            return ExitCode::FAILURE;
        }
    };

    if json {
        match serde_json::to_string_pretty(&report) {
            Ok(s) => println!("{}", s),
            Err(e) => {
                eprintln!("serialize: {}", e);
                return ExitCode::FAILURE;
            }
        }
    } else {
        print_text_report(&report);
    }
    ExitCode::SUCCESS
}

fn print_text_report(r: &sasy_policy::analysis::AnalysisReport) {
    println!("=== Static analysis report ===\n");

    if !r.broad_rules.is_empty() {
        println!("Broad rules ({}):", r.broad_rules.len());
        for n in &r.broad_rules {
            println!("  • {}", n);
        }
        println!();
    }

    println!("Contradictions ({}):", r.contradictions.len());
    for f in &r.contradictions {
        println!(
            "  [{}] allow@{}  vs  deny@{}",
            f.category, f.allow_location, f.deny_location
        );
        if !f.message.is_empty() {
            println!("      {}", f.message);
        }
        if !f.allow_body.is_empty() {
            println!("      allow: {}", f.allow_body);
        }
        if !f.deny_body.is_empty() {
            println!("      deny:  {}", f.deny_body);
        }
    }
    println!();

    println!("Redundancies ({}):", r.redundancies.len());
    for f in &r.redundancies {
        println!(
            "  [{}] {}  subsumed by  {}",
            f.head_relation, f.redundant_location, f.covered_by_location
        );
        if !f.redundant_body.is_empty() {
            println!("      redundant:   {}", f.redundant_body);
        }
        if !f.covered_by_body.is_empty() {
            println!("      covered by:  {}", f.covered_by_body);
        }
    }
    println!();

    println!("Reachability:");
    for f in &r.reachability {
        println!(
            "  {}: {} disjunct(s), {} pruned, opaque {{{}}}",
            f.target,
            f.disjuncts.len(),
            f.pruned,
            f.opaque.join(", ")
        );
        for (i, d) in f.disjuncts.iter().enumerate() {
            println!("    [{}] {}", i + 1, d);
        }
    }
}
