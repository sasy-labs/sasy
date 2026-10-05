fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_dir = "../../proto";

    // credential_server.proto and reference_monitor.proto have no
    // `package` declaration, so prost puts them in the default
    // (empty-string) package. We compile them all together so
    // cross-file imports resolve.
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(
            &[
                format!("{}/observability.proto", proto_dir),
                format!("{}/policy_engine.proto", proto_dir),
                format!("{}/credential_server.proto", proto_dir),
                format!("{}/reference_monitor.proto", proto_dir),
                format!("{}/policy_plugin.proto", proto_dir),
            ],
            &[proto_dir],
        )?;
    Ok(())
}
