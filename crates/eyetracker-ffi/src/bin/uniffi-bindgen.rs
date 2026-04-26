// uniffi-bindgen.rs — Generate Swift and Kotlin bindings
// Invoked as: cargo run --bin uniffi-bindgen --features cli -- generate --language swift --out-dir bindings/swift

#[cfg(feature = "cli")]
fn main() {
    uniffi::uniffi_bindgen_main()
}

#[cfg(not(feature = "cli"))]
fn main() {
    eprintln!("Error: uniffi-bindgen requires the 'cli' feature.");
    eprintln!("Run with: cargo run --bin uniffi-bindgen --features cli -- <args>");
    std::process::exit(1);
}
