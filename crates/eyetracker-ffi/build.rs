// build.rs — Generate UniFFI bindings
// Traces to: FR-EYE-INTEROP-001

fn main() {
    // Note: With uniffi 0.27, UDL compilation is handled via uniffi::include_scaffolding!()
    // For now, just rebuild on UDL changes
    println!("cargo:rerun-if-changed=src/eyetracker.udl");
}
