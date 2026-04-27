# Security Policy

## Reporting a Vulnerability

Do not open public issues for security-sensitive reports.

Send a private disclosure to the repository owner or use GitHub private
vulnerability reporting if it is enabled for this repository.

Include:

- A concise description of the issue.
- Affected crates, generated bindings, or workflow files.
- Reproduction steps or proof of concept, if safe to share.
- Suggested mitigation, if known.

## Scope

This repository contains Rust eye-tracking crates and generated UniFFI bindings.
Security reports should focus on unsafe FFI behavior, memory-safety defects,
dependency vulnerabilities, workflow behavior, and documentation that could
expose secrets.
