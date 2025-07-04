ci:
    just clippy
    just fmt

clippy:
    cargo clippy --all-targets --all-features

fmt:
    cargo fmt
