# Tsetlin Machine MNIST classifier

A Rust implementation of https://github.com/BooBSD/Tsetlin.jl

### Quickstart

Install Rust `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

Prepare data: `unzip mnist.zip -d mnist`

Run basic training and evaluation: `cargo r --release`

### Profiling

Install [samply](https://github.com/mstange/samply) `cargo install --locked samply`

Run profiler `samply record -r 5000 cargo r --release`

