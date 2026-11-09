# Tokspur

Tokspur runs Qwen3.8 Flash-Next locally on a Mac. It uses a native Metal worker and serves an OpenAI-compatible chat API.

Requires an M5 Max with a 40-core GPU, 128 GB of unified memory and macOS 27 or later. Follow the [setup guide](https://tokspur.com/get-started/) to install and connect a client.

## Add the model

Use the pinned target shards and shared Q8 sidecar from the [model guide](https://tokspur.com/docs/models/). Adoption reuses the original files. Check available space before building the local pack.

```sh
tokspur adopt --target /path/to/UD-Q4_K_XL --draft /path/to/snapshot
tokspur configure
tokspur pack plan --check
tokspur pack build
tokspur serve
```

Connect to `http://127.0.0.1:18081/v1` with model `tokspur` and the generated API key.

## Build

Install Xcode with the Metal toolchain, Git, Python 3.12 and Rust 1.99.1. Run these commands from the checkout root.

```sh
python3.12 scripts/bootstrap.py
.venv/bin/tokspur doctor
```

## Licences

[Apache 2.0](LICENSE). See [NOTICE](NOTICE) for third-party credits and [model licences](MODEL_LICENCES.md) for the weights. Report vulnerabilities through the [Security page](https://github.com/san-digital/tokspur/security).
