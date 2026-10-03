# llamacpp-ini

llama-server router presets and a sync script for a nested LM Studio style model tree.

`llama-server --models-dir` doesn't recurse, so LM Studio's `<publisher>/<repo>/<file>.gguf` layout is invisible to it. `sync-models.py` symlinks the tree flat, probes each GGUF with `llama-gguf`, and writes `.generated/router.ini` with per-model speculative decoding on top of the host's base preset.

Needs `llama-server` and `llama-gguf` on PATH, plus Python 3.

## Usage

```sh
./run-llama-server.sh [extra llama-server args]
```

Syncs, then starts the router on `127.0.0.1`. `SKIP_SYNC=1` skips the sync; `./sync-models.py` runs it on its own.

`./run-llama-server.sh --bonsai` runs Ternary-Bonsai-2-27B on the PrismML fork (`~/git/llama.cpp-prismml`, override with `PRISMML_SERVER_BIN`) at `127.0.0.1:8081` (`BONSAI_PORT`), beside the router. Its PQ2_0 / PTQ1_0 quants don't load in upstream llama.cpp, so sync keeps them out of the router.

Diffusion models (dream, llada, diffusion-gemma) can't load in llama-server. They're excluded from the router and listed in `.generated/diffusion-models.tsv` for use with `llama-diffusion-cli`.

## Per-host config

The base preset is picked by `hostname -s`: `<hostname>.ini`, falling back to `samm-mbp.ini`. An optional `<hostname>.env` is sourced for sync thresholds. `LLAMA_BASE_INI` overrides both.

Presets here are `samm-mbp.ini` (M5 Max, 128 GB) and `samm-mba.ini` (M1 Air, 16 GB). Keys are llama-server long arguments without the leading dashes.

## Env

`LLAMA_SERVER_BIN`, `LLAMA_GGUF`, `LLAMA_BASE_INI`, `MODELS_SRC` (default `~/.lmstudio/models`), `MODELS_DIR` (default `./models`), `SKIP_SYNC`.

`models/` and `.generated/` are generated per machine and gitignored.

MIT licensed.
