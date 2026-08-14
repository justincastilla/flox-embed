# flox-embed

A reproducible `POST /embed` endpoint backed by a real sentence-embedding
model, defined entirely by one Flox manifest. No Dockerfile, no `flake.nix`, no
`requirements.txt`, no virtualenv.

Flox provides the whole stack — the Python interpreter, the Python packages, and
the native libraries underneath them — pinned by hash in a single lockfile.
Cloning this repo is the install step.

Works on `aarch64-darwin`, `aarch64-linux`, and `x86_64-linux`.

## Run it

```bash
flox activate -s
./demo.sh
```

`-s` starts the API declared in the manifest. The first run downloads the model
once and caches it inside the project.

```
$ curl -s localhost:8000/healthz | jq
{
  "status": "ok",
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "device": "cpu",
  "dimensions": 384
}
```

```
$ curl -s localhost:8000/embed \
    -H 'content-type: application/json' \
    -d '{"input":["the ferry from Southworth"]}' | jq '.dimensions'
384
```

## What's here

| File | What it is |
|---|---|
| `.flox/env/manifest.toml` | The whole environment in one file. Read this one first. |
| `.flox/env/manifest.lock` | Every package pinned by hash, resolved for all three systems. |
| `app.py` | FastAPI + sentence-transformers. Two endpoints, under 50 lines. |
| `demo.sh` | A guided tour of the running environment. |

There is no second dependency file.

## What `demo.sh` shows

| Part | Shows |
|---|---|
| 1 | `GET /healthz` — service already up, model already resident. |
| 1a | `POST /embed` with three input strings, then dimensions, the first few floats, and pairwise cosine similarity. |
| 2 | `flox list` next to what actually imported — one lockfile, one dependency system. |
| 3 | `flox services status` — supervision from eight words of TOML. |

The sample input is chosen so the numbers mean something: *ferry from
Southworth* ↔ *boat to Seattle* scores **0.403**, against **0.086** and
**0.042** for *tire rotation*.

## Live reload

```bash
EMBED_RELOAD=1 flox activate -s
```

The service picks up edits to `app.py` without a restart.

## Use an accelerator

Inference runs on CPU by default, which keeps results identical on every
machine. To use Apple Metal or a GPU:

```bash
EMBED_DEVICE=mps flox activate -s     # or cuda
```

## Add a vector store

`[install]` and `[services.qdrant]` both have a commented Qdrant block in the
manifest. Uncomment both, then `flox edit` to relock — it resolves to
`qdrant@1.18.2`. `flox activate -s` will then start both services.

## Containerize it

```bash
flox containerize --runtime docker --tag local
docker run -p 8000:8000 -v "$PWD":/app -w /app floxembed:local \
  uvicorn app:app --host 0.0.0.0 --port 8000
```

The image carries the environment; mount the project so the app comes with it.

## CI

`demo.sh` runs end to end on all three systems on every push — see
[`.github/workflows/demo.yml`](.github/workflows/demo.yml).
