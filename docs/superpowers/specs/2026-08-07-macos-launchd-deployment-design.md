# macOS / Apple Silicon Deployment Design

Date: 2026-08-07
Target: `bytedance@bytedance:~/deploy/clip-classifier-service`, managed by launchd.

## Problem

The service is written for a Windows host with an NVIDIA RTX 4090 and is installed
as a Windows service through NSSM. The deployment target is a different machine and
a different platform. Verified facts about `bytedance@bytedance`:

| Property | Value |
| --- | --- |
| OS | macOS 26.6, build 25G72, Darwin 25.6.0 |
| Chip | Apple M1 Pro, arm64 |
| RAM | 32 GB unified |
| Disk free | 382 GB on `/` |
| Python | `/usr/bin/python3` 3.9.6 only. No brew, no uv, no pyenv |
| `sudo` | requires a password, which is not available |
| Xcode CLT | present at `/Applications/Xcode.app/Contents/Developer` |
| launchd | `gui/501` login session exists |
| Port 8000 | free, nothing listening |
| Network | pypi.org, huggingface.co, astral.sh, github.com all reachable |
| GitHub SSH | authenticates as `ruoyiran` |
| Existing clone | `~/workspace/AI/clip-classifier-service` at `782d200`, clean, no `.venv`, no `models/` |

Four hard blockers follow from this:

1. **No CUDA.** `requirements.txt` pins `torch==2.2.2+cu121` and
   `torchvision==0.17.2+cu121`. No such wheel exists for macOS arm64 on any
   torch version.
2. **Startup would crash.** `openclip_backend.py:73` raises
   `RuntimeError("CUDA was requested but is not available")` whenever
   `CLIP_DEVICE=cuda`, which is the default in both `settings.py` and
   `.env.example`.
3. **Wrong Python.** `pyproject.toml` requires `>=3.12,<3.13`; the box has 3.9.6
   and no way to install a newer one via the usual package managers without sudo.
4. **Wrong service manager.** `scripts/install_windows_service.ps1` targets NSSM.
   launchd needs a plist.

This is therefore a port, not a file copy.

## Decisions

Settled with the user before design:

- **Dependencies:** add a separate `requirements-macos.txt`. Leave
  `requirements.txt` untouched so the existing Windows/CUDA production host is
  unaffected.
- **Device:** `mps`, with automatic fallback to CPU.
- **Security:** enable both Bearer token authentication and the encrypted upload
  endpoints.
- **Deploy directory:** local-clone the existing remote `~/workspace/AI/...`
  checkout into `~/deploy/clip-classifier-service`.
- **Change delivery:** commit locally, push to `origin/master`, `git pull` on the
  remote.

## Architecture

Five stages. Stage 1 is local; stages 2 through 5 run on the remote over SSH.

### Stage 1: code changes, committed locally and pushed

All three changes are additive. No existing file changes behaviour on Windows/CUDA.

**a. `requirements-macos.txt` (new)**

Mirrors `requirements.txt` but drops the `--extra-index-url` for the cu121 wheel
index and repins the torch stack to stock PyPI wheels. Verified to exist as
`cp312` / `macosx_11_0_arm64`:

- `torch==2.8.0`
- `torchvision==0.23.0` — its metadata pins `torch==2.8.0` exactly, so the pair is
  internally consistent
- `numpy>=1.26` — the bound is deliberately relaxed from the project's `<2.0`.
  torch 2.8 wheels are compiled against numpy 2.x, and forcing numpy 1.x
  alongside them risks an ABI mismatch at import. Resolved numpy version is
  recorded during install (see Verification).
- `open-clip-torch>=2.24,<3.0` — unchanged. Latest matching release is 2.32.0,
  which declares only `torch>=2.0`, so it does not fight the 2.8.0 pin.

All other pins carry over unchanged from `requirements.txt`.

**b. `src/clip_classifier_service/openclip_backend.py`**

Two layers of fallback, since MPS can fail either at device level or per-operator:

- Module scope: `os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")` so an
  individual unsupported operator silently routes to CPU instead of raising.
- `_resolve_device`: accept `mps`. If `mps` is requested but
  `torch.backends.mps.is_available()` is false, return `torch.device("cpu")`.
  The `cuda` branch keeps raising exactly as before — the Windows host must not
  silently degrade to CPU.
- Model construction: wrap `create_model_and_transforms` so that if the model
  fails to load onto MPS, the backend retries on CPU and records the effective
  device.

The existing dtype logic needs no change: it already selects `float32` for any
non-CUDA device, and passes `precision="fp32"` for the same case.

**c. `scripts/` additions**

- `scripts/start_macos.sh` — resolves the repo root, execs uvicorn from the
  venv with `--workers 1`. launchd runs this as the program.
- `scripts/com.clip-classifier-service.plist` — template, checked in for
  reproducibility. The live copy is installed into `~/Library/LaunchAgents/`.

A wrapper script rather than invoking uvicorn directly from the plist, because
launchd does not expand shell variables and the venv path must be resolved
relative to the deploy root.

### Stage 2: Python 3.12 on the remote

Install `uv` via `https://astral.sh/uv/install.sh`. It lands in `~/.local/bin`,
which already exists, and needs no sudo. Then `uv python install 3.12`, which
fetches a standalone CPython build from GitHub releases (endpoint verified
reachable, returns 302).

This leaves `/usr/bin/python3` 3.9.6 untouched. Nothing system-wide is modified.

### Stage 3: the deploy directory

1. `git -C ~/workspace/AI/clip-classifier-service pull` to collect Stage 1.
2. `git clone ~/workspace/AI/clip-classifier-service ~/deploy/clip-classifier-service`
   — a local clone, so it costs seconds and no network, and retains git history
   so later updates are a `git pull`.
3. Create the venv with `uv venv --python 3.12`, then install
   `requirements-macos.txt` followed by `uv pip install -e . --no-deps`. The
   `--no-deps` matters: without it, the resolver re-reads `pyproject.toml` and
   would drag the cu121 pins back in.

All installs go through `uv pip`, not `python -m pip`: `uv venv` does not place
`pip` inside the venv, so `python -m pip` would not resolve there.

The deploy directory gets its own `.venv/` and `models/`. Experimental edits in
`~/workspace` cannot affect the running service.

### Stage 4: configuration

Written directly on the remote as `~/deploy/clip-classifier-service/.env`.
Never committed — `.gitignore` already covers `.env` and both key files.

- `CLIP_DEVICE=mps`, `CLIP_PRECISION=fp32`
- `CLIP_AUTH_ENABLED=true`, `CLIP_AUTH_DISABLE_DOCS=true`
- `CLIP_API_TOKENS=<32 random bytes, generated on the remote>`
- `CLIP_ENCRYPTION_ENABLED=true`, keys from `scripts/generate_encryption_keys.py`
- `CLIP_HOST=0.0.0.0`, `CLIP_PORT=8000`
- `CLIP_MODEL_CACHE_DIR=models`, `CLIP_LABELS_PATH=config/labels.yaml`
- `CLIP_BASE_PATH=/openai-clip` — matches the existing deployment convention in
  the README and the default in `scripts/encrypt_sample_image.py`. Every route is
  served under this prefix, so health checks live at
  `/openai-clip/healthz`, not `/healthz`.

Then run `scripts/preload_model.py` **before** enabling the service. This
downloads roughly 3.9 GB of weights and encodes the text features for all 69
enabled labels in `config/labels.yaml`, about 700 prompts total. Doing it here
means the first live request does not pay for it, and a download failure surfaces
during deploy rather than as a 503 later.

### Stage 5: launchd

`~/Library/LaunchAgents/com.clip-classifier-service.plist`, loaded into the
`gui/501` domain. A user agent, not a system daemon: a `LaunchDaemon` in
`/Library/LaunchDaemons` would need sudo, which is unavailable.

| Key | Value | Reason |
| --- | --- | --- |
| `Label` | `com.clip-classifier-service` | |
| `ProgramArguments` | `scripts/start_macos.sh` | |
| `WorkingDirectory` | the deploy dir | Required. `Settings` uses `env_file=".env"` and defaults `model_cache_dir`/`labels_path` to relative paths, all resolved against cwd |
| `RunAtLoad` | true | starts on login/boot |
| `KeepAlive` | true | restarts on crash |
| `ThrottleInterval` | 60 | Startup loads about 4 GB of weights. Without this, a persistent startup failure becomes a crash-loop that hammers the machine |
| `StandardOutPath` / `StandardErrorPath` | `logs/launchd.out.log`, `logs/launchd.err.log` | `logs/` is gitignored |

Loaded with `launchctl bootstrap gui/501 <plist>`.

## Verification

Each check must produce real output before the deployment is called done. No
success is claimed from a command that was not run.

1. `uv python list` confirms 3.12 is present.
2. `python -c "import torch, numpy; print(torch.__version__, numpy.__version__, torch.backends.mps.is_available())"`
   inside the venv — confirms the torch/numpy pair imports together without ABI
   error, and records whether MPS is actually available.
3. `python -m pytest` and `python -m ruff check .` in the deploy directory.
   Note that `requirements-dev.txt` cannot be used here: its first line is
   `-r requirements.txt`, which would reintroduce the cu121 pins. The three dev
   tools are installed directly instead — `uv pip install httpx pytest ruff` —
   rather than adding a fourth requirements file for three packages.
4. `scripts/preload_model.py` prints `labels=69` and `ready=true`.
5. `launchctl print gui/501/com.clip-classifier-service` shows the job with a
   live PID.
6. `GET /healthz` returns 200.
7. `GET /readyz` returns 200 — this is the meaningful one, since it fails with 503
   unless the classifier finished loading.
8. One real image through `POST /v1/classify` with the Bearer token, plus
   `GET /v1/encryption/public-key` to confirm encryption is wired up.
9. Measure and report actual single-image latency.
10. Confirm an unauthenticated request is rejected, proving auth is active.

## Known limitations

Both are accepted, not open questions.

1. **Throughput regresses against the 4090.** ViT-H-14-378 is about 1B
   parameters, running fp32 on an M1 Pro. Expect on the order of a second or more
   per image versus the GPU host. Functionally correct, not performance
   comparable. The measured figure is reported after deploy rather than estimated
   here.
2. **The machine is configured to sleep.** `pmset` reports `sleep 1`, currently
   held off by `sharingd` and `powerd`. If the machine ever does sleep the agent
   pauses and the service stops answering. No `caffeinate` workaround is included;
   it is deliberately out of scope until it is observed to matter.

## Out of scope

- TLS. The service listens on plain HTTP; terminate TLS upstream if needed.
- Reverse proxy or gateway configuration.
- Any change to `requirements.txt` or the Windows/NSSM deployment path.
- Log rotation for `logs/`.
