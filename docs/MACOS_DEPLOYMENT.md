# Testing the macOS deployment

Commands for verifying the service on `bytedance@bytedance`, deployed at
`~/deploy/clip-classifier-service` under launchd.

## Two things that will surprise you

1. **Every endpoint needs the token, including health checks.** `/healthz`,
   `/readyz`, and `/metrics` are registered on the same router that carries the
   auth dependency (`api.py:126`), so with `CLIP_AUTH_ENABLED=true` an
   unauthenticated `/healthz` returns 401, not 200. Any external uptime monitor
   or load balancer probe must send the header too.
2. **Every path is prefixed with `/openai-clip`.** `CLIP_BASE_PATH=/openai-clip`,
   so bare `/healthz` returns **404**. Verified.

## Setup

```bash
ssh bytedance@bytedance
cd ~/deploy/clip-classifier-service
export TOKEN=$(grep '^CLIP_API_TOKENS=' .env | cut -d= -f2)
export BASE=http://127.0.0.1:8000/openai-clip
```

## Is the agent running?

```bash
launchctl print gui/501/com.clip-classifier-service | grep -E "state|pid|last exit"
tail -30 logs/launchd.err.log
```

`state = running` with a live `pid`. A nonzero `last exit code` with a pid that
keeps changing means it is crash-looping; `ThrottleInterval` holds that to one
restart per minute.

## Health and readiness

```bash
curl -s -o /dev/null -w "healthz: %{http_code}\n" -H "Authorization: Bearer $TOKEN" $BASE/healthz
curl -s -o /dev/null -w "readyz:  %{http_code}\n" -H "Authorization: Bearer $TOKEN" $BASE/readyz
```

`readyz` is the meaningful one. It returns **503** until the weights finish
loading, so it is the real "model is live" signal; `healthz` returns 200 while
the model is still loading. After a restart, expect `readyz` to take a while
before it turns 200.

## Auth is actually enforced

```bash
curl -s -o /dev/null -w "no token:  %{http_code}\n" $BASE/healthz
curl -s -o /dev/null -w "bad token: %{http_code}\n" -H "Authorization: Bearer wrong" $BASE/healthz
curl -s -o /dev/null -w "no prefix: %{http_code}\n" -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/healthz
```

Expect `401`, `401`, `404`. If either of the first two returns 200, auth failed
to load and the service is exposed on the LAN.

## Classification

```bash
curl -s -H "Authorization: Bearer $TOKEN" -F "file=@/tmp/landscape.jpg" \
  "$BASE/v1/classify?top_k=5"
```

Use a real photo for a meaningful result. The 69 labels are photo categories, so
a synthetic or flat image produces low, arbitrary scores even when the pipeline
is completely healthy.

Each returned label carries both `score` (softmax across the 69 labels) and
`similarity` (raw cosine). The `threshold` parameter filters on `score`, not
`similarity`.

## Latency

```bash
for i in 1 2 3 4; do
  curl -s -o /dev/null -w "run $i: %{time_total}s\n" \
    -H "Authorization: Bearer $TOKEN" -F "file=@/tmp/landscape.jpg" "$BASE/v1/classify?top_k=5"
done
```

Discard run 1, which pays for warm-up. This is fp32 on an M1 Pro rather than
fp16 on a 4090, so expect a significant regression against the Windows host.

## Encryption

```bash
curl -s -H "Authorization: Bearer $TOKEN" $BASE/v1/encryption/public-key
.venv/bin/python scripts/encrypt_sample_image.py /tmp/landscape.jpg --base-url $BASE --token "$TOKEN"
```

`--base-url` is mandatory: the script defaults to
`http://127.0.0.1:8080/openai-clip`, which is the wrong port.

## Batch

```bash
curl -s -H "Authorization: Bearer $TOKEN" \
  -F "files=@/tmp/landscape.jpg" -F "files=@/tmp/landscape.jpg" \
  "$BASE/v1/classify/batch?top_k=3"
```

`CLIP_MAX_BATCH_SIZE=8`; a ninth image in one request is rejected.

## From another machine on the LAN

The service binds `0.0.0.0`, so it is reachable across the network with the
token as the only gate.

```bash
ssh bytedance@bytedance 'ipconfig getifaddr en0'
curl -s -o /dev/null -w "%{http_code}\n" -H "Authorization: Bearer <token>" \
  http://<that-ip>:8000/openai-clip/healthz
```

Use the IP, not the hostname `bytedance` — that is an SSH alias and does not
necessarily resolve elsewhere.

## test_external_api.py

This script requires `--internal-api-key` and exits with code 2 without one. That
header is for an upstream gateway, not this service, so pass a dummy value when
testing directly:

```bash
.venv/bin/python test_external_api.py /tmp/landscape.jpg --base-url $BASE \
  --internal-api-key dummy --bearer-token "$TOKEN" --top-k 3
```

## Service control

```bash
launchctl kickstart -k gui/501/com.clip-classifier-service   # restart
launchctl bootout gui/501/com.clip-classifier-service        # stop and unload
launchctl bootstrap gui/501 ~/Library/LaunchAgents/com.clip-classifier-service.plist
```

A restart reloads roughly 4 GB of weights, so `readyz` will report 503 for a
while afterwards. Weights are cached in `models/` and are not re-downloaded.

## Updating the deployment

```bash
cd ~/deploy/clip-classifier-service
git pull origin feat/macos-launchd-deployment
export PATH="$HOME/.local/bin:$PATH"
uv pip install -r requirements-macos.txt        # only if dependencies changed
launchctl kickstart -k gui/501/com.clip-classifier-service
```

After editing `config/labels.yaml`, run `.venv/bin/python scripts/preload_model.py`
before restarting so text features are rebuilt outside the request path.
