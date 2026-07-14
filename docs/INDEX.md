# Documentation Index

This directory contains the project handoff documentation for the CLIP classifier service.

## Start Here

- `../README.md`: Quick setup, local run commands, and API examples.
- `../AGENTS.md`: Agent and maintainer instructions.

## Reference Docs

- `CODE_INDEX.md`: File-level source, test, and script map.
- `ARCHITECTURE.md`: Service architecture, request flow, GPU execution model, and boundaries.
- `OPERATIONS.md`: Environment setup, configuration, deployment, verification, and production notes.
- `API_EXAMPLES.md`: curl, Python, and Go client examples for authenticated encrypted API calls.

## Common Workflows

- For code changes: read `AGENTS.md`, then `CODE_INDEX.md`, then run tests.
- For API changes: update `README.md`, `ARCHITECTURE.md`, tests, and this index if new docs are added.
- For deployment changes: update `OPERATIONS.md`, `.env.example`, and service scripts.
- For model or label changes: update `config/labels.yaml`, `OPERATIONS.md`, and any relevant tests.

## Current Service Surface

- `GET /healthz`
- `GET /readyz`
- `GET /metrics`
- `GET /v1/encryption/public-key`
- `POST /v1/classify`
- `POST /v1/classify/encrypted`
- `POST /v1/classify/batch`
- `POST /v1/classify/batch/encrypted`

When `CLIP_AUTH_ENABLED=true`, every route requires `Authorization: Bearer <token>`.
