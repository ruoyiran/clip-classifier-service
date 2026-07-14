from dataclasses import dataclass
from pathlib import Path

import yaml


class LabelConfigError(ValueError):
    """Raised when the label configuration is invalid."""


@dataclass(frozen=True)
class Label:
    id: str
    name: str
    prompt: str
    enabled: bool = True


def load_labels(path: Path) -> list[Label]:
    if not path.exists():
        raise LabelConfigError(f"labels file does not exist: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = raw.get("labels")
    if not isinstance(entries, list):
        raise LabelConfigError("labels file must contain a 'labels' list")

    labels: list[Label] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise LabelConfigError(f"label at index {index} must be an object")

        label_id = str(entry.get("id", "")).strip()
        name = str(entry.get("name", "")).strip()
        prompt = str(entry.get("prompt", "")).strip()
        enabled = bool(entry.get("enabled", True))

        if not label_id or not name or not prompt:
            raise LabelConfigError(f"label at index {index} must define id, name, and prompt")
        if label_id in seen_ids:
            raise LabelConfigError(f"duplicate label id: {label_id}")

        seen_ids.add(label_id)
        if enabled:
            labels.append(Label(id=label_id, name=name, prompt=prompt, enabled=True))

    if not labels:
        raise LabelConfigError("labels file must contain at least one enabled label")

    return labels
