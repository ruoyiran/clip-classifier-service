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
    prompts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        prompts = self.prompts or (self.prompt,)
        object.__setattr__(self, "prompts", tuple(prompts))
        object.__setattr__(self, "prompt", prompts[0])


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
        prompts = _parse_prompts(entry, index)
        enabled = bool(entry.get("enabled", True))

        if not label_id or not name or not prompts:
            raise LabelConfigError(
                f"label at index {index} must define id, name, and prompt or prompts"
            )
        if label_id in seen_ids:
            raise LabelConfigError(f"duplicate label id: {label_id}")

        seen_ids.add(label_id)
        if enabled:
            labels.append(
                Label(
                    id=label_id,
                    name=name,
                    prompt=prompt or prompts[0],
                    prompts=tuple(prompts),
                    enabled=True,
                )
            )

    if not labels:
        raise LabelConfigError("labels file must contain at least one enabled label")

    return labels


def _parse_prompts(entry: dict, index: int) -> list[str]:
    raw_prompts = entry.get("prompts")
    if raw_prompts is not None:
        if not isinstance(raw_prompts, list):
            raise LabelConfigError(f"label at index {index} prompts must be a list")
        prompts = [
            _parse_prompt(prompt, index, prompt_index)
            for prompt_index, prompt in enumerate(raw_prompts)
        ]
    else:
        prompts = [_parse_prompt(entry.get("prompt", ""), index, None)]
    return prompts


def _parse_prompt(value, label_index: int, prompt_index: int | None) -> str:
    if not isinstance(value, str):
        location = f"prompt {prompt_index}" if prompt_index is not None else "prompt"
        raise LabelConfigError(f"label at index {label_index} {location} must be a string")
    prompt = value.strip()
    if not prompt:
        location = f"prompt {prompt_index}" if prompt_index is not None else "prompt"
        raise LabelConfigError(f"label at index {label_index} {location} must not be empty")
    return prompt
