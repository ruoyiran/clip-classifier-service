from pathlib import Path

import pytest

from clip_classifier_service.labels import LabelConfigError, load_labels


def test_load_labels_returns_only_enabled_labels(tmp_path: Path) -> None:
    labels_file = tmp_path / "labels.yaml"
    labels_file.write_text(
        """
labels:
  - id: cat
    name: Cat
    prompt: a photo of a cat
    enabled: true
  - id: dog
    name: Dog
    prompt: a photo of a dog
    enabled: false
""",
        encoding="utf-8",
    )

    labels = load_labels(labels_file)

    assert [label.id for label in labels] == ["cat"]
    assert labels[0].prompt == "a photo of a cat"
    assert labels[0].prompts == ("a photo of a cat",)


def test_load_labels_accepts_multiple_prompts(tmp_path: Path) -> None:
    labels_file = tmp_path / "labels.yaml"
    labels_file.write_text(
        """
labels:
  - id: cat
    name: Cat
    prompts:
      - a photo of a cat
      - a photo of a kitten
    enabled: true
""",
        encoding="utf-8",
    )

    labels = load_labels(labels_file)

    assert labels[0].prompt == "a photo of a cat"
    assert labels[0].prompts == ("a photo of a cat", "a photo of a kitten")


def test_default_labels_include_wechat_screenshot() -> None:
    labels = load_labels(Path("config/labels.yaml"))
    labels_by_id = {label.id: label for label in labels}

    assert labels_by_id["screenshot"].name == "截图"
    assert labels_by_id["wechat_screenshot"].name == "微信截图"
    assert any("WeChat" in prompt for prompt in labels_by_id["wechat_screenshot"].prompts)


def test_load_labels_rejects_empty_enabled_set(tmp_path: Path) -> None:
    labels_file = tmp_path / "labels.yaml"
    labels_file.write_text(
        """
labels:
  - id: dog
    name: Dog
    prompt: a photo of a dog
    enabled: false
""",
        encoding="utf-8",
    )

    with pytest.raises(LabelConfigError, match="at least one enabled label"):
        load_labels(labels_file)


def test_load_labels_rejects_duplicate_ids(tmp_path: Path) -> None:
    labels_file = tmp_path / "labels.yaml"
    labels_file.write_text(
        """
labels:
  - id: cat
    name: Cat
    prompt: a photo of a cat
  - id: cat
    name: Cat duplicate
    prompt: a second cat prompt
""",
        encoding="utf-8",
    )

    with pytest.raises(LabelConfigError, match="duplicate label id"):
        load_labels(labels_file)


def test_load_labels_rejects_invalid_prompt_values(tmp_path: Path) -> None:
    labels_file = tmp_path / "labels.yaml"
    labels_file.write_text(
        """
labels:
  - id: cat
    name: Cat
    prompts:
      - a photo of a cat
      - 123
""",
        encoding="utf-8",
    )

    with pytest.raises(LabelConfigError, match="prompt 1 must be a string"):
        load_labels(labels_file)


def test_load_labels_rejects_empty_prompt_values(tmp_path: Path) -> None:
    labels_file = tmp_path / "labels.yaml"
    labels_file.write_text(
        """
labels:
  - id: cat
    name: Cat
    prompt: "  "
""",
        encoding="utf-8",
    )

    with pytest.raises(LabelConfigError, match="prompt must not be empty"):
        load_labels(labels_file)
