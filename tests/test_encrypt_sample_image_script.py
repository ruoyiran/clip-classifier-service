import json
from pathlib import Path
from urllib.request import Request

import scripts.encrypt_sample_image as encrypt_sample_image


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def test_encrypt_sample_image_sends_bearer_token_to_all_api_calls(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"fake-image")
    captured_requests: list[Request] = []

    def fake_urlopen(target, timeout: int):
        assert timeout in {15, 60}
        assert isinstance(target, Request)
        captured_requests.append(target)
        if target.full_url.endswith("/v1/encryption/public-key"):
            return FakeResponse(json.dumps({"public_key": "server-public-key"}).encode("utf-8"))
        return FakeResponse(b'{"labels":[]}')

    monkeypatch.setattr(encrypt_sample_image.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        encrypt_sample_image,
        "encrypt_image_for_server",
        lambda public_key, image_bytes: {
            "alg": "AES-256-GCM+SealedBox-v1",
            "encrypted_key": "encrypted-key",
            "nonce": "nonce",
            "ciphertext": "ciphertext",
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "encrypt_sample_image.py",
            str(image_path),
            "--base-url",
            "http://127.0.0.1:8080/openai-clip",
            "--token",
            "primary-token",
        ],
    )

    encrypt_sample_image.main()

    assert len(captured_requests) == 2
    assert [request.headers["Authorization"] for request in captured_requests] == [
        "Bearer primary-token",
        "Bearer primary-token",
    ]
    assert captured_requests[1].headers["Content-type"] == "application/json"
    assert capsys.readouterr().out == '{"labels":[]}\n'
