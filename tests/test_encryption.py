import pytest

from clip_classifier_service.encryption import (
    ENCRYPTION_ALGORITHM,
    decrypt_encrypted_image_payload,
    encrypt_image_for_server,
    generate_key_pair,
)


def test_hybrid_encryption_round_trip() -> None:
    key_pair = generate_key_pair()

    payload = encrypt_image_for_server(
        public_key_b64=key_pair.public_key_b64,
        image_bytes=b"fake-image",
    )

    assert payload["alg"] == ENCRYPTION_ALGORITHM
    assert decrypt_encrypted_image_payload(payload, key_pair.private_key_b64) == b"fake-image"


def test_hybrid_encryption_rejects_tampered_ciphertext() -> None:
    key_pair = generate_key_pair()
    payload = encrypt_image_for_server(
        public_key_b64=key_pair.public_key_b64,
        image_bytes=b"fake-image",
    )
    payload["ciphertext"] = payload["ciphertext"][:-2] + "AA"

    with pytest.raises(ValueError, match="invalid encrypted payload"):
        decrypt_encrypted_image_payload(payload, key_pair.private_key_b64)


def test_hybrid_encryption_rejects_unknown_algorithm() -> None:
    key_pair = generate_key_pair()
    payload = encrypt_image_for_server(
        public_key_b64=key_pair.public_key_b64,
        image_bytes=b"fake-image",
    )
    payload["alg"] = "AES-256-CBC"

    with pytest.raises(ValueError, match="unsupported encryption algorithm"):
        decrypt_encrypted_image_payload(payload, key_pair.private_key_b64)
