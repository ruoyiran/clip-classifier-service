from __future__ import annotations

import base64
import binascii
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from nacl.exceptions import CryptoError
from nacl.public import PrivateKey, PublicKey, SealedBox

ENCRYPTION_ALGORITHM = "AES-256-GCM+SealedBox-v1"
AES_256_KEY_BYTES = 32
AES_GCM_NONCE_BYTES = 12


@dataclass(frozen=True)
class EncryptionKeyPair:
    private_key_b64: str
    public_key_b64: str


def generate_key_pair() -> EncryptionKeyPair:
    private_key = PrivateKey.generate()
    public_key = private_key.public_key
    return EncryptionKeyPair(
        private_key_b64=encode_b64(bytes(private_key)),
        public_key_b64=encode_b64(bytes(public_key)),
    )


def write_key_pair(private_key_path: Path, public_key_path: Path) -> EncryptionKeyPair:
    key_pair = generate_key_pair()
    private_key_path.parent.mkdir(parents=True, exist_ok=True)
    public_key_path.parent.mkdir(parents=True, exist_ok=True)
    private_key_path.write_text(key_pair.private_key_b64, encoding="ascii")
    public_key_path.write_text(key_pair.public_key_b64, encoding="ascii")
    return key_pair


def read_key_b64(path: Path) -> str:
    if not path.exists():
        raise ValueError(f"encryption key file does not exist: {path}")
    return path.read_text(encoding="ascii").strip()


def encrypt_image_for_server(public_key_b64: str, image_bytes: bytes) -> dict[str, str]:
    aes_key = os.urandom(AES_256_KEY_BYTES)
    nonce = os.urandom(AES_GCM_NONCE_BYTES)
    ciphertext = AESGCM(aes_key).encrypt(nonce, image_bytes, None)
    encrypted_key = SealedBox(PublicKey(decode_b64(public_key_b64))).encrypt(aes_key)
    return {
        "alg": ENCRYPTION_ALGORITHM,
        "encrypted_key": encode_b64(encrypted_key),
        "nonce": encode_b64(nonce),
        "ciphertext": encode_b64(ciphertext),
    }


def decrypt_encrypted_image_payload(payload: dict[str, Any], private_key_b64: str) -> bytes:
    if payload.get("alg") != ENCRYPTION_ALGORITHM:
        raise ValueError("unsupported encryption algorithm")

    try:
        encrypted_key = decode_b64(str(payload["encrypted_key"]))
        nonce = decode_b64(str(payload["nonce"]))
        ciphertext = decode_b64(str(payload["ciphertext"]))
        private_key = PrivateKey(decode_b64(private_key_b64))
        aes_key = SealedBox(private_key).decrypt(encrypted_key)
        if len(aes_key) != AES_256_KEY_BYTES or len(nonce) != AES_GCM_NONCE_BYTES:
            raise ValueError("invalid encrypted payload")
        return AESGCM(aes_key).decrypt(nonce, ciphertext, None)
    except (KeyError, binascii.Error, CryptoError, InvalidTag, ValueError) as exc:
        if str(exc) == "unsupported encryption algorithm":
            raise
        raise ValueError("invalid encrypted payload") from exc


def encode_b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def decode_b64(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"), validate=True)
