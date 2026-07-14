# API Call Examples

These examples target the current local deployment:

- Base URL: `http://127.0.0.1:8080/openai-clip`
- Auth: `Authorization: Bearer <token>`
- Classification mode: encrypted upload
- Supported image formats: JPEG/JPG, PNG, BMP, HEIC, HEIF
- Max decrypted image size: 30 MiB
- Max encrypted JSON request body: 42 MiB
- nginx max request body: 50 MiB

Read the token from the deployed `.env` on the server:

```powershell
$token = ((Select-String -Path C:\Users\Admin\deploy\clip-classifier-service\.env -Pattern '^CLIP_API_TOKENS=').Line -split '=', 2)[1].Split(',')[0]
```

Do not send plaintext classification requests in the current production config. `CLIP_REQUIRE_ENCRYPTED_UPLOADS=true` rejects `/v1/classify` and `/v1/classify/batch`.

## curl

Health and readiness:

```bash
TOKEN="<token>"
BASE_URL="http://127.0.0.1:8080/openai-clip"

curl -H "Authorization: Bearer ${TOKEN}" "${BASE_URL}/healthz"
curl -H "Authorization: Bearer ${TOKEN}" "${BASE_URL}/readyz"
```

Fetch Prometheus metrics:

```bash
curl -H "Authorization: Bearer ${TOKEN}" "${BASE_URL}/metrics"
```

Fetch the encryption public key:

```bash
curl -H "Authorization: Bearer ${TOKEN}" "${BASE_URL}/v1/encryption/public-key"
```

Classify with an already encrypted payload JSON:

```bash
curl -X POST \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  --data-binary @payload.json \
  "${BASE_URL}/v1/classify/encrypted?top_k=3"
```

`payload.json` shape:

```json
{
  "alg": "AES-256-GCM+SealedBox-v1",
  "encrypted_key": "<base64 sealed AES-256 key>",
  "nonce": "<base64 12-byte AES-GCM nonce>",
  "ciphertext": "<base64 AES-GCM ciphertext plus tag>",
  "filename": "image.heic"
}
```

Batch encrypted classify:

```bash
curl -X POST \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  --data-binary @batch-payload.json \
  "${BASE_URL}/v1/classify/batch/encrypted?top_k=3"
```

`batch-payload.json` shape:

```json
{
  "items": [
    {
      "alg": "AES-256-GCM+SealedBox-v1",
      "encrypted_key": "<base64 sealed AES-256 key>",
      "nonce": "<base64 12-byte AES-GCM nonce>",
      "ciphertext": "<base64 AES-GCM ciphertext plus tag>",
      "filename": "one.png"
    }
  ]
}
```

If you are testing locally with plaintext endpoints explicitly allowed, this multipart request is valid:

```bash
curl -X POST \
  -H "Authorization: Bearer ${TOKEN}" \
  -F "file=@C:\path\image.heic" \
  "${BASE_URL}/v1/classify?top_k=3"
```

## Python

Install client dependencies:

```bash
pip install requests cryptography pynacl
```

Single encrypted classification:

```python
import base64
import os
from pathlib import Path

import requests
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from nacl.public import PublicKey, SealedBox


BASE_URL = "http://127.0.0.1:8080/openai-clip"
TOKEN = "<token>"
IMAGE_PATH = r"C:\path\image.heic"


def b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def encrypt_image(public_key_b64: str, image_bytes: bytes) -> dict[str, str]:
    aes_key = os.urandom(32)
    nonce = os.urandom(12)
    ciphertext = AESGCM(aes_key).encrypt(nonce, image_bytes, None)
    sealed_key = SealedBox(PublicKey(base64.b64decode(public_key_b64))).encrypt(aes_key)
    return {
        "alg": "AES-256-GCM+SealedBox-v1",
        "encrypted_key": b64(sealed_key),
        "nonce": b64(nonce),
        "ciphertext": b64(ciphertext),
        "filename": Path(IMAGE_PATH).name,
    }


headers = {"Authorization": f"Bearer {TOKEN}"}

public_key_response = requests.get(
    f"{BASE_URL}/v1/encryption/public-key",
    headers=headers,
    timeout=15,
)
public_key_response.raise_for_status()
public_key = public_key_response.json()["public_key"]

payload = encrypt_image(public_key, Path(IMAGE_PATH).read_bytes())

response = requests.post(
    f"{BASE_URL}/v1/classify/encrypted",
    params={"top_k": 3},
    json=payload,
    headers=headers,
    timeout=60,
)
response.raise_for_status()
print(response.json())
```

Batch encrypted classification:

```python
from pathlib import Path

image_paths = [Path(r"C:\path\one.png"), Path(r"C:\path\two.bmp")]

items = []
for image_path in image_paths:
    item = encrypt_image(public_key, image_path.read_bytes())
    item["filename"] = image_path.name
    items.append(item)

response = requests.post(
    f"{BASE_URL}/v1/classify/batch/encrypted",
    params={"top_k": 3},
    json={"items": items},
    headers=headers,
    timeout=120,
)
response.raise_for_status()
print(response.json())
```

Expected successful single-image response shape:

```json
{
  "model": "hf-hub:apple/DFN5B-CLIP-ViT-H-14-378",
  "elapsed_ms": 123.45,
  "labels": [
    {"id": "cat", "name": "猫", "score": 0.8703, "similarity": 0.3124}
  ]
}
```

Batch endpoints wrap the same per-image shape in `results` and include each
item's `filename` when supplied:

```json
{
  "results": [
    {
      "filename": "one.png",
      "model": "hf-hub:apple/DFN5B-CLIP-ViT-H-14-378",
      "elapsed_ms": 123.45,
      "labels": [
        {"id": "cat", "name": "猫", "score": 0.8703, "similarity": 0.3124}
      ]
    }
  ]
}
```

## Go

Initialize dependencies:

```bash
go mod init clip-client
go get golang.org/x/crypto/nacl/box
```

Single encrypted classification:

```go
package main

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"

	"golang.org/x/crypto/nacl/box"
)

const (
	baseURL = "http://127.0.0.1:8080/openai-clip"
	token   = "<token>"
	image   = `C:\path\image.heic`
)

type PublicKeyResponse struct {
	Alg       string `json:"alg"`
	PublicKey string `json:"public_key"`
}

type EncryptedImagePayload struct {
	Alg          string `json:"alg"`
	EncryptedKey string `json:"encrypted_key"`
	Nonce        string `json:"nonce"`
	Ciphertext   string `json:"ciphertext"`
	Filename     string `json:"filename,omitempty"`
}

func main() {
	publicKey, err := fetchPublicKey()
	if err != nil {
		panic(err)
	}

	imageBytes, err := os.ReadFile(image)
	if err != nil {
		panic(err)
	}

	payload, err := encryptImage(publicKey, imageBytes, filepath.Base(image))
	if err != nil {
		panic(err)
	}

	body, err := json.Marshal(payload)
	if err != nil {
		panic(err)
	}

	req, err := http.NewRequest("POST", baseURL+"/v1/classify/encrypted?top_k=3", bytes.NewReader(body))
	if err != nil {
		panic(err)
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()

	responseBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		panic(fmt.Sprintf("status=%d body=%s", resp.StatusCode, responseBody))
	}
	fmt.Println(string(responseBody))
}

func fetchPublicKey() (string, error) {
	req, err := http.NewRequest("GET", baseURL+"/v1/encryption/public-key", nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+token)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("status=%d body=%s", resp.StatusCode, body)
	}

	var parsed PublicKeyResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return "", err
	}
	return parsed.PublicKey, nil
}

func encryptImage(publicKeyB64 string, imageBytes []byte, filename string) (EncryptedImagePayload, error) {
	publicKeyBytes, err := base64.StdEncoding.DecodeString(publicKeyB64)
	if err != nil {
		return EncryptedImagePayload{}, err
	}
	if len(publicKeyBytes) != 32 {
		return EncryptedImagePayload{}, fmt.Errorf("public key must be 32 bytes")
	}
	var publicKey [32]byte
	copy(publicKey[:], publicKeyBytes)

	aesKey := make([]byte, 32)
	if _, err := rand.Read(aesKey); err != nil {
		return EncryptedImagePayload{}, err
	}
	nonce := make([]byte, 12)
	if _, err := rand.Read(nonce); err != nil {
		return EncryptedImagePayload{}, err
	}

	block, err := aes.NewCipher(aesKey)
	if err != nil {
		return EncryptedImagePayload{}, err
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return EncryptedImagePayload{}, err
	}
	ciphertext := gcm.Seal(nil, nonce, imageBytes, nil)

	sealedKey, err := box.SealAnonymous(nil, aesKey, &publicKey, rand.Reader)
	if err != nil {
		return EncryptedImagePayload{}, err
	}

	return EncryptedImagePayload{
		Alg:          "AES-256-GCM+SealedBox-v1",
		EncryptedKey: base64.StdEncoding.EncodeToString(sealedKey),
		Nonce:        base64.StdEncoding.EncodeToString(nonce),
		Ciphertext:   base64.StdEncoding.EncodeToString(ciphertext),
		Filename:     filename,
	}, nil
}
```

Batch encrypted classification uses the same `EncryptedImagePayload` type:

```go
type BatchPayload struct {
	Items []EncryptedImagePayload `json:"items"`
}

// Build []EncryptedImagePayload with encryptImage(...), then:
body, _ := json.Marshal(BatchPayload{Items: items})
req, _ := http.NewRequest("POST", baseURL+"/v1/classify/batch/encrypted?top_k=3", bytes.NewReader(body))
req.Header.Set("Authorization", "Bearer "+token)
req.Header.Set("Content-Type", "application/json")
```

## Common Errors

- `401 missing bearer token`: Add `Authorization: Bearer <token>`.
- `401 invalid bearer token`: The token is wrong or no longer configured.
- `403 encrypted uploads are required`: Use `/v1/classify/encrypted` or `/v1/classify/batch/encrypted`.
- `413 request body too large`: The encrypted JSON body exceeded 42 MiB or nginx exceeded 50 MiB.
- `413 image exceeds max_image_bytes`: The decrypted image exceeded 30 MiB.
- `400 invalid encrypted payload`: The payload is malformed, the public key is stale, or ciphertext was modified.
