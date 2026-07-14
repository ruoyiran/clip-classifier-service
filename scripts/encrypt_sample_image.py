import argparse
import json
from pathlib import Path
from urllib import request

from clip_classifier_service.encryption import encrypt_image_for_server


def main() -> None:
    parser = argparse.ArgumentParser(description="Encrypt an image and call the classifier API.")
    parser.add_argument("image", help="Path to the image to classify.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/openai-clip")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--token", help="Bearer token for authenticated API calls.")
    args = parser.parse_args()

    auth_headers = {}
    if args.token:
        auth_headers["Authorization"] = f"Bearer {args.token}"

    public_key_url = f"{args.base_url.rstrip('/')}/v1/encryption/public-key"
    public_key_request = request.Request(public_key_url, headers=auth_headers, method="GET")
    with request.urlopen(public_key_request, timeout=15) as response:
        public_key_response = json.loads(response.read().decode("utf-8"))

    image_bytes = Path(args.image).read_bytes()
    payload = encrypt_image_for_server(public_key_response["public_key"], image_bytes)
    body = json.dumps(payload).encode("utf-8")
    classify_url = f"{args.base_url.rstrip('/')}/v1/classify/encrypted?top_k={args.top_k}"
    classify_request = request.Request(
        classify_url,
        data=body,
        headers={**auth_headers, "Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(classify_request, timeout=60) as response:
        print(response.read().decode("utf-8"))


if __name__ == "__main__":
    main()
