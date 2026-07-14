import argparse
import json
import os
import sys
from pathlib import Path
from urllib import error, request

from clip_classifier_service.encryption import encrypt_image_for_server


def build_headers(bearer_token: str | None, internal_api_key: str) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "internal-api-key": internal_api_key,
    }
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    return headers


def request_json(
    method: str,
    url: str,
    headers: dict[str, str],
    *,
    body: bytes | None = None,
    timeout: int,
) -> dict:
    request_headers = dict(headers)
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    api_request = request.Request(url, data=body, headers=request_headers, method=method)
    try:
        with request.urlopen(api_request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
            content_type = response.headers.get("Content-Type", "")
            if "application/json" not in content_type:
                raise RuntimeError(
                    f"expected JSON from {url}, got content-type={content_type!r}, "
                    f"body_prefix={response_body[:200]!r}"
                )
            return json.loads(response_body)
    except error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{method} {url} failed: status={exc.code}, "
            f"content-type={exc.headers.get('Content-Type')!r}, body={response_body[:1000]!r}"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test the authenticated encrypted CLIP classifier API.",
    )
    parser.add_argument("image", help="Path to the image to classify.")
    parser.add_argument(
        "--base-url",
        default="https://aidp.bytedance.com/aitesting/internal/openai-clip",
        help="API base URL including base path.",
    )
    parser.add_argument(
        "--internal-api-key",
        default=os.getenv("INTERNAL_API_KEY"),
        help="Value for the internal-api-key header. Can also be set with INTERNAL_API_KEY.",
    )
    parser.add_argument(
        "--bearer-token",
        default=os.getenv("CLIP_API_TOKEN"),
        help="Bearer token for the CLIP service. Can also be set with CLIP_API_TOKEN.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    if not args.internal_api_key:
        print(
            "Missing internal API key. Pass --internal-api-key or set INTERNAL_API_KEY.",
            file=sys.stderr,
        )
        return 2

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image does not exist: {image_path}", file=sys.stderr)
        return 2

    base_url = args.base_url.rstrip("/")
    headers = build_headers(args.bearer_token, args.internal_api_key)

    try:
        print(f"[1/3] GET {base_url}/readyz")
        ready = request_json(
            "GET",
            f"{base_url}/readyz",
            headers,
            timeout=args.timeout,
        )
        print(json.dumps(ready, ensure_ascii=False))

        print(f"[2/3] GET {base_url}/v1/encryption/public-key")
        public_key_response = request_json(
            "GET",
            f"{base_url}/v1/encryption/public-key",
            headers,
            timeout=args.timeout,
        )
        print(json.dumps({"alg": public_key_response.get("alg")}, ensure_ascii=False))

        image_bytes = image_path.read_bytes()
        payload = encrypt_image_for_server(public_key_response["public_key"], image_bytes)
        payload["filename"] = image_path.name
        body = json.dumps(payload).encode("utf-8")

        print(f"[3/3] POST {base_url}/v1/classify/encrypted?top_k={args.top_k}")
        result = request_json(
            "POST",
            f"{base_url}/v1/classify/encrypted?top_k={args.top_k}",
            headers,
            body=body,
            timeout=args.timeout,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"Test failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
