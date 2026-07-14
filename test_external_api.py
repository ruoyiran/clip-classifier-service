import argparse
import json
import os
import sys
from pathlib import Path
from urllib import error, request

IMAGE_SUFFIXES = {
    ".avif",
    ".bmp",
    ".gif",
    ".heic",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def add_local_src_to_path() -> None:
    src_path = Path(__file__).resolve().parent / "src"
    if src_path.is_dir() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def load_encrypt_image_for_server():
    add_local_src_to_path()
    try:
        from clip_classifier_service.encryption import encrypt_image_for_server
    except ModuleNotFoundError as exc:
        if exc.name == "clip_classifier_service":
            raise RuntimeError(
                "could not import clip_classifier_service. Run this script from the "
                "repository root or install the package with "
                "`python -m pip install -e . --no-deps`."
            ) from exc
        raise RuntimeError(
            f"missing Python dependency {exc.name!r}. Install project dependencies first, "
            "for example `python -m pip install -r requirements-dev.txt`."
        ) from exc
    return encrypt_image_for_server


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


def expand_image_paths(inputs: list[str], recursive: bool) -> list[Path]:
    image_paths: list[Path] = []
    for value in inputs:
        path = Path(value).expanduser()
        if path.is_file():
            image_paths.append(path)
            continue
        if path.is_dir():
            candidates = path.rglob("*") if recursive else path.iterdir()
            image_paths.extend(
                sorted(
                    candidate
                    for candidate in candidates
                    if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
                )
            )
            continue
        raise RuntimeError(f"image path does not exist: {path}")
    return image_paths


def chunks(items: list[Path], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def extract_batch_results(response: dict, expected_count: int) -> list[dict]:
    results = response.get("results")
    if not isinstance(results, list):
        raise RuntimeError("batch response is missing a results list")
    if len(results) != expected_count:
        raise RuntimeError(
            f"batch response count mismatch: expected {expected_count}, got {len(results)}"
        )
    return results


def query_string(top_k: int, threshold: float | None) -> str:
    query = f"top_k={top_k}"
    if threshold is not None:
        query += f"&threshold={threshold}"
    return query


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test the authenticated encrypted CLIP classifier API.",
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Image file paths, or directories containing images.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="API base URL including base path. Pass the production URL explicitly.",
    )
    parser.add_argument(
        "--internal-api-key",
        default=os.getenv("AI_INTERNAL_API_KEY"),
        help="Value for the internal-api-key header. Can also be set with AI_INTERNAL_API_KEY.",
    )
    parser.add_argument(
        "--bearer-token",
        default=os.getenv("OPENAI_CLIP_AUTH_TOKEN"),
        help="Bearer token for the CLIP service. Can also be set with OPENAI_CLIP_AUTH_TOKEN.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional service score threshold. Omitted by default.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Images per encrypted batch request. Must not exceed CLIP_MAX_BATCH_SIZE.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan directory arguments for images.",
    )
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    if not args.internal_api_key:
        print(
            "Missing internal API key. Pass --internal-api-key or set AI_INTERNAL_API_KEY.",
            file=sys.stderr,
        )
        return 2

    if args.batch_size < 1:
        print("--batch-size must be >= 1.", file=sys.stderr)
        return 2

    try:
        image_paths = expand_image_paths(args.images, recursive=args.recursive)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not image_paths:
        print("No images found.", file=sys.stderr)
        return 2

    base_url = args.base_url.rstrip("/")
    headers = build_headers(args.bearer_token, args.internal_api_key)
    query = query_string(args.top_k, args.threshold)

    try:
        encrypt_image_for_server = load_encrypt_image_for_server()

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

        if len(image_paths) == 1:
            image_path = image_paths[0]
            payload = encrypt_image_for_server(
                public_key_response["public_key"],
                image_path.read_bytes(),
            )
            payload["filename"] = image_path.name
            body = json.dumps(payload).encode("utf-8")

            print(f"[3/3] POST {base_url}/v1/classify/encrypted?{query}")
            result = request_json(
                "POST",
                f"{base_url}/v1/classify/encrypted?{query}",
                headers,
                body=body,
                timeout=args.timeout,
            )
        else:
            collected_results = []
            batches = list(chunks(image_paths, args.batch_size))
            for index, batch in enumerate(batches, start=1):
                items = []
                for image_path in batch:
                    payload = encrypt_image_for_server(
                        public_key_response["public_key"],
                        image_path.read_bytes(),
                    )
                    payload["filename"] = image_path.name
                    items.append(payload)
                body = json.dumps({"items": items}).encode("utf-8")

                print(
                    f"[3/3] POST {base_url}/v1/classify/batch/encrypted?{query} "
                    f"(batch {index}/{len(batches)}, images={len(batch)})"
                )
                response = request_json(
                    "POST",
                    f"{base_url}/v1/classify/batch/encrypted?{query}",
                    headers,
                    body=body,
                    timeout=args.timeout,
                )
                collected_results.extend(
                    extract_batch_results(response, expected_count=len(batch))
                )
            result = {"results": collected_results}
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"Test failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
