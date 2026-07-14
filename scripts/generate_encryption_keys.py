import argparse
from pathlib import Path

from clip_classifier_service.encryption import write_key_pair


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate service encryption keys.")
    parser.add_argument("--private-key", default="config/server_private.key")
    parser.add_argument("--public-key", default="config/server_public.key")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    private_key_path = Path(args.private_key)
    public_key_path = Path(args.public_key)
    if not args.overwrite and (private_key_path.exists() or public_key_path.exists()):
        raise SystemExit("key file already exists; pass --overwrite to replace it")

    write_key_pair(private_key_path, public_key_path)
    print(f"private_key={private_key_path}")
    print(f"public_key={public_key_path}")


if __name__ == "__main__":
    main()
