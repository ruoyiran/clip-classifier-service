import torch


def main() -> None:
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device_count={torch.cuda.device_count()}")
        print(f"device_name={torch.cuda.get_device_name(0)}")
        print(f"capability={torch.cuda.get_device_capability(0)}")
        print(f"cuda_runtime={torch.version.cuda}")
        tensor = torch.ones((1024, 1024), device="cuda")
        print(f"gpu_sum={tensor.sum().item():.1f}")


if __name__ == "__main__":
    main()
