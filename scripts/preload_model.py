from clip_classifier_service.api import create_classifier
from clip_classifier_service.settings import Settings


def main() -> None:
    settings = Settings()
    classifier = create_classifier(settings)
    print(f"model={classifier.model_name}")
    print(f"labels={len(classifier.labels)}")
    print("ready=true")


if __name__ == "__main__":
    main()
