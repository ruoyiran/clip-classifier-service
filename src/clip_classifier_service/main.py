from clip_classifier_service.api import create_app
from clip_classifier_service.settings import Settings

settings = Settings()
app = create_app(settings)
