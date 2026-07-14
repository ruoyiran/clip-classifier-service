from prometheus_client import Counter, Histogram

REQUESTS = Counter(
    "clip_classifier_requests_total",
    "Total classification requests.",
    ["endpoint", "status"],
)
INFERENCE_SECONDS = Histogram(
    "clip_classifier_inference_seconds",
    "Image classification inference latency.",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
