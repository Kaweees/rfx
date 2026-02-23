from __future__ import annotations

import atexit
import logging
import os
from contextlib import nullcontext
from typing import Any

logger = logging.getLogger(__name__)

_INITIALIZED = False


def _enabled() -> bool:
    value = os.getenv("RFX_OTEL", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def init_otel(service_name: str = "rfx") -> bool:
    """Initialize OpenTelemetry tracing if enabled via env.

    Environment variables:
    - RFX_OTEL=1                         enable instrumentation
    - RFX_OTEL_EXPORTER=console|otlp     default: console
    - RFX_OTEL_OTLP_ENDPOINT=<url>       default: http://localhost:4318/v1/traces
    """
    global _INITIALIZED
    if _INITIALIZED:
        return True
    if not _enabled():
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )
    except Exception as exc:
        logger.warning(
            "OpenTelemetry enabled but dependencies are missing (%s). "
            "Install: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp",
            exc,
        )
        return False

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    exporter = os.getenv("RFX_OTEL_EXPORTER", "console").strip().lower()
    processor = None
    if exporter == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            endpoint = os.getenv("RFX_OTEL_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
            span_exporter = OTLPSpanExporter(endpoint=endpoint)
            processor = BatchSpanProcessor(span_exporter)
        except Exception as exc:
            logger.warning("Falling back to console OpenTelemetry exporter: %s", exc)
            span_exporter = ConsoleSpanExporter()
            processor = SimpleSpanProcessor(span_exporter)
    else:
        span_exporter = ConsoleSpanExporter()
        # Console debugging should be immediate; batch mode hides spans on quick Ctrl+C exits.
        processor = SimpleSpanProcessor(span_exporter)

    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    atexit.register(flush_otel)
    _INITIALIZED = True
    logger.info("OpenTelemetry initialized (service=%s exporter=%s)", service_name, exporter)
    return True


class _NoopSpan:
    def set_attribute(self, _key: str, _value: Any) -> None:
        return None

    def add_event(self, _name: str, _attributes: dict[str, Any] | None = None) -> None:
        return None


class _NoopTracer:
    def start_as_current_span(self, _name: str, **_kwargs: Any):
        return nullcontext(_NoopSpan())


def get_tracer(name: str):
    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except Exception:
        return _NoopTracer()


def flush_otel(timeout_millis: int = 2000) -> None:
    """Force-flush OTel exporters to avoid losing spans on process shutdown."""
    try:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        force_flush = getattr(provider, "force_flush", None)
        if callable(force_flush):
            force_flush(timeout_millis=timeout_millis)
    except Exception:
        return None
