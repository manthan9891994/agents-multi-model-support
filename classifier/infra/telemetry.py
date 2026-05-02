"""OpenTelemetry tracing — emits a span for every router.classify() call.

Zero-config: if `opentelemetry-api` is not installed, all helpers degrade to
no-ops. If it IS installed, host applications can configure exporters
(Jaeger / OTLP / Cloud Trace) however they like — we only emit, we don't
configure.

Usage from host application:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    trace.set_tracer_provider(TracerProvider())
    trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))

    # All router.classify() calls now emit traces.
"""
from __future__ import annotations

import contextlib
from typing import Any, Iterator

try:
    from opentelemetry import trace as _otel_trace
    _OTEL_AVAILABLE = True
    _tracer = _otel_trace.get_tracer("dynamic-model-router", "0.1.0")
except ImportError:
    _OTEL_AVAILABLE = False
    _tracer = None


@contextlib.contextmanager
def span(name: str, **attrs: Any) -> Iterator[Any]:
    """Start a span. No-op if opentelemetry-api isn't installed.

    Example:
        with span("router.classify", task_len=len(task)) as s:
            decision = ...
            s.set_attribute("tier", decision.tier.value) if s else None
    """
    if not _OTEL_AVAILABLE or _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as s:
        for k, v in attrs.items():
            try:
                s.set_attribute(k, v)
            except Exception:
                pass
        yield s


def set_attribute(span_obj: Any, key: str, value: Any) -> None:
    """Safely set an attribute on a span (no-op if span is None)."""
    if span_obj is None:
        return
    try:
        span_obj.set_attribute(key, value)
    except Exception:
        pass


def is_enabled() -> bool:
    return _OTEL_AVAILABLE
