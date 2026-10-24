"""Langfuse observability integration supporting v4 OpenTelemetry client.
Uses env LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST.
Wrapped so a tracing hiccup can never break /query."""
import os
from . import config as C

_lf = None

def _client():
    global _lf
    if _lf is None and os.getenv("LANGFUSE_PUBLIC_KEY"):
        try:
            from langfuse import Langfuse
            _lf = Langfuse()
        except Exception:
            _lf = False
    return _lf or None

class TraceWrapper:
    def __init__(self, client, name, inp, meta=None):
        self.client = client
        self.cm = client.start_as_current_observation(name=name, as_type="span", input=inp)
        self.span = self.cm.__enter__()
        try:
            self.url = client.get_trace_url()
        except Exception:
            self.url = None

    def end(self, output, meta=None):
        try:
            if self.span:
                self.span.update(output=output[:4000] if isinstance(output, str) else output, metadata=meta or {})
        except Exception:
            pass
        finally:
            try:
                self.cm.__exit__(None, None, None)
            except Exception:
                pass
            try:
                self.client.flush()
            except Exception:
                pass

def lf_trace(name, inp, meta=None):
    c = _client()
    if not c:
        return None
    try:
        return TraceWrapper(c, name, inp, meta)
    except Exception:
        return None

def lf_generation(trace, name, model, inp, out, pin, pout, cost, sec):
    c = _client()
    if not c or not trace:
        return
    try:
        with c.start_as_current_observation(
            name=name,
            as_type="generation",
            model=model,
            input=inp[:4000] if isinstance(inp, str) else inp,
            metadata={"latency_s": round(sec, 3), "cost_usd": round(cost, 6)}
        ) as gen:
            gen.update(
                output=out[:4000] if isinstance(out, str) else out,
                usage_details={"input": pin, "output": pout, "total": pin + pout}
            )
    except Exception:
        pass

def lf_end(trace, output, meta):
    if trace and isinstance(trace, TraceWrapper):
        trace.end(output, meta)
