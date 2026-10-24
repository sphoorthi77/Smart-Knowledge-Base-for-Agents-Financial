import os

# Load .env if present
_env_file = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_env_file):
    with open(_env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

# ---- Secrets: ONLY from environment variables (never hardcode, never commit) ----
OPENCODE_API_KEY = os.getenv("OPENCODE_API_KEY", "")

# Langfuse tracing: enabled only when both keys are set in the environment.
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
    os.environ["LANGFUSE_HOST"] = LANGFUSE_HOST
else:
    # make tracing.py's _client() a no-op instead of failing
    os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
    os.environ.pop("LANGFUSE_SECRET_KEY", None)

OPENCODE_URL = os.getenv("OPENCODE_URL", "https://opencode.ai/inference/openai/v1/chat/completions")
MODEL = os.getenv("KB_MODEL", "deepseek-v4.1-flash")     # paid model, NOT a -free one (per organizer guide)
DATA_DIR = os.getenv("KB_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
# $ per 1M tokens (edit to the real OpenCode price so Langfuse cost is right)
PRICE_IN = float(os.getenv("PRICE_IN_PER_M", "0.15"))
PRICE_OUT = float(os.getenv("PRICE_OUT_PER_M", "0.60"))
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "900"))      # ~ benchmark chunk size (avg 858 chars)
OCR_MIN_CHARS = 40                                       # page with less text than this -> OCR
# evidence budget by question breadth. NOTE: dev gold sets reach ~100-350 chunks for broad questions;
# a cap of 45 limits context recall there. Tune on the full 100 dev questions.
K_BY_BREADTH = {"narrow": 8, "medium": 20, "broad": 45}
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "90000"))
DENSE = os.getenv("DENSE", "0") == "1"
