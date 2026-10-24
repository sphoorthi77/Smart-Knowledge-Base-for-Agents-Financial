import os, json, time, requests, re
from . import config as C
from .tracing import lf_generation

EXTRA = json.loads(os.getenv("KB_EXTRA_PAYLOAD", "{}") or "{}")
MAX_BUDGET = 32000

class EmptyContent(Exception):
    """Model returned no visible text even after escalating the token budget."""

class LLM:
    """OpenCode chat client with automatic token-budget escalation and robust JSON parsing."""

    def _post(self, payload, user, name, trace, retries=3):
        h = {"Authorization": f"Bearer {C.OPENCODE_API_KEY}", "Content-Type": "application/json"}
        err = None
        for a in range(retries):
            try:
                t0 = time.time()
                r = requests.post(C.OPENCODE_URL, headers=h, json=payload, timeout=180)
                r.raise_for_status()
                d = r.json()
                msg = d["choices"][0]["message"]
                txt = msg.get("content") or ""
                u = d.get("usage", {}) or {}
                pin, pout = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
                cost = pin / 1e6 * C.PRICE_IN + pout / 1e6 * C.PRICE_OUT
                lf_generation(trace, name, C.MODEL, user, txt, pin, pout, cost, time.time() - t0)
                return {
                    "text": txt,
                    "in": pin,
                    "out": pout,
                    "cost": cost,
                    "sec": time.time() - t0,
                    "finish": d["choices"][0].get("finish_reason"),
                }
            except Exception as e:
                err = e
                time.sleep(1.5 * (a + 1))
        raise RuntimeError(f"LLM failed after {retries} retries: {err}")

    def call(self, system, user, json_mode=False, max_tokens=6000, name="llm", trace=None, retries=3):
        budget = max_tokens
        tin = tout = 0
        cost = 0.0
        sec = 0.0
        last = ""
        for attempt in range(3):
            payload = {
                "model": C.MODEL,
                "temperature": 0,
                "max_tokens": min(budget, MAX_BUDGET),
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            payload.update(EXTRA)
            r = self._post(payload, user, name, trace, retries)
            tin += r["in"]
            tout += r["out"]
            cost += r["cost"]
            sec += r["sec"]
            if r["text"].strip():
                return {"text": r["text"], "in": tin, "out": tout, "cost": cost, "sec": sec}
            last = f"finish_reason={r['finish']}, out_tokens={r['out']}"
            budget *= 2  # empty content -> hidden reasoning consumed the budget; double budget
        raise EmptyContent(f"empty content after escalation ({last})")

    @staticmethod
    def _extract_json(t: str) -> dict:
        t = t.strip()
        if t.startswith("```"):
            t = re.sub(r"^```(?:json)?\s*", "", t)
            t = re.sub(r"\s*```$", "", t)
        t = t.strip()
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            m = re.search(r"(\{.*\})", t, re.DOTALL)
            if m:
                return json.loads(m.group(1))
            raise

    def json(self, system, user, **kw):
        """JSON call. Retries with larger budget if reply is empty, truncated, or unparseable."""
        base = kw.pop("max_tokens", 2000)
        last = None
        tin = tout = 0
        cost = 0.0
        for mult in (1, 2, 4):
            try:
                r = self.call(system, user, json_mode=True, max_tokens=base * mult, **kw)
            except Exception as e:
                last = e
                break
            tin += r["in"]
            tout += r["out"]
            cost += r["cost"]
            try:
                r["json"] = self._extract_json(r["text"])
                r["in"], r["out"], r["cost"] = tin, tout, cost
                return r
            except Exception as e:
                last = e
        raise RuntimeError(f"bad JSON after retries: {last}")
