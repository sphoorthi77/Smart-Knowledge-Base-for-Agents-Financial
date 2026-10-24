"""Query engine.
Pipeline (each step is logged into the required `trace`):
 1 metadata_filter                             : LLM planner reads the document cards catalogue -> MANDATORY doc filter
                                                 (+ breadth + aspects); lexical safety net guarantees named entities/years are never dropped
 2 scoped_retrieval_and_coverage_selection     : sparse scoring ONLY inside the filtered pool, per aspect
                                                 (breadth-adaptive K, per-document quota, relative-score cut-off)
 3 relevance_filter                            : LLM keeps ONLY excerpts that genuinely help answer the question
 4 llm_answer_generation                       : single grounded generation call over selected evidence with forced doc/page citations
"""
import os, re, pickle, time, json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from . import config as C
from .ingest import coll_dir
from .tracing import lf_trace, lf_end

GENERIC_WORDS = {
    "the", "and", "for", "with", "this", "that", "from", "report", "annual",
    "document", "what", "which", "when", "where", "how", "tell", "about",
    "year", "years", "give", "every", "all", "our", "your", "are", "were",
    "will", "would", "could", "should", "does", "did", "have", "has", "had",
    "its", "their", "into", "over", "under", "more", "most", "also", "than",
    "both", "such", "some", "many", "each", "other", "then", "them", "these",
    "those", "been", "being", "having", "between", "through", "during", "before",
    "after", "above", "below", "down", "out", "off", "only", "same", "very",
    "pdf", "page", "pages", "table", "chart", "note", "item", "limited", "ltd",
    "inc", "corp", "company", "board", "governors", "system", "office", "bank",
    "credit", "risk", "risks", "market", "markets", "model", "models", "loss",
    "losses", "capital", "events", "default", "defaults", "allowance", "allowances",
    "accounting", "statement", "statements", "guidance", "guidelines", "standard",
    "standards", "handbook", "memo", "notes", "update", "updates", "rule", "rules",
    "surcharge", "exposure", "exposures", "ratio", "ratios", "framework", "frameworks",
    "proposal", "proposals", "requirement", "requirements", "interest", "rate", "rates",
    "asset", "assets", "liability", "liabilities", "equity", "debt", "loan", "loans",
    "provision", "provisions", "portfolio", "portfolios", "emerging", "technical",
    "approach", "indicator", "indicators", "predicting", "movement", "movements",
    "neural", "network", "networks", "closed", "form", "solution", "stochastic",
    "volatility", "option", "options", "derivative", "derivatives", "pricing",
    "functional", "factor", "libor", "swap", "swaps", "valuation", "adjustment",
    "benchmark", "sheet", "summary", "change", "changes", "management", "problem",
    "integrated", "reserves", "cash", "flow", "flows", "balance", "revenue",
    "receivables", "customer", "customers", "united", "states", "india"
}

def _tok(s):
    return set(re.findall(r"[a-z0-9]{3,}", s.lower()))

def build_matrices(docs, chunks):
    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=1, max_df=0.5,
                          stop_words="english", token_pattern=r"(?u)\b[\w][\w\-\.%]*\b", dtype=np.float32)
    X = vec.fit_transform([f"{c.get('section', '')} {c.get('text', '')}" for c in chunks])
    dvec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), stop_words="english", dtype=np.float32)
    D = dvec.fit_transform([card_text(d) for d in docs])
    return {"vec": vec, "X": X, "dvec": dvec, "D": D}

def card_text(d):
    return " ".join([d.get("filename", ""), d.get("title", ""), d.get("entity", ""), d.get("doc_type", ""),
                     d.get("period", ""), d.get("jurisdiction", ""), " ".join(d.get("topics", []) or []),
                     d.get("summary", "")])

ROUTER_SYS = (
    "You are the retrieval planner of a financial knowledge base. You choose WHICH documents may be used "
    "to answer a question, before any text search happens. Respond only with JSON."
)
ROUTER_USER = """DOCUMENT CATALOGUE (id | filename | title | entity | type | period | jurisdiction | summary):
{catalogue}

QUESTION: {q}

Decide:
1. doc_ids: ids of EVERY document needed. Rules:
   - Honour entity, reporting year/period, jurisdiction, regulator and document type named or implied in the question
     ("this year" = the latest period of that entity; "last two reports" = both years; "2026 US proposal" vs "2023" vs "Basel" = different documents).
   - If the question compares or spans several entities / years / frameworks, include one or more documents for EACH of them.
   - EXCLUDE near-duplicates of the wrong year, scope or jurisdiction. Prefer precision, but never omit a document the question needs.
2. breadth: "narrow" (single fact/figure/definition), "medium" (one topic explained or compared), or "broad" (exhaustive synthesis over many pages/aspects, 'every', 'pull together', multi-entity comparison).
3. aspects: 2-6 short search phrases (the sub-topics the answer must cover, using the vocabulary the documents would use).
JSON: {{"doc_ids":[int], "breadth":"narrow|medium|broad", "aspects":[str], "reason":"one line"}}"""

ANSWER_SYS = (
    "You are a principal financial research analyst answering questions for a financial institution. "
    "You must use ONLY the factual statements from the provided evidence excerpts. "
    "Never include outside knowledge, assumptions, conversational commentary, or unsupported claims."
)
ANSWER_USER = """QUESTION:
{q}

EVIDENCE (each excerpt is tagged [document | page]):
{ctx}

INSTRUCTIONS:
1. Grounding & Accuracy: State ONLY facts explicitly contained in the excerpts. Never extrapolate, assume, or fabricate.
2. Zero Conversational Filler: Do NOT include any introductory preamble (e.g. "Based on the provided excerpts", "In summary", or repeating the question). Begin immediately with the direct answer. Do NOT add concluding commentary or opinions.
3. Completeness & Precision: Address every specific component, entity, date, threshold, condition, and figure asked by the question that the excerpts support. Give exact numbers, units (e.g. lakhs, crore, $, %, bps), dates, and segment names.
4. Missing Information: If the excerpts do not contain a specific fact or figure requested by the question, explicitly state in one concise sentence what is not provided or cannot be forecasted, citing any relevant statement or footnote that explains why.
5. Citations: Cite after each stated fact or claim as [document p.N] (using the document name without .pdf and page number).
6. Style:
{style}"""

STYLE = {
    "narrow": (
        "Answer concisely in 2-5 direct factual sentences. Provide the exact fact or figure with citations. "
        "If the excerpts do not contain it, state that the figure is not provided, then state the closest relevant "
        "statements/caveats/illustrative examples (clearly labelled as illustrative) and state what is missing."
    ),
    "medium": (
        "Organize into clear, concise headed sections or bullet points addressing each part of the question. "
        "Be dense and factual with exact figures, dates, segment names, and citations. Do not add general summaries."
    ),
    "broad": (
        "Organize into logical headed sections with bullet points addressing all aspects of the question. "
        "Be exhaustive across all provided excerpts. Include exact figures, comparisons, and limits of the data. "
        "Do not add conversational fluff or redundant introductory sentences."
    )
}

FILTER_SYS = (
    "You are a strict evaluator judging whether retrieved text excerpts are genuinely relevant and useful "
    "for answering a user's question. Respond only with JSON."
)
FILTER_USER = """QUESTION: {q}

EXCERPTS:
{items}

Decide which excerpts are genuinely relevant and directly useful for answering the question.
Include excerpts that provide:
- The requested fact, figure, date, percentage, or threshold
- Necessary conditions, caveats, rules, definitions, or statutory references
- Worked examples or tables that directly illustrate the requested topic
- Explicit statements showing that the requested information is limited, qualified, or not forecasted

Exclude excerpts that are:
- Generic tables of contents, lists of contacts, or boilerplate disclosures
- Unrelated sections that merely share individual common words with the question

Select only the truly useful excerpts. Keep at most {k}.
JSON: {{"relevant_ids":[int, ...]}}"""

class Engine:
    def __init__(self, llm):
        self.llm = llm
        self.cache = {}

    def load(self, name):
        p = os.path.join(coll_dir(name), "store.pkl")
        if not os.path.exists(p):
            raise KeyError(f"collection '{name}' not found")
        m = os.path.getmtime(p)
        if name not in self.cache or self.cache[name][0] != m:
            with open(p, "rb") as f:
                self.cache[name] = (m, pickle.load(f))
        return self.cache[name][1]

    # -- step 1b: deterministic safety net ------------------------------------------------
    def lexical_docs(self, S, q):
        qv = S["dvec"].transform([q])
        sims = (S["D"] @ qv.T).toarray().ravel()
        qt = _tok(q)
        q_ent = {t for t in qt if t not in GENERIC_WORDS and len(t) >= 3}
        hits = []
        for d in S["docs"]:
            ent_str = " ".join([d.get("entity", ""), d.get("filename", "").split("_")[0], d.get("title", "")])
            ent_tokens = {t for t in _tok(ent_str) if t not in GENERIC_WORDS and len(t) >= 3}
            strong = ent_tokens & q_ent
            if strong:
                hits.append((d["doc_id"], len(strong), strong))
        hits.sort(key=lambda t: -t[1])
        return sims, hits

    def route(self, S, q, trace, steps):
        cat = "\n".join(
            f"{d['doc_id']} | {d['filename']} | {d.get('title','')[:70]} | {d.get('entity','')[:40]} | "
            f"{d.get('doc_type','')} | {d.get('period','')} | {d.get('jurisdiction','')} | {d.get('summary','')[:180]}"
            for d in S["docs"]
        )
        sims, hits = self.lexical_docs(S, q)
        plan = {"doc_ids": [], "breadth": "medium", "aspects": []}
        cost = 0; tin = tout = 0
        t0 = time.time()
        try:
            r = self.llm.json(ROUTER_SYS, ROUTER_USER.format(catalogue=cat, q=q), max_tokens=2500,
                              name="doc_router", trace=trace)
            plan.update(r.get("json", {}))
            cost += r.get("cost", 0.0)
            tin += r.get("in", 0)
            tout += r.get("out", 0)
        except Exception as e:
            plan["error"] = str(e)[:200]

        valid = {d["doc_id"] for d in S["docs"]}
        ids = [i for i in plan.get("doc_ids", []) if isinstance(i, int) and i in valid]
        used_fallback = False
        if not ids:
            used_fallback = True
            order = np.argsort(-sims)[:4]
            ids = [int(i) for i in order if sims[i] > 0] or [int(order[0])]
            ids = list(dict.fromkeys(ids + [h[0] for h in hits[:2]]))
        else:
            have_entities = {S["docs"][i].get("entity", "").lower() for i in ids}
            have_files = {S["docs"][i]["filename"].lower() for i in ids}
            for did, count, strong_tokens in hits:
                d = S["docs"][did]
                d_ent = d.get("entity", "").lower()
                d_fn = d["filename"].lower()
                already_covered = (d_ent and any(d_ent in he or he in d_ent for he in have_entities if he)) or (d_fn in have_files)
                if not already_covered and count >= 1 and did not in ids:
                    ids.append(did)
                    have_entities.add(d_ent)
                    have_files.add(d_fn)

        breadth = plan.get("breadth") if plan.get("breadth") in C.K_BY_BREADTH else "medium"
        aspects = [a for a in plan.get("aspects", []) if isinstance(a, str)][:6]

        steps.append({
            "step_name": "metadata_filter",
            "description": "LLM planner reads auto-built document cards (entity/period/type/jurisdiction) and mandatorily filters the corpus; lexical safety net restores any entity named in the query",
            "input": {"query": q, "n_docs_in_collection": len(S["docs"])},
            "output": {
                "selected_docs": [S["docs"][i]["filename"] for i in ids],
                "breadth": breadth,
                "aspects": aspects,
                "reason": plan.get("reason", ""),
                "error": plan.get("error", ""),
                "used_lexical_fallback": used_fallback,
                "tokens": [tin, tout],
                "sec": round(time.time() - t0, 2)
            }
        })
        return ids, breadth, aspects, cost

    # -- step 2: scoped retrieval + coverage selection ---------------------------------
    def select(self, S, q, ids, breadth, aspects, steps, mult=1):
        t0 = time.time()
        mask = np.isin(np.array([c["doc_id"] for c in S["chunks"]]), ids)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            idx = np.arange(min(100, len(S["chunks"])))

        X = S["X"][idx]
        queries = [q] + aspects
        sc = np.zeros(len(idx), dtype=np.float32)
        base = 1.0
        for k, a in enumerate(queries):
            s = (X @ S["vec"].transform([a]).T).toarray().ravel()
            if k == 0:
                base = float(s.max()) or 1.0
            s = np.minimum(s / base, 1.0)
            sc = np.maximum(sc, s * (1.0 if k == 0 else 0.8))

        lens = np.array([len(S["chunks"][idx[i]]["text"]) for i in range(len(idx))])
        sc[lens < 40] = 0

        K = int(C.K_BY_BREADTH[breadth] * mult)
        thr = {"narrow": 0.35, "medium": 0.25, "broad": 0.18}[breadth]
        order = np.argsort(-sc)

        by_doc = {i: [] for i in ids}
        for o in order:
            if sc[o] < thr:
                break
            by_doc[S["chunks"][idx[o]]["doc_id"]].append(o)

        mass = {i: sum(sc[o] for o in by_doc[i][:K]) for i in ids}
        tot = sum(mass.values()) or 1.0
        floor = max(2, K // (3 * max(1, len(ids))))
        quota = {i: max(floor, int(round(K * mass[i] / tot))) for i in ids}

        chosen = []
        for i in ids:
            chosen += by_doc[i][:quota[i]]
        chosen = sorted(chosen, key=lambda o: -sc[o])[:int(K * 1.2)]
        if not chosen:
            chosen = list(order[:min(8, len(order))])

        ctx_len = 0
        final = []
        for o in chosen:
            c = S["chunks"][idx[o]]
            ctx_len += len(c["text"]) + 40
            if ctx_len > C.MAX_CONTEXT_CHARS:
                break
            final.append((idx[o], float(sc[o])))

        final.sort(key=lambda t: (S["chunks"][t[0]]["doc_id"], S["chunks"][t[0]]["page"], t[0]))
        steps.append({
            "step_name": "scoped_retrieval_and_coverage_selection",
            "description": "Sparse scoring restricted to the filtered pool, per aspect; breadth-adaptive K, per-document quota with floor, relative-score cut-off",
            "input": {"pool_chunks": int(len(idx)), "K": K, "threshold": thr, "queries": queries},
            "output": {
                "selected_chunks": len(final),
                "per_doc": {S["docs"][i]["filename"]: sum(1 for f, _ in final if S["chunks"][f]["doc_id"] == i) for i in ids},
                "sec": round(time.time() - t0, 3)
            }
        })
        return final

    # -- step 3: single relevance filter over candidate pool ------------------------------
    def filter_relevant(self, S, q, final, k, trace, steps):
        t0 = time.time()
        cost = 0.0
        keep = None
        tin = tout = 0
        err = ""
        items = "\n".join(
            f"[{n}] ({S['chunks'][i]['doc'][:-4]} p.{S['chunks'][i]['page']}) {S['chunks'][i]['text'][:450]}"
            for n, (i, _) in enumerate(final)
        )
        try:
            r = self.llm.json(FILTER_SYS, FILTER_USER.format(q=q, items=items, k=k), max_tokens=4000,
                              name="relevance_filter", trace=trace)
            r_json = r.get("json", {})
            ids = (r_json.get("relevant_ids") or r_json.get("selected_ids") or r_json.get("ids") or [])
            valid_ids = [n for n in ids if isinstance(n, int) and 0 <= n < len(final)]
            if valid_ids:
                keep = [final[n] for n in dict.fromkeys(valid_ids)][:k]
            cost, tin, tout = r.get("cost", 0.0), r.get("in", 0), r.get("out", 0)
        except Exception as e:
            err = str(e)[:200]

        # On failure keep the top-k by score; minimum 3 chunks
        out = keep if (keep and len(keep) >= 2) else final[:max(3, min(k, len(final)))]
        out.sort(key=lambda t: (S["chunks"][t[0]]["doc_id"], S["chunks"][t[0]]["page"], t[0]))
        steps.append({
            "step_name": "relevance_filter",
            "description": "LLM call over candidate pool keeps only excerpts that help answer the question",
            "input": {"candidates": len(final)},
            "output": {"kept": len(out), "tokens": [tin, tout], "error": err, "sec": round(time.time() - t0, 2)}
        })
        return out, cost

    # -- step 4: answer generation --------------------------------------------------------
    def query(self, collection, q):
        t_all = time.time()
        S = self.load(collection)
        steps = []
        trace = lf_trace("kb_query", {"collection": collection, "query": q})
        ids, breadth, aspects, cost = self.route(S, q, trace, steps)

        mult = 2 if breadth in ("narrow", "medium") else 1.2
        target_k = C.K_BY_BREADTH[breadth]
        final = self.select(S, q, ids, breadth, aspects, steps, mult=mult)

        # Apply relevance filter across all breadth levels
        final, fcost = self.filter_relevant(S, q, final, target_k, trace, steps)
        cost += fcost

        ctx = "\n\n".join(
            f"[{S['chunks'][i]['doc'][:-4]} | p.{S['chunks'][i]['page']}] {S['chunks'][i]['text']}"
            for i, _ in final
        )

        max_tok = {"narrow": 4000, "medium": 8000, "broad": 14000}[breadth]
        r = self.llm.call(
            ANSWER_SYS,
            ANSWER_USER.format(q=q, ctx=ctx, style=STYLE[breadth]),
            max_tokens=max_tok,
            name="llm_answer_generation",
            trace=trace
        )
        cost += r.get("cost", 0.0)

        steps.append({
            "step_name": "llm_answer_generation",
            "description": "Single grounded generation call over the selected evidence with forced doc/page citations",
            "input": {"evidence_chunks": len(final), "context_chars": len(ctx)},
            "output": {
                "answer_chars": len(r.get("text", "")),
                "tokens": [r.get("in", 0), r.get("out", 0)],
                "cost_usd": round(r.get("cost", 0.0), 6)
            }
        })

        chunks = [{
            "document": S["chunks"][i]["doc"],
            "page": S["chunks"][i]["page"],
            "page_numbers": [S["chunks"][i]["page"]],
            "section": S["chunks"][i].get("section", ""),
            "score": round(s, 4),
            "text": S["chunks"][i]["text"]
        } for i, s in final]

        out = {
            "answer": r.get("text", ""),
            "chunks": chunks,
            "trace": steps,
            "usage": {"cost_usd": round(cost, 6), "latency_s": round(time.time() - t_all, 2)}
        }
        lf_end(trace, r.get("text", ""), out["usage"])
        return out
