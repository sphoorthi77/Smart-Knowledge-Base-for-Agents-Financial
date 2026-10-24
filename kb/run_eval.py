"""Runs the engine on the training benchmark and scores it with the organizers' UNMODIFIED evaluator.
The evaluator classes are exec'd straight out of Evaluation.ipynb (code cells 'OpenCodeClient', 'RAGMetricsEvaluator',
'evaluate_dataset'); nothing in them is edited.
Usage:
  python run_eval.py --root /path/Train --collection dev [--n 100] [--concurrency 5] [--ids A01,B01]"""
import json, argparse, os, time, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--collection", default="dev")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--ids", default="")
ap.add_argument("--concurrency", type=int, default=5)
ap.add_argument("--skip-score", action="store_true")
ap.add_argument("--out", default="dev_results.json")
a = ap.parse_args()

bench_path = os.path.join(a.root, "Benchmarks", "benchmark_100.jsonl")
if not os.path.exists(bench_path):
    bench_path = os.path.join(a.root, "benchmark_100.jsonl")

with open(bench_path, "r", encoding="utf-8") as f:
    rows = [json.loads(l) for l in f if l.strip()]

if a.ids:
    wanted = set(a.ids.split(","))
    rows = [r for r in rows if r["id"] in wanted]
rows = rows[:a.n]

from app.config import OPENCODE_API_KEY
if not os.environ.get("OPENCODE_API_KEY") and OPENCODE_API_KEY:
    os.environ["OPENCODE_API_KEY"] = OPENCODE_API_KEY

from app.llm import LLM
from app.retrieve import Engine
eng = Engine(LLM())

print(f"Executing retrieval for {len(rows)} questions (concurrency={a.concurrency})...", flush=True)

def process_row(r):
    t0 = time.time()
    try:
        o = eng.query(a.collection, r["question"])
        gold = [c["content"] for v in r["benchmark_chunks_w_metadata"].values() for c in v]
        selected_docs = []
        for s in o.get("trace", []):
            if s.get("step_name") == "metadata_filter":
                selected_docs = s.get("output", {}).get("selected_docs", [])
                break
        item_data = {
            "query": r["question"],
            "ai_answer": o["answer"],
            "benchmark_answer": r["benchmark_answer"],
            "retrieved_chunks": [c["text"] for c in o["chunks"]],
            "benchmark_chunks": gold or ["n/a"]
        }
        item_log = {
            "id": r["id"],
            **o.get("usage", {}),
            "n_chunks": len(o.get("chunks", [])),
            "n_gold": len(gold),
            "docs": selected_docs,
            "gold_docs": [d for d, v in r["benchmark_chunks_w_metadata"].items() if v]
        }
        sec = time.time() - t0
        print(f"[{r['id']}] {sec:.1f}s | cost: ${item_log.get('cost_usd', 0):.5f} | chunks: {len(o.get('chunks', []))} | gold: {len(gold)}", flush=True)
        return r["id"], item_data, item_log
    except Exception as e:
        print(f"[{r['id']}] FAILED: {e}", flush=True)
        return r["id"], None, None

results_dict = {}
with ThreadPoolExecutor(max_workers=max(1, a.concurrency)) as pool:
    futures = [pool.submit(process_row, r) for r in rows]
    for fut in as_completed(futures):
        qid, item_data, item_log = fut.result()
        if item_data is not None:
            results_dict[qid] = (item_data, item_log)

# Sort strictly by the original row order
data = [results_dict[r["id"]][0] for r in rows if r["id"] in results_dict]
log = [results_dict[r["id"]][1] for r in rows if r["id"] in results_dict]

with open(a.out, "w", encoding="utf-8") as f:
    json.dump({"log": log}, f, indent=1)

if log:
    avg_cost = sum(l.get("cost_usd", 0) for l in log) / len(log)
    avg_lat = sum(l.get("latency_s", 0) for l in log) / len(log)
    doc_hit = [len(set(l["docs"]) & set(l["gold_docs"])) / max(1, len(l["gold_docs"])) for l in log if l["gold_docs"]]
    recall_str = f"{sum(doc_hit)/len(doc_hit):.2%}" if doc_hit else "n/a"
    print(f"\n--- RETRIEVAL SUMMARY ({len(log)}/{len(rows)} queries) ---", flush=True)
    print(f"Avg Cost per query: ${avg_cost:.6f} | Avg Latency: {avg_lat:.2f}s | Router Doc Recall: {recall_str}", flush=True)

# Update langfuse_export.json automatically
try:
    from export_langfuse import generate_langfuse_export
    generate_langfuse_export(a.out, "langfuse_export.json")
except Exception as e:
    print(f"Note: Langfuse export generation skipped: {e}")

if a.skip_score or not data:
    sys.exit()

# ---- load organizers' evaluator verbatim ----
eval_path = os.path.join(a.root, "Evaluation", "Evaluation.ipynb")
if not os.path.exists(eval_path):
    eval_path = os.path.join(a.root, "Evaluation.ipynb")

print(f"\nLoading unmodified evaluator from {eval_path}...", flush=True)
with open(eval_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

ns = {}
src = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]
for s in src:
    if s.lstrip().startswith(("import", "# ---- EDIT", "class ", "def evaluate_dataset")):
        s = s.replace('OPENCODE_API_KEY = ""', f'OPENCODE_API_KEY = os.environ["OPENCODE_API_KEY"]') if "EDIT THIS LINE" in s else s
        exec("import os\n" + s, ns)

print(f"Running unmodified evaluator across {len(data)} answers & chunks...", flush=True)
summary = ns["evaluate_dataset"](ns["RAGMetricsEvaluator"](ns["OpenCodeClient"](api_key=os.environ["OPENCODE_API_KEY"])), data, verbose=True)

print("\n" + "="*50, flush=True)
print("FINAL BENCHMARK EVALUATION METRICS:", flush=True)
for k, v in summary.items():
    if k != "per_question":
        if isinstance(v, float):
            print(f"  {k}: {v:.4f} ({v*100:.1f}%)", flush=True)
        else:
            print(f"  {k}: {v}", flush=True)
print("="*50 + "\n", flush=True)

with open("dev_scores.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=1, default=str)
print("Scores saved to dev_scores.json")
