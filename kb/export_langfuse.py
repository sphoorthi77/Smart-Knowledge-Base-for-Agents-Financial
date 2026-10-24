"""Generates a standalone Langfuse-format JSON export file from the benchmark execution logs."""
import json
import os
import argparse
from datetime import datetime

def generate_langfuse_export(results_file="dev_results.json", output_file="langfuse_export.json"):
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found. Run evaluation first.")
        return

    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    logs = data.get("log", [])
    if not logs:
        print("No log items found in results.")
        return

    traces = []
    total_cost = 0.0
    total_latency = 0.0

    for item in logs:
        qid = item.get("id", "Q")
        cost = float(item.get("cost_usd", 0.0))
        latency = float(item.get("latency_s", 0.0))
        n_chunks = item.get("n_chunks", 0)
        docs = item.get("docs", [])

        total_cost += cost
        total_latency += latency

        trace_entry = {
            "id": f"trace-{qid}-{int(datetime.utcnow().timestamp())}",
            "name": f"kb_query_{qid}",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "metadata": {
                "question_id": qid,
                "collection": "dev",
                "retrieved_chunks": n_chunks,
                "selected_docs": docs
            },
            "metrics": {
                "latency_seconds": round(latency, 2),
                "total_cost_usd": round(cost, 6)
            },
            "status": "SUCCESS"
        }
        traces.append(trace_entry)

    avg_cost = total_cost / len(logs)
    avg_latency = total_latency / len(logs)

    export_payload = {
        "export_metadata": {
            "project": "Smart Knowledge Base for Agents",
            "format": "langfuse_trace_export_v2",
            "exported_at": datetime.utcnow().isoformat() + "Z",
            "total_queries_evaluated": len(logs),
            "summary_metrics": {
                "total_cost_usd": round(total_cost, 6),
                "avg_cost_per_query_usd": round(avg_cost, 6),
                "avg_latency_per_query_seconds": round(avg_latency, 2)
            }
        },
        "traces": traces
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)

    print(f"Successfully generated {output_file} with {len(traces)} traces.")
    print(f"Summary: Avg Latency = {avg_latency:.2f}s | Avg Cost = ${avg_cost:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="dev_results.json")
    parser.add_argument("--out", default="langfuse_export.json")
    args = parser.parse_args()
    generate_langfuse_export(args.results, args.out)

