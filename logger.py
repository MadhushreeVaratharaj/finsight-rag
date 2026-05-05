import json
import time 
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st 
 
LOG_FILE = Path("query_log.jsonl")


def log_query(question, answer, latency_s, num_sources, source_pages, session_id, relevance_score=None):
    record = {
        "timestamp" : datetime.utcnow().isoformat(),
        "session_id" : session_id,
        "question" : question,
        "answer_length" : len(answer),
        "latency_s" : round(latency_s, 3),
        "num_sources" : num_sources,
        "source_pages" : source_pages,
        "relevance_score": round(relevance_score, 3) if relevance_score else None,
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

    if "query_log" not in st.session_state:
        st.session_state.query_log = []
    st.session_state.query_log.append(record)

    return record

def get_session_stats():
    log = st.session_state.get("query_log", [])
    if not log:
        return{}

    latencies = [r["latency_s"] for r in log]
    scores = [r["relevance_score"] for r in log if r.get("relevance_score")]
    return {
        "total_queries": len(log),
        "avg_latency_s": round(sum(latencies) / len(latencies), 2),
        "max_latency_s": round(max(latencies), 2),
        "min_latency_s": round(min(latencies), 2),
        "avg_sources": round(sum(r["num_sources"] for r in log) / len(log), 1),
        "avg_relevance": round(sum(scores) / len(scores), 3) if scores else None,


    }
