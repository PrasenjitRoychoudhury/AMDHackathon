"""AGENTS026 — Live Service Topology Graph (NetworkX + Matplotlib)"""
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import json, requests, time
from datetime import datetime, timezone
from pathlib import Path
from openai import OpenAI

st.set_page_config(page_title="AGENTS026 Topology", page_icon="🕸️", layout="wide")

# ── config ────────────────────────────────────────────────────────────────────
METRICS_CSV = Path("/workspace/shared/minicluster/live_metrics.csv")
AUDIT_FILE  = Path("/workspace/shared/audit_log.jsonl")
HITL_FILE   = Path("/workspace/shared/hitl_queue.jsonl")

SERVICES = {"payments": 7001, "auth": 7002, "checkout": 7003, "fraud": 7004}

THRESHOLDS = {
    "cpu_utilization": 70.0,
    "latency_p95_ms":  500.0,
    "error_rate":      0.05,
    "mem_mb":          800.0,
}

# Banking service call dependencies (directed graph)
# edge = (caller, callee, weight_label)
DEPENDENCIES = [
    ("checkout", "payments",  "payment_auth"),
    ("checkout", "auth",      "session_check"),
    ("checkout", "fraud",     "fraud_screen"),
    ("payments", "auth",      "token_verify"),
    ("fraud",    "payments",  "risk_signal"),
    ("auth",     "payments",  "auth_confirm"),
]

# vLLM client
try:
    llm = OpenAI(base_url="http://localhost:8000/v1", api_key="abc-123")
    LLM_AVAILABLE = True
except:
    LLM_AVAILABLE = False

# ── helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=10)
def load_metrics():
    if not METRICS_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(METRICS_CSV, parse_dates=["timestamp"])
    return df.tail(600)

def get_latest(df):
    if df.empty:
        return {}
    latest = df.sort_values("timestamp").groupby("service").last()
    return latest.to_dict("index")

def health_score(row):
    """0.0 (critical) to 1.0 (healthy)"""
    score = 1.0
    if row.get("cpu_utilization", 0)  > THRESHOLDS["cpu_utilization"]: score -= 0.3
    if row.get("latency_p95_ms", 0)   > THRESHOLDS["latency_p95_ms"]:  score -= 0.3
    if row.get("error_rate", 0)        > THRESHOLDS["error_rate"]:       score -= 0.3
    if row.get("mem_mb", 0)            > THRESHOLDS["mem_mb"]:           score -= 0.1
    return max(0.0, score)

def node_color(score):
    if score >= 0.8:  return "#00c853"   # green
    if score >= 0.5:  return "#ffd600"   # amber
    return "#ff1744"                      # red

def node_status(score):
    if score >= 0.8:  return "HEALTHY"
    if score >= 0.5:  return "DEGRADED"
    return "CRITICAL"

def fault_post(port, path, payload):
    try:
        r = requests.post(f"http://127.0.0.1:{port}{path}", json=payload, timeout=3)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def write_hitl(event):
    HITL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HITL_FILE, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")

def write_audit(event):
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_FILE, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")

def ts():
    return datetime.now(timezone.utc).isoformat()

# ── GPU anomaly narrative ──────────────────────────────────────────────────────
def gpu_anomaly_summary(latest_metrics, anomalies):
    if not LLM_AVAILABLE or not anomalies:
        return None
    try:
        prompt = f"""You are a banking SRE. Given these live service metrics and anomalies,
write a 2-sentence plain-English status summary for an operator dashboard.
Be specific about which services are affected and why.

Metrics: {json.dumps(latest_metrics, default=str)}
Anomalies: {json.dumps(anomalies, default=str)}

Respond with ONLY the 2-sentence summary, no JSON, no headers."""

        resp = llm.chat.completions.create(
            model="Qwen3-30B-A3B",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=120,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"GPU summary unavailable: {e}"

# ── draw topology graph ───────────────────────────────────────────────────────
def draw_topology(latest_metrics):
    G = nx.DiGraph()

    # Add nodes
    for svc in SERVICES:
        G.add_node(svc)

    # Add edges
    for src, dst, label in DEPENDENCIES:
        G.add_edge(src, dst, label=label)

    # Fixed positions (banking layout)
    pos = {
        "checkout": (0,    1.2),
        "auth":     (-1.5, 0),
        "payments": (0,    0),
        "fraud":    (1.5,  0),
    }

    # Node health
    scores  = {}
    colors  = {}
    sizes   = {}
    for svc in SERVICES:
        row    = latest_metrics.get(svc, {})
        score  = health_score(row)
        scores[svc] = score
        colors[svc] = node_color(score)
        # bigger node if anomalous
        sizes[svc]  = 4000 if score < 0.8 else 3000

    node_colors = [colors[n] for n in G.nodes()]
    node_sizes  = [sizes[n]  for n in G.nodes()]

    # Edge colors — red if either endpoint is degraded
    edge_colors = []
    edge_widths = []
    for src, dst in G.edges():
        if scores.get(src, 1) < 0.5 or scores.get(dst, 1) < 0.5:
            edge_colors.append("#ff1744")
            edge_widths.append(2.5)
        elif scores.get(src, 1) < 0.8 or scores.get(dst, 1) < 0.8:
            edge_colors.append("#ffd600")
            edge_widths.append(2.0)
        else:
            edge_colors.append("#00d4ff")
            edge_widths.append(1.5)

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("#0a0a1a")
    ax.set_facecolor("#0a0a1a")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.8, 2.0)
    ax.axis("off")

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_colors,
        width=edge_widths,
        arrows=True,
        arrowsize=20,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.1",
        min_source_margin=30,
        min_target_margin=30,
    )

    # Edge labels
    edge_labels = {(s, d): data["label"] for s, d, data in G.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, ax=ax,
        font_color="#aaaaaa", font_size=6,
        bbox=dict(boxstyle="round,pad=0.1", facecolor="#0a0a1a", alpha=0.6, edgecolor="none")
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
    )

    # Node labels with metrics
    for svc, (x, y) in pos.items():
        row   = latest_metrics.get(svc, {})
        score = scores[svc]
        color = colors[svc]
        status = node_status(score)

        # Service name
        ax.text(x, y + 0.08, svc.upper(), ha="center", va="center",
                fontsize=9, fontweight="bold", color="white",
                zorder=10)

        # Status badge
        ax.text(x, y - 0.08, status, ha="center", va="center",
                fontsize=7, color=color, fontweight="bold", zorder=10)

        # Metrics ring label
        cpu  = row.get("cpu_utilization", 0)
        lat  = row.get("latency_p95_ms", 0)
        err  = row.get("error_rate", 0)
        metrics_txt = f"CPU:{cpu:.0f}%  Lat:{lat:.0f}ms\nErr:{err:.3f}"
        ax.text(x, y - 0.35, metrics_txt, ha="center", va="center",
                fontsize=6.5, color="#cccccc", zorder=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e",
                          alpha=0.8, edgecolor=color, linewidth=1))

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#00c853", label="Healthy"),
        mpatches.Patch(facecolor="#ffd600", label="Degraded"),
        mpatches.Patch(facecolor="#ff1744", label="Critical"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              facecolor="#1a1a2e", labelcolor="white", fontsize=8, framealpha=0.8)

    ax.set_title("Live Service Dependency Graph — AGENTS026",
                 color="white", fontsize=12, pad=10)

    return fig, scores

# ══════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════════════════════════════════
st.title("🕸️ AGENTS026 — Live Service Topology")
st.caption("AMD Instinct MI300X · Qwen3-30B · Real-time service dependency graph")

col_r, col_a, col_l = st.columns([2, 2, 6])
with col_r:
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
with col_a:
    auto = st.toggle("Auto (15s)", value=False)
if auto:
    time.sleep(15)
    st.cache_data.clear()
    st.rerun()

st.divider()

df = load_metrics()
latest_metrics = get_latest(df)

# ── detect anomalies ──────────────────────────────────────────────────────────
anomalies = []
for svc, row in latest_metrics.items():
    for metric, thresh in THRESHOLDS.items():
        val = float(row.get(metric, 0))
        if val > thresh:
            anomalies.append({"service": svc, "metric": metric,
                               "value": round(val, 3), "threshold": thresh})

# ── layout: graph left, panel right ──────────────────────────────────────────
col_graph, col_panel = st.columns([3, 2])

with col_graph:
    if df.empty:
        st.warning("No metrics yet — waiting for collector...")
    else:
        fig, scores = draw_topology(latest_metrics)
        st.pyplot(fig)
        plt.close()
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')} | "
                   f"Nodes: {len(SERVICES)} | Edges: {len(DEPENDENCIES)}")

with col_panel:
    # ── service health cards ──────────────────────────────────────────────
    st.markdown("### 📊 Service Health")
    for svc in SERVICES:
        row   = latest_metrics.get(svc, {})
        score = health_score(row)
        icon  = "🟢" if score >= 0.8 else ("🟡" if score >= 0.5 else "🔴")
        with st.expander(f"{icon} {svc.upper()} — {node_status(score)}", expanded=(score < 0.8)):
            c1, c2 = st.columns(2)
            c1.metric("CPU %",      f"{row.get('cpu_utilization', 0):.1f}%")
            c1.metric("Latency",    f"{row.get('latency_p95_ms', 0):.0f}ms")
            c2.metric("Error rate", f"{row.get('error_rate', 0):.4f}")
            c2.metric("Memory",     f"{row.get('mem_mb', 0):.0f}MB")

    st.divider()

    # ── GPU anomaly narrative ─────────────────────────────────────────────
    st.markdown("### 🧠 GPU Insight (Qwen3-30B)")
    if anomalies:
        with st.spinner("Calling GPU for analysis..."):
            summary = gpu_anomaly_summary(latest_metrics, anomalies)
        if summary:
            st.info(summary)
        else:
            st.info("No GPU summary available.")
    else:
        st.success("✅ All services healthy — no GPU analysis needed.")

    st.divider()

    # ── fault injection quick panel ───────────────────────────────────────
    st.markdown("### 💥 Quick Fault Injection")
    fi_svc  = st.selectbox("Service", list(SERVICES.keys()), key="topo_svc")
    fi_port = SERVICES[fi_svc]

    fc1, fc2 = st.columns(2)
    with fc1:
        if st.button("🐢 +500ms latency"):
            r = fault_post(fi_port, "/fault/latency", {"ms": 500})
            write_audit({"event_type": "FAULT_INJECT", "service": fi_svc,
                         "fault": "latency", "ms": 500, "result": r, "timestamp": ts()})
            st.success(str(r))
        if st.button("💥 30% errors"):
            r = fault_post(fi_port, "/fault/errors", {"pct": 0.3})
            write_audit({"event_type": "FAULT_INJECT", "service": fi_svc,
                         "fault": "errors", "pct": 0.3, "result": r, "timestamp": ts()})
            st.success(str(r))
    with fc2:
        if st.button("🔥 CPU spike 60s"):
            r = fault_post(fi_port, "/fault/cpu_spin", {"seconds": 60})
            write_audit({"event_type": "FAULT_INJECT", "service": fi_svc,
                         "fault": "cpu_spin", "seconds": 60, "result": r, "timestamp": ts()})
            st.success(str(r))
        if st.button("🧹 Clear faults"):
            for port in SERVICES.values():
                fault_post(port, "/fault/clear", {})
            write_audit({"event_type": "FAULT_CLEAR_ALL", "timestamp": ts()})
            st.success("All faults cleared")

    st.divider()

    # ── HITL quick view ───────────────────────────────────────────────────
    st.markdown("### 🛑 HITL Pending")
    if HITL_FILE.exists():
        items = [json.loads(l) for l in HITL_FILE.read_text().strip().split("\n")
                 if l.strip()]
        pending = [x for x in items if x.get("status") == "PENDING"]
        if pending:
            for item in pending:
                st.error(f"⏳ {item.get('hitl_id')} — "
                         f"{item.get('rca',{}).get('action','?')} → "
                         f"{item.get('rca',{}).get('action_target','?')}")
                ca, cr = st.columns(2)
                with ca:
                    if st.button("✅ Approve", key=f"ta_{item['hitl_id']}"):
                        item["status"] = "APPROVED"
                        item["resolved_at"] = ts()
                        all_items = [x if x["hitl_id"] != item["hitl_id"]
                                     else item for x in items]
                        with open(HITL_FILE, "w") as f:
                            for rec in all_items:
                                f.write(json.dumps(rec, default=str) + "\n")
                        write_audit({**item, "event_type": "HITL_APPROVED"})
                        st.rerun()
                with cr:
                    if st.button("❌ Reject", key=f"tr_{item['hitl_id']}"):
                        item["status"] = "REJECTED"
                        item["resolved_at"] = ts()
                        all_items = [x if x["hitl_id"] != item["hitl_id"]
                                     else item for x in items]
                        with open(HITL_FILE, "w") as f:
                            for rec in all_items:
                                f.write(json.dumps(rec, default=str) + "\n")
                        write_audit({**item, "event_type": "HITL_REJECTED"})
                        st.rerun()
        else:
            st.success("✅ No pending HITL items")
    else:
        st.info("No HITL queue yet")
