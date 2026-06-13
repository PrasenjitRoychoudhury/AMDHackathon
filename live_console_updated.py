"""AGENTS026 — Live SRE Console (MiniCluster edition) — TONIGHT BUILD"""
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json, requests, time
from datetime import datetime, timezone
from pathlib import Path
from openai import OpenAI

st.set_page_config(page_title="AGENTS026 Live Console", page_icon="🖥️", layout="wide")

METRICS_CSV   = Path("/workspace/shared/minicluster/live_metrics.csv")
HITL_FILE     = Path("/workspace/shared/hitl_queue.jsonl")
AUDIT_FILE    = Path("/workspace/shared/audit_log.jsonl")
FORECAST_FILE = Path("/workspace/shared/uc3_forecasts.jsonl")
SILENCE_FILE  = Path("/workspace/shared/uc5_silence.jsonl")
SCORES_FILE   = Path("/workspace/shared/ae_scores.jsonl")
AE_LOSS_IMG   = Path("/workspace/shared/ae_loss_curve.png")
SCALER_PATH   = Path("/workspace/shared/autoencoder_scaler.json")
CHAOS_LOG     = Path("/workspace/shared/chaos_log.jsonl")
FAISS_META    = Path("/workspace/shared/faiss_incidents_meta.jsonl")
FAISS_RESULTS = Path("/workspace/shared/faiss_last_results.jsonl")
FAISS_PCA_IMG = Path("/workspace/shared/faiss_pca_plot.png")
UC4_RESULTS   = Path("/workspace/shared/uc4_alert_storm.jsonl")
JUDGE_RESULTS = Path("/workspace/shared/uc4_judge_results.jsonl")
UC4_GRAPH_IMG = Path("/workspace/shared/uc4_correlation_graph.png")
LOG_INDEX     = Path("/workspace/shared/faiss_logs.index")
LOG_META      = Path("/workspace/shared/faiss_logs_meta.jsonl")
LOG_PCA_IMG   = Path("/workspace/shared/log_embedding_pca.png")

SERVICES = {"payments": 7001, "auth": 7002, "checkout": 7003, "fraud": 7004}

THRESHOLDS = {
    "cpu_utilization": ("CPU %",       70.0,  "%"),
    "latency_p95_ms":  ("Latency p95", 500.0, "ms"),
    "error_rate":      ("Error rate",  0.05,  ""),
    "mem_mb":          ("Memory",      1800,  "MB"),
}

DEPENDENCIES = [
    ("checkout", "payments", "payment_auth"),
    ("checkout", "auth",     "session_check"),
    ("checkout", "fraud",    "fraud_screen"),
    ("payments", "auth",     "token_verify"),
    ("fraud",    "payments", "risk_signal"),
    ("auth",     "payments", "auth_confirm"),
]

try:
    llm = OpenAI(base_url="http://localhost:8000/v1", api_key="abc-123")
    LLM_AVAILABLE = True
except:
    LLM_AVAILABLE = False

@st.cache_data(ttl=8)
def load_metrics():
    if not METRICS_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(METRICS_CSV, parse_dates=["timestamp"])
    return df.tail(600)

def latest(df):
    if df.empty:
        return pd.DataFrame()
    return df.sort_values("timestamp").groupby("service").last().reset_index()

def get_latest_dict(df):
    if df.empty:
        return {}
    return df.sort_values("timestamp").groupby("service").last().to_dict("index")

def fault_post(port, path, payload):
    try:
        r = requests.post(f"http://127.0.0.1:{port}{path}", json=payload, timeout=3)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def clear_all_faults():
    return {svc: fault_post(port, "/fault/clear", {}) for svc, port in SERVICES.items()}

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

def health_score(row):
    score = 1.0
    if float(row.get("cpu_utilization", 0)) > 70.0:  score -= 0.3
    if float(row.get("latency_p95_ms",  0)) > 500.0: score -= 0.3
    if float(row.get("error_rate",      0)) > 0.05:  score -= 0.3
    if float(row.get("mem_mb",          0)) > 1800:  score -= 0.1
    return max(0.0, score)

def node_color(score):
    if score >= 0.8: return "#00c853"
    if score >= 0.5: return "#ffd600"
    return "#ff1744"

def node_status(score):
    if score >= 0.8: return "HEALTHY"
    if score >= 0.5: return "DEGRADED"
    return "CRITICAL"

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
            temperature=0.2, max_tokens=120,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"GPU summary unavailable: {e}"

def draw_topology(latest_metrics):
    G = nx.DiGraph()
    for svc in SERVICES:
        G.add_node(svc)
    for src, dst, label in DEPENDENCIES:
        G.add_edge(src, dst, label=label)

    pos = {"checkout": (0, 1.2), "auth": (-1.5, 0), "payments": (0, 0), "fraud": (1.5, 0)}

    scores = {}
    colors = {}
    sizes  = {}
    for svc in SERVICES:
        row = latest_metrics.get(svc, {})
        s = health_score(row)
        scores[svc] = s
        colors[svc] = node_color(s)
        sizes[svc]  = 4500 if s < 0.8 else 3200

    edge_colors, edge_widths = [], []
    for src, dst in G.edges():
        if scores.get(src, 1) < 0.5 or scores.get(dst, 1) < 0.5:
            edge_colors.append("#ff1744"); edge_widths.append(2.5)
        elif scores.get(src, 1) < 0.8 or scores.get(dst, 1) < 0.8:
            edge_colors.append("#ffd600"); edge_widths.append(2.0)
        else:
            edge_colors.append("#00d4ff"); edge_widths.append(1.5)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0a0a1a")
    ax.set_facecolor("#0a0a1a")
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-0.9, 2.0); ax.axis("off")

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths,
        arrows=True, arrowsize=20, arrowstyle="-|>",
        connectionstyle="arc3,rad=0.1", min_source_margin=35, min_target_margin=35)

    edge_labels = {(s, d): data["label"] for s, d, data in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
        font_color="#aaaaaa", font_size=6,
        bbox=dict(boxstyle="round,pad=0.1", facecolor="#0a0a1a", alpha=0.6, edgecolor="none"))

    nx.draw_networkx_nodes(G, pos, ax=ax,
        node_color=[colors[n] for n in G.nodes()],
        node_size=[sizes[n] for n in G.nodes()], alpha=0.9)

    for svc, (x, y) in pos.items():
        row = latest_metrics.get(svc, {})
        score = scores[svc]
        ax.text(x, y+0.10, svc.upper(), ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=10)
        ax.text(x, y-0.08, node_status(score), ha="center", va="center",
                fontsize=7, color=colors[svc], fontweight="bold", zorder=10)
        ax.text(x, y-0.38,
                f"CPU:{float(row.get('cpu_utilization',0)):.0f}%  Lat:{float(row.get('latency_p95_ms',0)):.0f}ms\nErr:{float(row.get('error_rate',0)):.3f}",
                ha="center", va="center", fontsize=6.5, color="#cccccc", zorder=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e",
                          alpha=0.8, edgecolor=colors[svc], linewidth=1))

    ax.legend(handles=[
        mpatches.Patch(facecolor="#00c853", label="Healthy"),
        mpatches.Patch(facecolor="#ffd600", label="Degraded"),
        mpatches.Patch(facecolor="#ff1744", label="Critical"),
    ], loc="lower right", facecolor="#1a1a2e", labelcolor="white", fontsize=8, framealpha=0.8)
    ax.set_title("Live Service Dependency Graph — AGENTS026", color="white", fontsize=12, pad=10)
    return fig, scores

# ── header ────────────────────────────────────────────────────────────────────
st.title("🖥️ AGENTS026 — Live SRE Console")
st.caption("AMD Instinct MI300X · Qwen3-30B via vLLM · MiniCluster banking stack")

col_ref, col_auto = st.columns([6, 1])
with col_ref:
    if st.button("🔄 Refresh now"):
        st.cache_data.clear()
with col_auto:
    auto = st.toggle("Auto-refresh (10s)", value=False)
if auto:
    time.sleep(10)
    st.cache_data.clear()
    st.rerun()

df  = load_metrics()
lat = latest(df)

t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14 = st.tabs([
    "❤️ Health", "📈 Live Telemetry", "⚠️ Anomalies",
    "💥 Fault Injection", "🤖 AI Actions", "🛑 HITL Queue",
    "🕸️ Topology", "📈 Trend Forecast", "🔇 Silence", "🤖 AE Detector",
    "🔥 Chaos Scheduler", "🔎 FAISS Search", "⛈️ Alert Storm", "📜 Log Embedding"
])

# ══════════════════════════════════════════════════════════════════════════════
# TABS 1–10 — unchanged from original
# ══════════════════════════════════════════════════════════════════════════════

with t1:
    st.subheader("Current cluster health")
    if lat.empty:
        st.warning("No metrics yet — waiting for collector...")
    else:
        cols = st.columns(len(lat))
        for i, row in lat.iterrows():
            svc  = row["service"]
            cpu  = float(row["cpu_utilization"])
            lat_ = float(row["latency_p95_ms"])
            err  = float(row["error_rate"])
            mem  = float(row["mem_mb"])
            bad  = (cpu > 70) or (lat_ > 500) or (err > 0.05) or (mem > 1800)
            icon = "🔴" if bad else "🟢"
            with cols[i % len(cols)]:
                st.metric(f"{icon} {svc.upper()}", "")
                st.metric("CPU %",       f"{cpu:.1f}%")
                st.metric("Latency p95", f"{lat_:.0f}ms")
                st.metric("Error rate",  f"{err:.4f}")
                st.metric("Memory MB",   f"{mem:.0f}")
        any_bad = any(
            float(r["cpu_utilization"]) > 70 or float(r["latency_p95_ms"]) > 500 or
            float(r["error_rate"]) > 0.05    or float(r["mem_mb"]) > 1800
            for _, r in lat.iterrows()
        )
        st.divider()
        st.error("⚠️  One or more services breaching thresholds") if any_bad else st.success("✅  All services within normal thresholds")
        st.caption(f"Last updated: {lat['timestamp'].max()}")

with t2:
    st.subheader("Live telemetry — last 30 minutes")
    if df.empty:
        st.warning("No data yet.")
    else:
        recent        = df[df["timestamp"] >= df["timestamp"].max() - pd.Timedelta("30min")]
        metric_choice = st.selectbox("Metric", ["latency_p95_ms","cpu_utilization","error_rate","mem_mb"], key="met")
        svc_filter    = st.multiselect("Services", df["service"].unique().tolist(),
                                       default=df["service"].unique().tolist(), key="svc")
        filtered = recent[recent["service"].isin(svc_filter)]
        fig, ax = plt.subplots(figsize=(10, 3))
        fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white"); ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
        for spine in ax.spines.values(): spine.set_edgecolor("#444")
        clrs = ["#00d4ff","#ff6b6b","#51cf66","#ffd43b","#cc5de8","#ff922b"]
        for idx2, svc in enumerate(svc_filter):
            s = filtered[filtered["service"] == svc].sort_values("timestamp")
            ax.plot(s["timestamp"], s[metric_choice], label=svc, color=clrs[idx2 % len(clrs)], linewidth=1.8)
        thresh_map = {"cpu_utilization": 70, "latency_p95_ms": 500, "error_rate": 0.05, "mem_mb": 1800}
        if metric_choice in thresh_map:
            ax.axhline(thresh_map[metric_choice], color="#ff4444", linestyle="--", linewidth=1, label="threshold")
        ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
        ax.set_xlabel("Time", color="white"); ax.set_ylabel(metric_choice, color="white")
        st.pyplot(fig); plt.close()
        st.dataframe(filtered.sort_values("timestamp", ascending=False).head(40), use_container_width=True)

with t3:
    st.subheader("Threshold-based anomaly detection")
    if lat.empty:
        st.warning("No data yet.")
    else:
        anomalies = []
        for _, row in lat.iterrows():
            for col, (label, thresh, unit) in THRESHOLDS.items():
                val = float(row[col])
                if val > thresh:
                    anomalies.append({"service": row["service"], "metric": label,
                        "value": f"{val:.4f} {unit}", "threshold": f"{thresh} {unit}",
                        "severity": "🔴 HIGH" if val > thresh * 1.5 else "🟡 WARN",
                        "node": row.get("node","?"), "detected": str(row["timestamp"])})
        if anomalies:
            st.error(f"⚠️  {len(anomalies)} anomaly/-ies detected")
            adf = pd.DataFrame(anomalies)
            st.dataframe(adf, use_container_width=True)
            st.divider()
            st.markdown("**Escalate to HITL queue**")
            sel_svc = st.selectbox("Select service to escalate", adf["service"].unique().tolist(), key="esc_svc")
            if st.button("📤 Send to HITL queue"):
                evt = {"hitl_id": f"hitl-{int(time.time())}", "timestamp": ts(),
                       "source": "anomaly_detector", "service": sel_svc,
                       "anomalies": [a for a in anomalies if a["service"] == sel_svc], "status": "PENDING"}
                write_hitl(evt); write_audit({**evt, "event_type": "HITL_CREATED"})
                st.success(f"Sent {sel_svc} anomalies to HITL queue ✅")
        else:
            st.success("✅  No anomalies — all metrics within threshold")

with t4:
    st.subheader("Fault injection control panel")
    st.warning("⚠️  These faults hit the live MiniCluster. Clear when done.")
    target = st.selectbox("Target service", list(SERVICES.keys()), key="fi_svc")
    port   = SERVICES[target]
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🐢 Latency spike")
        ms = st.slider("Latency (ms)", 100, 2000, 500, 100, key="lat_ms")
        if st.button(f"Inject {ms}ms latency → {target}"):
            r = fault_post(port, "/fault/latency", {"ms": ms})
            write_audit({"event_type":"FAULT_INJECT","service":target,"fault":"latency","ms":ms,"result":r,"timestamp":ts()})
            st.success(f"✅ {r}")
        st.markdown("#### 💥 Error rate")
        pct = st.slider("Error %", 5, 80, 30, 5, key="err_pct")
        if st.button(f"Inject {pct}% errors → {target}"):
            r = fault_post(port, "/fault/errors", {"pct": pct/100})
            write_audit({"event_type":"FAULT_INJECT","service":target,"fault":"errors","pct":pct/100,"result":r,"timestamp":ts()})
            st.success(f"✅ {r}")
    with col2:
        st.markdown("#### 🔥 CPU spike")
        secs = st.slider("Duration (s)", 10, 120, 30, 10, key="cpu_sec")
        if st.button(f"Inject CPU spin {secs}s → {target}"):
            r = fault_post(port, "/fault/cpu_spin", {"seconds": secs})
            write_audit({"event_type":"FAULT_INJECT","service":target,"fault":"cpu_spin","seconds":secs,"result":r,"timestamp":ts()})
            st.success(f"✅ {r}")
        st.markdown("#### 🧠 Memory leak")
        mb = st.slider("MB/min", 10, 200, 50, 10, key="mem_mb")
        if st.button(f"Inject mem leak {mb}MB/min → {target}"):
            r = fault_post(port, "/fault/mem_leak", {"mb_per_min": mb})
            write_audit({"event_type":"FAULT_INJECT","service":target,"fault":"mem_leak","mb_per_min":mb,"result":r,"timestamp":ts()})
            st.success(f"✅ {r}")
    st.divider()
    if st.button("🧹 CLEAR ALL FAULTS (all services)", type="primary"):
        results = clear_all_faults()
        write_audit({"event_type":"FAULT_CLEAR_ALL","results":results,"timestamp":ts()})
        st.success(f"All faults cleared: {results}")

with t5:
    st.subheader("AI-decided actions log")
    st.info("Actions requiring approval appear in the HITL Queue tab.")
    if AUDIT_FILE.exists():
        records = []
        with open(AUDIT_FILE) as f:
            for line in f:
                try: records.append(json.loads(line))
                except: pass
        if records:
            adf = pd.DataFrame(records)
            adf = adf.sort_values("timestamp", ascending=False) if "timestamp" in adf.columns else adf
            st.dataframe(adf, use_container_width=True)
            if "event_type" in adf.columns:
                st.divider(); st.markdown("**Event breakdown**")
                st.bar_chart(adf["event_type"].value_counts())
        else:
            st.info("No AI actions recorded yet.")
    else:
        st.info("Audit log not yet created.")

with t6:
    st.subheader("Human-in-the-Loop approval queue")
    if not HITL_FILE.exists() or HITL_FILE.stat().st_size == 0:
        st.info("No pending HITL items.")
    else:
        items = []
        with open(HITL_FILE) as f:
            for line in f:
                try: items.append(json.loads(line))
                except: pass
        pending  = [x for x in items if x.get("status") == "PENDING"]
        resolved = [x for x in items if x.get("status") != "PENDING"]
        st.metric("Pending approvals", len(pending))
        st.metric("Resolved", len(resolved))
        if pending:
            st.divider(); st.markdown("### ⏳ Pending items")
            for item in pending:
                with st.expander(f"🔴 {item.get('hitl_id')} — {item.get('service','rca')} — {item.get('timestamp','')}"):
                    st.json(item)
                    col_a, col_r = st.columns(2)
                    with col_a:
                        if st.button("✅ Approve", key=f"appr_{item['hitl_id']}"):
                            item.update({"status":"APPROVED","resolved_at":ts(),"operator":"prasenjit.roychoudhury"})
                            all_items = [x if x["hitl_id"] != item["hitl_id"] else item for x in items]
                            with open(HITL_FILE,"w") as f:
                                for rec in all_items: f.write(json.dumps(rec,default=str)+"\n")
                            write_audit({**item,"event_type":"HITL_APPROVED"})
                            st.success("Approved."); st.rerun()
                    with col_r:
                        if st.button("❌ Reject", key=f"rej_{item['hitl_id']}"):
                            item.update({"status":"REJECTED","resolved_at":ts(),"operator":"prasenjit.roychoudhury"})
                            all_items = [x if x["hitl_id"] != item["hitl_id"] else item for x in items]
                            with open(HITL_FILE,"w") as f:
                                for rec in all_items: f.write(json.dumps(rec,default=str)+"\n")
                            write_audit({**item,"event_type":"HITL_REJECTED"})
                            st.warning("Rejected."); st.rerun()
        if resolved:
            st.divider(); st.markdown("### ✅ Resolved items")
            st.dataframe(pd.DataFrame(resolved), use_container_width=True)

with t7:
    st.subheader("Live Service Dependency Graph")
    st.caption("🟢 Healthy  🟡 Degraded  🔴 Critical · Edge colour follows call-path health · GPU insight powered by Qwen3-30B")
    latest_metrics = get_latest_dict(df)
    topo_anomalies = []
    for svc, row in latest_metrics.items():
        for col, (label, thresh, unit) in THRESHOLDS.items():
            val = float(row.get(col, 0))
            if val > thresh:
                topo_anomalies.append({"service": svc, "metric": label,
                                       "value": round(val,3), "threshold": thresh})
    col_graph, col_panel = st.columns([3, 2])
    with col_graph:
        if df.empty:
            st.warning("No metrics yet.")
        else:
            fig, scores = draw_topology(latest_metrics)
            st.pyplot(fig); plt.close()
            st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')} · {len(SERVICES)} nodes · {len(DEPENDENCIES)} edges")
    with col_panel:
        st.markdown("### 📊 Service Health")
        for svc in SERVICES:
            row   = latest_metrics.get(svc, {})
            score = health_score(row)
            icon  = "🟢" if score >= 0.8 else ("🟡" if score >= 0.5 else "🔴")
            with st.expander(f"{icon} {svc.upper()} — {node_status(score)}", expanded=(score < 0.8)):
                c1, c2 = st.columns(2)
                c1.metric("CPU %",      f"{float(row.get('cpu_utilization',0)):.1f}%")
                c1.metric("Latency",    f"{float(row.get('latency_p95_ms',0)):.0f}ms")
                c2.metric("Error rate", f"{float(row.get('error_rate',0)):.4f}")
                c2.metric("Memory",     f"{float(row.get('mem_mb',0)):.0f}MB")
        st.divider()
        st.markdown("### 🧠 GPU Insight (Qwen3-30B)")
        if topo_anomalies:
            with st.spinner("Calling GPU..."):
                summary = gpu_anomaly_summary(latest_metrics, topo_anomalies)
            st.info(summary or "No summary returned.")
        else:
            st.success("✅ All services healthy — no GPU analysis needed.")
        st.divider()
        st.markdown("### 🛑 HITL Pending")
        if HITL_FILE.exists():
            hitl_items = [json.loads(l) for l in HITL_FILE.read_text().strip().split("\n") if l.strip()]
            pending_t7 = [x for x in hitl_items if x.get("status") == "PENDING"]
            if pending_t7:
                for item in pending_t7:
                    st.error(f"⏳ {item.get('hitl_id')} — {item.get('rca',{}).get('action','?')} → {item.get('rca',{}).get('action_target','?')}")
                    ca, cr = st.columns(2)
                    with ca:
                        if st.button("✅ Approve", key=f"t7a_{item['hitl_id']}"):
                            item.update({"status":"APPROVED","resolved_at":ts()})
                            all_i = [x if x["hitl_id"]!=item["hitl_id"] else item for x in hitl_items]
                            with open(HITL_FILE,"w") as f:
                                for rec in all_i: f.write(json.dumps(rec,default=str)+"\n")
                            write_audit({**item,"event_type":"HITL_APPROVED"}); st.rerun()
                    with cr:
                        if st.button("❌ Reject", key=f"t7r_{item['hitl_id']}"):
                            item.update({"status":"REJECTED","resolved_at":ts()})
                            all_i = [x if x["hitl_id"]!=item["hitl_id"] else item for x in hitl_items]
                            with open(HITL_FILE,"w") as f:
                                for rec in all_i: f.write(json.dumps(rec,default=str)+"\n")
                            write_audit({**item,"event_type":"HITL_REJECTED"}); st.rerun()
            else:
                st.success("✅ No pending HITL items")
        else:
            st.info("No HITL queue yet")

with t8:
    st.subheader("UC-3 — Trend Forecasting Anomaly Detection")
    st.caption("Linear regression on last 15min → 5min ahead projection · 🧠 GPU advisory on breach trends")
    col_info, col_run = st.columns([5, 1])
    with col_info:
        st.info("Forecasts where each service metric will be in 5 minutes. 🔴 BREACH = projected to exceed threshold. 🟡 APPROACHING = >80% of threshold and rising.")
    with col_run:
        run_now = st.button("▶️ Run Forecast")
    st.divider()
    if FORECAST_FILE.exists() and FORECAST_FILE.stat().st_size > 0:
        recs = [json.loads(l) for l in FORECAST_FILE.read_text().strip().split("\n") if l.strip()]
        fdf  = pd.DataFrame(recs)
        breach = fdf[fdf["severity"] == "BREACH"]
        approx = fdf[fdf["severity"] == "APPROACHING"]
        ok     = fdf[fdf["severity"] == "OK"]
        c1, c2, c3 = st.columns(3)
        c1.metric("🔴 BREACH forecasts",     len(breach))
        c2.metric("🟡 APPROACHING forecasts", len(approx))
        c3.metric("✅ OK",                   len(ok))
        if not breach.empty or not approx.empty:
            st.divider()
            st.markdown("### ⚠️ Alert Forecasts")
            alert_df = pd.concat([breach, approx]).sort_values("pct_of_thresh", ascending=False)
            for _, row in alert_df.iterrows():
                icon = "🔴" if row["severity"] == "BREACH" else "🟡"
                pct  = row["pct_of_thresh"] * 100
                st.markdown(f"{icon} **{row['service'].upper()}.{row['metric']}** — now `{row['current']:.3f}` → projected `{row['projected']:.3f}` (thresh `{row['threshold']}`) · R²=`{row['r2']:.2f}` · {row['trend']}")
                st.progress(min(pct / 100, 1.0), text=f"{pct:.0f}% of threshold")
        st.divider()
        st.markdown("### 📊 Full Forecast Table")
        def sev_icon(s): return "🔴" if s == "BREACH" else ("🟡" if s == "APPROACHING" else "✅")
        fdf["status"] = fdf["severity"].apply(sev_icon)
        display_cols = ["status","service","metric","current","projected","threshold","pct_of_thresh","r2","trend","delta"]
        st.dataframe(fdf[[c for c in display_cols if c in fdf.columns]].sort_values("pct_of_thresh", ascending=False), use_container_width=True)
        st.divider()
        st.markdown("### 📉 Projection Comparison by Metric")
        metric_sel = st.selectbox("Metric", list(THRESHOLDS.keys()), key="uc3_metric")
        metric_df  = fdf[fdf["metric"] == metric_sel]
        if not metric_df.empty:
            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
            ax.tick_params(colors="white")
            for spine in ax.spines.values(): spine.set_edgecolor("#444")
            x = np.arange(len(metric_df)); width = 0.35
            svcs  = metric_df["service"].tolist()
            thresh2 = THRESHOLDS[metric_sel][1]
            ax.bar(x - width/2, metric_df["current"], width, label="Current", color="#00d4ff", alpha=0.8)
            ax.bar(x + width/2, metric_df["projected"], width, label="Projected",
                   color=[("#ff1744" if p > thresh2 else "#ffd600") for p in metric_df["projected"]], alpha=0.8)
            ax.axhline(thresh2, color="#ff4444", linestyle="--", linewidth=1.5, label=f"Threshold ({thresh2})")
            ax.set_xticks(x); ax.set_xticklabels(svcs, color="white")
            ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
            ax.set_title(f"{metric_sel} — Current vs 5min Forecast", color="white", fontsize=10)
            st.pyplot(fig); plt.close()
        st.divider()
        st.markdown("### 🧠 Latest GPU Advisory")
        if AUDIT_FILE.exists():
            audit_recs = [json.loads(l) for l in AUDIT_FILE.read_text().strip().split("\n") if l.strip()]
            uc3_audits = [r for r in audit_recs if r.get("event_type") == "UC3_FORECAST_ALERTS" and r.get("narrative")]
            if uc3_audits:
                latest_advisory = sorted(uc3_audits, key=lambda x: x.get("timestamp",""))[-1]
                st.warning(latest_advisory["narrative"])
                st.caption(f"Generated at {latest_advisory.get('timestamp','')}")
            else:
                st.info("No GPU advisory yet — run the UC-3 forecast notebook cell with active alerts.")
        else:
            st.info("No audit log yet.")
    else:
        st.info("No forecast data yet. Run **Cell 11** in the RCA notebook to generate forecasts.")

with t9:
    st.subheader("UC-5 — Silent Failure / Absence of Signal Detection")
    st.caption("Detects services that stopped reporting, flatlined, or dropped to zero traffic · 🧠 GPU reasoning on findings")
    st.info("**Three detection modes:** 🔴 **Gap** — service not reported in >3min. 🟡 **Flatline** — metric std dev ≈ 0. 🟡 **Zero-RPS** — service alive but serving no traffic.")
    st.divider()
    st.markdown("### ⚡ Live Gap Check (from metrics)")
    if df.empty:
        st.warning("No metrics loaded.")
    else:
        now_ts = df["timestamp"].max()
        gap_rows = []
        for svc in ["payments", "auth", "checkout", "fraud"]:
            svc_df = df[df["service"] == svc]
            if svc_df.empty:
                gap_rows.append({"service": svc, "last_seen": "NEVER", "gap_mins": "∞", "status": "🔴 MISSING"})
            else:
                last_ts  = svc_df["timestamp"].max()
                gap_mins = (now_ts - last_ts).total_seconds() / 60
                status   = "🔴 SILENT" if gap_mins > 3 else ("🟡 DELAYED" if gap_mins > 1.5 else "✅ OK")
                gap_rows.append({"service": svc, "last_seen": str(last_ts), "gap_mins": round(gap_mins, 2), "status": status})
        st.dataframe(pd.DataFrame(gap_rows), use_container_width=True)
        silent_svcs = [r["service"] for r in gap_rows if "SILENT" in r["status"] or "MISSING" in r["status"]]
        if silent_svcs:
            st.error(f"🔴 Silent services detected: {', '.join(silent_svcs)}")
        else:
            st.success("✅ All services reporting within expected interval")
    st.divider()
    st.markdown("### 📋 Full Silence Analysis (from notebook UC-5)")
    if SILENCE_FILE.exists() and SILENCE_FILE.stat().st_size > 0:
        findings = [json.loads(l) for l in SILENCE_FILE.read_text().strip().split("\n") if l.strip()]
        fdf2 = pd.DataFrame(findings)
        critical2 = [f for f in findings if f["severity"] == "CRITICAL"]
        high2     = [f for f in findings if f["severity"] == "HIGH"]
        c1, c2, c3 = st.columns(3)
        c1.metric("🔴 CRITICAL", len(critical2)); c2.metric("🟡 HIGH", len(high2)); c3.metric("Total", len(findings))
        for f in findings:
            icon = "🔴" if f["severity"] == "CRITICAL" else "🟡"
            with st.expander(f"{icon} {f['service'].upper()} — {f.get('type','?')} — {f['severity']}"):
                st.markdown(f"**Detail:** {f.get('detail','')}")
                st.caption(f"Detected: {f.get('timestamp','')}")
        st.dataframe(fdf2, use_container_width=True)
    else:
        st.info("No silence analysis yet. Run **Cell 15** (UC-5) in the notebook.")

with t10:
    st.subheader("🤖 GPU Autoencoder Anomaly Detection")
    st.caption("PyTorch autoencoder trained on MI300X · Reconstruction error flags subtle anomalies · Qwen3-30B RCA on triggers")
    st.info("**Why this catches what thresholds miss:** The autoencoder learns the *correlation pattern* between metrics during healthy operation. An anomaly score > 1.0 means the current metric combination is unlike anything in the baseline.")
    st.divider()
    st.markdown("### 🏋️ Model Status")
    mc1, mc2, mc3 = st.columns(3)
    model_exists  = Path("/workspace/shared/autoencoder_model.pt").exists()
    scaler_exists = SCALER_PATH.exists()
    mc1.metric("Model file",   "✅ Trained" if model_exists  else "❌ Not trained")
    mc2.metric("Scaler file",  "✅ Ready"   if scaler_exists else "❌ Missing")
    if scaler_exists:
        try:
            with open(SCALER_PATH) as f: sc = json.load(f)
            thresh = sc.get("anomaly_threshold", "N/A")
            mc3.metric("Anomaly threshold", f"{thresh:.6f}" if isinstance(thresh, float) else thresh)
            st.caption(f"Features: {sc.get('cols', [])}")
        except:
            mc3.metric("Anomaly threshold", "Parse error")
    else:
        mc3.metric("Anomaly threshold", "Not set")
    if AE_LOSS_IMG.exists():
        st.divider(); st.markdown("### 📉 Training Loss Curve (GPU)")
        st.image(str(AE_LOSS_IMG), caption="Autoencoder MSE loss over epochs — trained on AMD MI300X")
    st.divider()
    st.markdown("### 🔍 Live Reconstruction Error Scores")
    if SCORES_FILE.exists() and SCORES_FILE.stat().st_size > 0:
        scores2 = [json.loads(l) for l in SCORES_FILE.read_text().strip().split("\n") if l.strip()]
        sdf = pd.DataFrame(scores2)
        sdf["timestamp"] = pd.to_datetime(sdf["timestamp"]); sdf = sdf.sort_values("timestamp")
        n_total = len(sdf); n_anom = sdf["is_anomaly"].sum()
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Total timesteps", n_total); sc2.metric("🔴 Anomalies", int(n_anom))
        sc3.metric("🟡 Warnings", int((sdf["severity"]=="WARN").sum())); sc4.metric("✅ Normal", int((sdf["severity"]=="OK").sum()))
        fig, ax = plt.subplots(figsize=(11, 3))
        fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white"); ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
        for spine in ax.spines.values(): spine.set_edgecolor("#444")
        anom_df2 = sdf[sdf["is_anomaly"]]
        ax.plot(sdf["timestamp"], sdf["recon_error"], color="#00d4ff", linewidth=1.2, alpha=0.7, label="Reconstruction error")
        if not anom_df2.empty:
            ax.scatter(anom_df2["timestamp"], anom_df2["recon_error"], color="#ff1744", s=60, zorder=5, label="🔴 Anomaly")
        ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
        ax.set_xlabel("Time", color="white"); ax.set_ylabel("Reconstruction Error", color="white")
        ax.set_title("Autoencoder Reconstruction Error — Live Metrics", color="white", fontsize=10)
        st.pyplot(fig); plt.close()
        st.dataframe(sdf[["timestamp","recon_error","anomaly_score","severity","is_anomaly"]].sort_values("timestamp", ascending=False).head(50), use_container_width=True)
    else:
        st.info("No AE scores yet. Run Cell 7 (live scoring) in `ae_anomaly_detector.ipynb`.")
    st.divider()
    ae_svc = st.selectbox("Service", list(SERVICES.keys()), key="ae_svc")
    ae_port = SERVICES[ae_svc]
    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        if st.button("🐢 500ms latency", key="ae_lat"):
            r = fault_post(ae_port, "/fault/latency", {"ms": 500})
            write_audit({"event_type":"FAULT_INJECT","service":ae_svc,"fault":"latency","timestamp":ts()}); st.success(str(r))
    with ac2:
        if st.button("💥 30% errors", key="ae_err"):
            r = fault_post(ae_port, "/fault/errors", {"pct": 0.3})
            write_audit({"event_type":"FAULT_INJECT","service":ae_svc,"fault":"errors","timestamp":ts()}); st.success(str(r))
    with ac3:
        if st.button("🧹 Clear all", key="ae_clear"):
            clear_all_faults(); write_audit({"event_type":"FAULT_CLEAR_ALL","timestamp":ts()}); st.success("Cleared")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 11 — CHAOS SCHEDULER
# ══════════════════════════════════════════════════════════════════════════════
with t11:
    st.subheader("🔥 Chaos Scheduler — UC-6")
    st.caption("GPU-selected fault scenarios · autonomous inject → monitor → clear loop · HITL gate on CRITICAL blasts")
    st.info("**How it works:** Qwen3-30B reads live metrics and selects the most impactful chaos scenario. Faults fire automatically, metrics snapshotted pre/post, GPU writes a post-mortem. CRITICAL blast-radius scenarios require HITL approval.")

    CHAOS_SCENARIOS = [
        {"id":"SC-01","name":"Payment Latency Spike",     "blast":"medium",   "faults":1,"duration":90},
        {"id":"SC-02","name":"Auth Cascade Failure",      "blast":"high",     "faults":2,"duration":90},
        {"id":"SC-03","name":"Fraud CPU Saturation",      "blast":"low",      "faults":1,"duration":90},
        {"id":"SC-04","name":"Memory Leak + Errors",      "blast":"medium",   "faults":2,"duration":90},
        {"id":"SC-05","name":"Alert Storm — All Services","blast":"critical", "faults":4,"duration":120},
        {"id":"SC-06","name":"Silent Fraud Drop",         "blast":"low",      "faults":1,"duration":90},
    ]
    BLAST_ICON = {"low":"🟢","medium":"🟡","high":"🔴","critical":"💀"}

    st.markdown("### 📋 Scenario Library")
    cols_hdr = st.columns([1, 3, 2, 1, 1])
    for h, txt in zip(cols_hdr, ["**ID**","**Name**","**Blast Radius**","**Faults**","**Duration**"]):
        h.markdown(txt)
    for sc in CHAOS_SCENARIOS:
        c0,c1,c2,c3,c4 = st.columns([1,3,2,1,1])
        c0.code(sc["id"]); c1.write(sc["name"])
        c2.write(f"{BLAST_ICON[sc['blast']]} {sc['blast'].upper()}")
        c3.write(str(sc["faults"])); c4.write(f"{sc['duration']}s")

    st.divider()
    st.markdown("### ▶️ Run a Scenario")
    chosen_id = st.selectbox("Select scenario", [f"{s['id']} — {s['name']}" for s in CHAOS_SCENARIOS], key="chaos_sel")
    selected_sc = next(s for s in CHAOS_SCENARIOS if chosen_id.startswith(s["id"]))
    if selected_sc["blast"] == "critical":
        st.error("💀 CRITICAL blast radius — HITL approval required before execution")
    elif selected_sc["blast"] == "high":
        st.warning("🔴 HIGH blast radius — all services may be affected")

    QUICK_FAULTS = {
        "SC-01": [("payments", "/fault/latency", {"ms": 800})],
        "SC-02": [("auth", "/fault/errors", {"pct": 0.40}), ("auth", "/fault/latency", {"ms": 600})],
        "SC-03": [("fraud", "/fault/cpu_spin", {"seconds": 80})],
        "SC-04": [("checkout", "/fault/mem_leak", {"mb_per_min": 80}), ("checkout", "/fault/errors", {"pct": 0.25})],
        "SC-05": [("payments","/fault/latency",{"ms":700}),("auth","/fault/errors",{"pct":0.30}),("checkout","/fault/latency",{"ms":500}),("fraud","/fault/errors",{"pct":0.20})],
        "SC-06": [("fraud", "/fault/latency", {"ms": 3000})],
    }

    c_fire, c_clear = st.columns(2)
    with c_fire:
        if selected_sc["blast"] != "critical":
            if st.button(f"💥 Fire {selected_sc['id']} faults now", type="primary", key="chaos_fire"):
                fired = []
                for svc, ep, payload in QUICK_FAULTS.get(selected_sc["id"], []):
                    r = fault_post(SERVICES[svc], ep, payload); fired.append(f"{svc}{ep}")
                    write_audit({"event_type":"CHAOS_FAULT_FIRED","run":selected_sc["id"],"service":svc,"endpoint":ep,"result":r,"timestamp":ts()})
                st.success(f"✅ Faults fired: {', '.join(fired)}")
        else:
            st.button("💥 Fire (blocked — HITL required)", disabled=True, key="chaos_fire_blocked")
    with c_clear:
        if st.button("🧹 Clear ALL faults", key="chaos_clear"):
            clear_all_faults(); write_audit({"event_type":"CHAOS_CLEAR_ALL","timestamp":ts()}); st.success("All faults cleared ✅")

    st.divider()
    st.markdown("### 📊 Chaos Run History")
    if CHAOS_LOG.exists() and CHAOS_LOG.stat().st_size > 0:
        chaos_records  = [json.loads(l) for l in CHAOS_LOG.read_text().strip().split("\n") if l.strip()]
        complete_runs  = [r for r in chaos_records if r.get("status") == "COMPLETE"]
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Completed", len(complete_runs))
        cc2.metric("Total logged", len(chaos_records))
        cc3.metric("Scenarios available", len(CHAOS_SCENARIOS))
        for run in reversed(complete_runs[-5:]):
            with st.expander(f"{BLAST_ICON.get(run.get('blast_radius','low'),'⚪')} {run.get('run_id','')} — {run.get('scenario_name','')}"):
                st.markdown(f"**🧠 GPU Post-mortem:** {run.get('post_mortem','—')}")
                if run.get("deltas"):
                    delta_rows = [{"service":s,"cpu Δ%":d.get("cpu_delta",0),"latency Δms":d.get("latency_delta",0),"error Δ":d.get("error_delta",0)} for s,d in run["deltas"].items()]
                    st.dataframe(pd.DataFrame(delta_rows), use_container_width=True)
    else:
        st.info("No chaos runs yet. Run `chaos_scheduler.ipynb` Cell 5 or 6 to begin.")

    st.divider()
    st.markdown("### 🧠 GPU Scenario Recommendation (live)")
    if st.button("🤖 Ask GPU which scenario to run next", key="chaos_gpu_pick"):
        latest_m = get_latest_dict(df)
        sc_list  = "\n".join(f"- {s['id']}: {s['name']} [blast={s['blast']}]" for s in CHAOS_SCENARIOS)
        pick_prompt = f"AIOps agent. Live metrics: {json.dumps(latest_m, default=str)}\nScenarios:\n{sc_list}\nPick best. JSON: {{\"selected_id\":\"SC-XX\",\"reasoning\":\"one sentence\"}}"
        try:
            with st.spinner("Consulting Qwen3-30B..."):
                pick_resp = llm.chat.completions.create(
                    model="Qwen3-30B-A3B", messages=[{"role":"user","content":pick_prompt}],
                    temperature=0.3, max_tokens=120, extra_body={"chat_template_kwargs":{"enable_thinking":False}})
            raw  = pick_resp.choices[0].message.content.strip().replace("```json","").replace("```","")
            pick = json.loads(raw)
            matched = next((s for s in CHAOS_SCENARIOS if s["id"]==pick["selected_id"]),None)
            if matched:
                st.success(f"🧠 GPU recommends: **{matched['id']} — {matched['name']}** [{BLAST_ICON[matched['blast']]} {matched['blast'].upper()}]")
            st.info(f"Reasoning: {pick.get('reasoning','')}")
        except Exception as e:
            st.error(f"GPU pick failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 12 — FAISS VECTOR SEARCH
# ══════════════════════════════════════════════════════════════════════════════
with t12:
    st.subheader("🔎 FAISS Semantic Incident Search — UC-7")
    st.caption("BGE-large embeddings on MI300X · FAISS IndexFlatIP · evidence-backed RCA via retrieval")
    st.info("**Why this matters:** When a new anomaly fires, semantic search retrieves the 3 most similar past incidents from the vector store. The RCA agent gets real evidence instead of guessing from scratch.")

    col_idx, col_search = st.columns([1, 2])
    with col_idx:
        st.markdown("### 📦 Index Status")
        faiss_ready = FAISS_META.exists() and FAISS_META.stat().st_size > 0
        if faiss_ready:
            meta_count = sum(1 for _ in open(FAISS_META))
            st.success(f"✅ Index ready — {meta_count} incidents indexed")
            pca_exists = FAISS_PCA_IMG.exists()
            st.metric("Incidents in index", meta_count)
            st.metric("PCA visualisation", "✅ Ready" if pca_exists else "❌ Run Cell 7")
        else:
            st.warning("Index not built yet")
            st.markdown("""
**Build the index:**
1. Open `faiss_vector_search.ipynb`
2. Run Cells 1 → 2 → 3 → 4
3. Refresh this tab
            """)

    with col_search:
        st.markdown("### 🔍 Semantic Search")
        query_input = st.text_input("Search past incidents", placeholder="payment gateway timeout", key="faiss_query")
        if st.button("🔍 Search", key="faiss_search") and query_input and faiss_ready:
            st.info("Run semantic_search() in notebook Cell 6 and results will appear below.")

    st.divider()

    # Show last search results
    st.markdown("### 📋 Last Retrieval Results")
    if FAISS_RESULTS.exists() and FAISS_RESULTS.stat().st_size > 0:
        results = [json.loads(l) for l in FAISS_RESULTS.read_text().strip().split("\n") if l.strip()]
        if results:
            query_used = results[0].get("query","")
            diagnosis  = results[0].get("diagnosis","")
            st.markdown(f"**Query:** `{query_used}`")
            st.divider()
            st.markdown("**Top-3 Similar Past Incidents:**")
            for i, r in enumerate(results[:3]):
                sim = r.get("similarity", 0)
                bar_len = int(sim * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                with st.expander(f"[{i+1}] {r.get('id','?')} | {r.get('service','?')} | {r.get('fault','?')} | sim={sim:.3f}  {bar}"):
                    c1, c2 = st.columns(2)
                    c1.markdown(f"**Root cause:** {r.get('root_cause','—')}")
                    c2.markdown(f"**Resolution:** {r.get('resolution','—')}")
                    c1.metric("Blast radius", r.get("blast_radius","?"))
                    c2.metric("Similarity",   f"{sim:.3f}")
            st.divider()
            st.markdown("### 🧠 GPU RCA with Retrieved Context")
            st.info(diagnosis or "No diagnosis yet — run Cell 6 in notebook.")
    else:
        st.info("No search results yet. Run Cell 6 in `faiss_vector_search.ipynb`.")

    # PCA visualisation
    if FAISS_PCA_IMG.exists():
        st.divider()
        st.markdown("### 📊 Incident Embedding Space (PCA 2D)")
        st.image(str(FAISS_PCA_IMG), caption="BGE-large embeddings on MI300X — 2D PCA projection, coloured by service")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 13 — ALERT STORM UC-4
# ══════════════════════════════════════════════════════════════════════════════
with t13:
    st.subheader("⛈️ Alert Storm Correlation — UC-4")
    st.caption("NetworkX graph clustering groups correlated alerts · ONE GPU RCA call per group · LLM-as-Judge quality scoring")
    st.info("**AIOps differentiator:** When 10+ alerts fire simultaneously, UC-4 groups them by service dependency graph. One consolidated RCA call per group — not 10 separate calls. Reduces LLM cost 60–80% during alert storms.")

    # Live alert count
    if not lat.empty:
        live_alert_count = sum(
            1 for _, row in lat.iterrows()
            for col, (label, thresh, unit) in THRESHOLDS.items()
            if float(row[col]) > thresh
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Live anomalies now", live_alert_count)
        c2.metric("Storm threshold",    "≥ 5 alerts")
        c3.metric("Status", "⛈️ STORM" if live_alert_count >= 5 else "✅ CALM")
        if live_alert_count >= 5:
            st.error(f"⛈️ Alert storm detected — {live_alert_count} simultaneous anomalies. Run UC-4 notebook!")
        elif live_alert_count > 0:
            st.warning(f"🟡 {live_alert_count} anomalies — inject more faults to trigger storm (need ≥5)")
        else:
            st.success("✅ Cluster calm — inject SC-05 via Chaos Scheduler to trigger a storm")

    st.divider()

    # UC-4 results from notebook
    st.markdown("### 📊 UC-4 Correlation Results")
    if UC4_RESULTS.exists() and UC4_RESULTS.stat().st_size > 0:
        uc4_recs = [json.loads(l) for l in UC4_RESULTS.read_text().strip().split("\n") if l.strip()]
        rca_recs = [r for r in uc4_recs if r.get("event_type") == "UC4_RCA_COMPLETE"]
        if rca_recs:
            latest_run = sorted(rca_recs, key=lambda x: x.get("timestamp",""))[-1]
            total_alerts = latest_run.get("total_alerts", "?")
            st.markdown(f"**Last run:** {latest_run.get('timestamp','')[:19]}")
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Alerts in storm",    total_alerts)
            rc2.metric("Incident groups",    latest_run.get("group_id","?")[-2:] if rca_recs else "?")
            rc3.metric("LLM calls used",     len(rca_recs))

            st.divider()
            for rec in rca_recs[-5:]:
                sev_icon = "💀" if rec.get("severity")=="CRITICAL" else ("🔴" if rec.get("severity")=="HIGH" else "🟡")
                with st.expander(f"{sev_icon} {rec.get('group_id')} — {', '.join(rec.get('services',[]))} — {rec.get('severity')}"):
                    st.markdown(f"**Root candidate:** `{rec.get('root_candidate','?')}`")
                    st.markdown("**GPU RCA:**")
                    for line in rec.get("rca","").split("\n"):
                        if line.strip(): st.markdown(f"  {line}")
                    st.caption(f"Alerts: {rec.get('total_alerts',0)} total, {rec.get('high_alerts',0)} HIGH · Time: {rec.get('timestamp','')[:19]}")
    else:
        st.info("No UC-4 results yet. Run `uc4_alert_storm.ipynb` Cells 1 → 5.")
        st.markdown("""
**Quick start:**
1. Inject SC-05 via Chaos Scheduler tab to create an alert storm
2. Open `uc4_alert_storm.ipynb` → run Cells 1 → 2 → 3 → 4 → 5
3. Refresh this tab
        """)

    # Correlation graph image
    if UC4_GRAPH_IMG.exists():
        st.divider()
        st.markdown("### 🕸️ Alert Correlation Graph")
        st.image(str(UC4_GRAPH_IMG), caption="NetworkX correlation graph — nodes=alerted services, edges=dependency paths, width=alert weight")

    # Judge results
    st.divider()
    st.markdown("### ⚖️ LLM-as-Judge Results")
    if JUDGE_RESULTS.exists() and JUDGE_RESULTS.stat().st_size > 0:
        judge_recs = [json.loads(l) for l in JUDGE_RESULTS.read_text().strip().split("\n") if l.strip()]
        if judge_recs:
            judge_rows = []
            for j in judge_recs:
                ev = j.get("evaluation", {})
                verdict = ev.get("verdict","?")
                icon = "✅" if verdict=="PASS" else ("⚠️" if verdict=="REVIEW" else "❌")
                judge_rows.append({
                    "group":          j.get("group_id","?"),
                    "services":       ", ".join(j.get("services",[])),
                    "verdict":        f"{icon} {verdict}",
                    "overall":        ev.get("overall",0),
                    "specificity":    ev.get("specificity",0),
                    "actionability":  ev.get("actionability",0),
                    "causality":      ev.get("causality",0),
                    "feedback":       ev.get("feedback",""),
                })
            jdf = pd.DataFrame(judge_rows)
            avg = jdf["overall"].mean()
            j1, j2, j3 = st.columns(3)
            j1.metric("Avg Judge Score", f"{avg:.1f}/5")
            j2.metric("PASS",   sum(1 for r in judge_rows if "PASS" in r["verdict"]))
            j3.metric("REVIEW/FAIL", sum(1 for r in judge_rows if "PASS" not in r["verdict"]))
            st.dataframe(jdf, use_container_width=True)
    else:
        st.info("No judge results yet. Run Cell 6 in `uc4_alert_storm.ipynb`.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 14 — LOG EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════
with t14:
    st.subheader("📜 Log Embedding — UC-8")
    st.caption("BGE-large semantic log search on MI300X · FAISS log index · GPU narrative from anomalous clusters")
    st.info("**What this enables:** Every log entry is embedded as a vector. Searching 'payment gateway timeout' retrieves semantically similar log lines across all services — not just keyword matches. Anomalous log clusters feed directly into GPU narrative generation.")

    col_s1, col_s2, col_s3 = st.columns(3)
    log_idx_ready = LOG_META.exists() and LOG_META.stat().st_size > 0
    log_count = sum(1 for _ in open(LOG_META)) if log_idx_ready else 0
    col_s1.metric("Log index",   "✅ Ready" if log_idx_ready else "❌ Not built")
    col_s2.metric("Logs indexed", log_count if log_idx_ready else 0)
    col_s3.metric("PCA plot",    "✅ Ready" if LOG_PCA_IMG.exists() else "❌ Run Cell 5")

    if not log_idx_ready:
        st.divider()
        st.markdown("""
**Build the log index:**
1. Open `log_embedding.ipynb`
2. Run Cells 1 → 2 → 3 (builds FAISS log index on GPU)
3. Run Cell 4 for semantic search demo
4. Run Cell 5 for PCA visualisation
5. Run Cell 6 for GPU narrative
6. Refresh this tab
        """)
        st.code("# Upload notebook:\ncurl -L 'https://raw.githubusercontent.com/PrasenjitRoychoudhury/AMDHackathon/main/log_embedding.ipynb' -o /workspace/shared/minicluster/log_embedding.ipynb", language="bash")
    else:
        st.divider()
        st.markdown("### 🔍 Semantic Log Search")
        log_query = st.text_input("Search logs", placeholder="JWT token authentication failure", key="log_search")
        if st.button("🔍 Search logs", key="log_btn") and log_query and log_idx_ready:
            st.info("Run search_logs() in notebook Cell 4 with this query and results will appear below.")

        # Show recent audit log entries as live log stream
        st.divider()
        st.markdown("### 📋 Live Log Stream (Audit Log)")
        if AUDIT_FILE.exists():
            audit_lines = AUDIT_FILE.read_text().strip().split("\n")[-20:]
            audit_entries = []
            for line in audit_lines:
                if not line.strip(): continue
                try:
                    rec = json.loads(line)
                    level_map = {"FAULT_INJECT":"WARN","FAULT_CLEAR_ALL":"INFO","HITL_APPROVED":"INFO",
                                 "HITL_REJECTED":"WARN","CHAOS_FAULT_FIRED":"WARN","UC4_RCA_COMPLETE":"INFO",
                                 "LLM_JUDGE":"INFO","AE_RCA_COMPLETE":"WARN","LOG_EMBEDDING_NARRATIVE":"INFO"}
                    level = level_map.get(rec.get("event_type",""), "INFO")
                    audit_entries.append({
                        "timestamp": rec.get("timestamp","")[:19],
                        "level":     level,
                        "service":   rec.get("service","system"),
                        "event":     rec.get("event_type",""),
                    })
                except: pass
            if audit_entries:
                adf2 = pd.DataFrame(audit_entries)
                st.dataframe(adf2.sort_values("timestamp", ascending=False), use_container_width=True)

        st.divider()
        st.markdown("### 📊 Log Embedding Space (PCA 2D)")
        if LOG_PCA_IMG.exists():
            st.image(str(LOG_PCA_IMG), caption="BGE-large log embeddings on MI300X — left: coloured by level, right: coloured by service")
        else:
            st.info("PCA plot not yet generated — run Cell 5 in `log_embedding.ipynb`.")

        # GPU narrative from audit
        st.divider()
        st.markdown("### 🧠 GPU Log Narrative (Qwen3-30B)")
        if AUDIT_FILE.exists():
            audit_recs2 = [json.loads(l) for l in AUDIT_FILE.read_text().strip().split("\n") if l.strip()]
            narratives = [r for r in audit_recs2 if r.get("event_type") == "LOG_EMBEDDING_NARRATIVE"]
            if narratives:
                latest_n = sorted(narratives, key=lambda x: x.get("timestamp",""))[-1]
                st.warning(latest_n.get("narrative",""))
                st.caption(f"Generated at {latest_n.get('timestamp','')[:19]} · {latest_n.get('anomalous_count',0)} anomalous logs · GPU: {latest_n.get('gpu_time_secs',0)}s")
            else:
                st.info("No GPU narrative yet — run Cell 6 in `log_embedding.ipynb`.")
