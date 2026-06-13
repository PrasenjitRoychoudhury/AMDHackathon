"""AGENTS026 — Live SRE Console (MiniCluster edition)"""
import streamlit as st
import pandas as pd
import numpy as np
import json, requests, time
from datetime import datetime, timezone
from pathlib import Path

st.set_page_config(page_title="AGENTS026 Live Console", page_icon="🖥️", layout="wide")

# ── config ────────────────────────────────────────────────────────────────────
METRICS_CSV = Path("/workspace/shared/minicluster/live_metrics.csv")
HITL_FILE   = Path("/workspace/shared/hitl_queue.jsonl")
AUDIT_FILE  = Path("/workspace/shared/audit_log.jsonl")

SERVICES = {
    "payments": 7001,
    "auth":     7002,
    "checkout": 7003,
    "fraud":    7004,
}

THRESHOLDS = {
    "cpu_pct":      ("CPU %",      70.0,  "%"),
    "latency_p99":  ("Latency p99", 2.0,  "s"),
    "error_rate":   ("Error rate",  0.05,  ""),
    "mem_gb":       ("Memory",      1.8,   "GB"),
}

# ── helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=8)
def load_metrics():
    if not METRICS_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(METRICS_CSV, names=["timestamp","service","error_rate",
                                          "latency_p99","mem_gb","cpu_pct","disk_mb","node"],
                     parse_dates=["timestamp"])
    return df.tail(600)   # last 10 rows per service × 60 mins

def latest(df):
    if df.empty:
        return pd.DataFrame()
    return df.sort_values("timestamp").groupby("service").last().reset_index()

def fault_post(port, path, payload):
    try:
        r = requests.post(f"http://127.0.0.1:{port}{path}",
                          json=payload, timeout=3)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def clear_all_faults():
    results = {}
    for svc, port in SERVICES.items():
        results[svc] = fault_post(port, "/fault/clear", {})
    return results

def write_hitl(event):
    HITL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HITL_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")

def write_audit(event):
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")

def ts():
    return datetime.now(timezone.utc).isoformat()

# ── header ────────────────────────────────────────────────────────────────────
st.title("🖥️ AGENTS026 — Live SRE Console")
st.caption("AMD Instinct MI300X · Qwen3-30B via vLLM · MiniCluster banking stack")

col_ref, col_auto = st.columns([6,1])
with col_ref:
    if st.button("🔄 Refresh now"):
        st.cache_data.clear()
with col_auto:
    auto = st.toggle("Auto-refresh (10s)", value=False)
if auto:
    time.sleep(10)
    st.cache_data.clear()
    st.rerun()

df = load_metrics()
lat = latest(df)

# ── tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5, t6 = st.tabs([
    "❤️ Health", "📈 Live Telemetry",
    "⚠️ Anomalies", "💥 Fault Injection",
    "🤖 AI Actions", "🛑 HITL Queue"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — HEALTH
# ══════════════════════════════════════════════════════════════════════════════
with t1:
    st.subheader("Current cluster health")
    if lat.empty:
        st.warning("No metrics yet — waiting for collector...")
    else:
        cols = st.columns(len(lat))
        for i, row in lat.iterrows():
            svc  = row["service"]
            cpu  = row["cpu_pct"]
            lat_ = row["latency_p99"]
            err  = row["error_rate"]
            mem  = row["mem_gb"]

            bad = (cpu > 70) or (lat_ > 2.0) or (err > 0.05) or (mem > 1.8)
            icon = "🔴" if bad else "🟢"

            with cols[i % len(cols)]:
                st.metric(f"{icon} {svc.upper()}", "")
                st.metric("CPU %",       f"{cpu:.1f}%",   delta=None)
                st.metric("Latency p99", f"{lat_:.3f}s",  delta=None)
                st.metric("Error rate",  f"{err:.4f}",    delta=None)
                st.metric("Memory GB",   f"{mem:.2f}",    delta=None)

        # Overall verdict
        any_bad = any(
            (r["cpu_pct"] > 70) or (r["latency_p99"] > 2.0) or
            (r["error_rate"] > 0.05) or (r["mem_gb"] > 1.8)
            for _, r in lat.iterrows()
        )
        st.divider()
        if any_bad:
            st.error("⚠️  One or more services are breaching thresholds — check Anomalies tab")
        else:
            st.success("✅  All services within normal thresholds")

        st.caption(f"Last updated: {lat['timestamp'].max()}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE TELEMETRY
# ══════════════════════════════════════════════════════════════════════════════
with t2:
    st.subheader("Live telemetry — last 30 minutes")
    if df.empty:
        st.warning("No data yet.")
    else:
        recent = df[df["timestamp"] >= df["timestamp"].max() - pd.Timedelta("30min")]
        metric_choice = st.selectbox("Metric", ["latency_p99","cpu_pct","error_rate","mem_gb"], key="met")
        svc_filter    = st.multiselect("Services", df["service"].unique().tolist(),
                                       default=df["service"].unique().tolist(), key="svc")
        filtered = recent[recent["service"].isin(svc_filter)]

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 3))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

        colors = ["#00d4ff","#ff6b6b","#51cf66","#ffd43b","#cc5de8","#ff922b"]
        for idx, svc in enumerate(svc_filter):
            s = filtered[filtered["service"] == svc].sort_values("timestamp")
            ax.plot(s["timestamp"], s[metric_choice],
                    label=svc, color=colors[idx % len(colors)], linewidth=1.8)

        # threshold line
        thresh_map = {"cpu_pct": 70, "latency_p99": 2.0, "error_rate": 0.05, "mem_gb": 1.8}
        if metric_choice in thresh_map:
            ax.axhline(thresh_map[metric_choice], color="#ff4444",
                       linestyle="--", linewidth=1, label="threshold")

        ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
        ax.set_xlabel("Time", color="white")
        ax.set_ylabel(metric_choice, color="white")
        st.pyplot(fig)
        plt.close()

        st.dataframe(filtered.sort_values("timestamp", ascending=False).head(40),
                     use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANOMALIES
# ══════════════════════════════════════════════════════════════════════════════
with t3:
    st.subheader("Threshold-based anomaly detection")
    if lat.empty:
        st.warning("No data yet.")
    else:
        anomalies = []
        for _, row in lat.iterrows():
            for col, (label, thresh, unit) in THRESHOLDS.items():
                val = row[col]
                if val > thresh:
                    anomalies.append({
                        "service":   row["service"],
                        "metric":    label,
                        "value":     f"{val:.4f} {unit}",
                        "threshold": f"{thresh} {unit}",
                        "severity":  "🔴 HIGH" if val > thresh * 1.5 else "🟡 WARN",
                        "node":      row.get("node","?"),
                        "detected":  str(row["timestamp"]),
                    })

        if anomalies:
            st.error(f"⚠️  {len(anomalies)} anomaly/-ies detected")
            adf = pd.DataFrame(anomalies)
            st.dataframe(adf, use_container_width=True)

            # Send selected to HITL
            st.divider()
            st.markdown("**Escalate to HITL queue**")
            sel_svc = st.selectbox("Select service to escalate",
                                   adf["service"].unique().tolist(), key="esc_svc")
            if st.button("📤 Send to HITL queue"):
                evt = {
                    "hitl_id":   f"hitl-{int(time.time())}",
                    "timestamp": ts(),
                    "source":    "anomaly_detector",
                    "service":   sel_svc,
                    "anomalies": [a for a in anomalies if a["service"] == sel_svc],
                    "status":    "PENDING",
                }
                write_hitl(evt)
                write_audit({**evt, "event_type": "HITL_CREATED"})
                st.success(f"Sent {sel_svc} anomalies to HITL queue ✅")
        else:
            st.success("✅  No anomalies — all metrics within threshold")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — FAULT INJECTION
# ══════════════════════════════════════════════════════════════════════════════
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
            write_audit({"event_type":"FAULT_INJECT","service":target,
                         "fault":"latency","ms":ms,"result":r,"timestamp":ts()})
            st.success(f"✅ {r}")

        st.markdown("#### 💥 Error rate")
        pct = st.slider("Error %", 5, 80, 30, 5, key="err_pct")
        if st.button(f"Inject {pct}% errors → {target}"):
            r = fault_post(port, "/fault/errors", {"pct": pct/100})
            write_audit({"event_type":"FAULT_INJECT","service":target,
                         "fault":"errors","pct":pct/100,"result":r,"timestamp":ts()})
            st.success(f"✅ {r}")

    with col2:
        st.markdown("#### 🔥 CPU spike")
        secs = st.slider("Duration (s)", 10, 120, 30, 10, key="cpu_sec")
        if st.button(f"Inject CPU spin {secs}s → {target}"):
            r = fault_post(port, "/fault/cpu_spin", {"seconds": secs})
            write_audit({"event_type":"FAULT_INJECT","service":target,
                         "fault":"cpu_spin","seconds":secs,"result":r,"timestamp":ts()})
            st.success(f"✅ {r}")

        st.markdown("#### 🧠 Memory leak")
        mb = st.slider("MB/min", 10, 200, 50, 10, key="mem_mb")
        if st.button(f"Inject mem leak {mb}MB/min → {target}"):
            r = fault_post(port, "/fault/mem_leak", {"mb_per_min": mb})
            write_audit({"event_type":"FAULT_INJECT","service":target,
                         "fault":"mem_leak","mb_per_min":mb,"result":r,"timestamp":ts()})
            st.success(f"✅ {r}")

    st.divider()
    if st.button("🧹 CLEAR ALL FAULTS (all services)", type="primary"):
        results = clear_all_faults()
        write_audit({"event_type":"FAULT_CLEAR_ALL","results":results,"timestamp":ts()})
        st.success(f"All faults cleared: {results}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — AI ACTIONS
# ══════════════════════════════════════════════════════════════════════════════
with t5:
    st.subheader("AI-decided actions log")
    st.info("This tab shows actions the autonomous agent has decided and executed. "
            "Actions requiring approval appear in the HITL Queue tab.")

    if AUDIT_FILE.exists():
        records = []
        with open(AUDIT_FILE) as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except:
                    pass
        if records:
            adf = pd.DataFrame(records)
            # show most recent first
            adf = adf.sort_values("timestamp", ascending=False) if "timestamp" in adf.columns else adf
            st.dataframe(adf, use_container_width=True)

            # summary counts
            if "event_type" in adf.columns:
                st.divider()
                st.markdown("**Event breakdown**")
                st.bar_chart(adf["event_type"].value_counts())
        else:
            st.info("No AI actions recorded yet — trigger a fault to see the agent respond.")
    else:
        st.info("Audit log not yet created. Inject a fault or escalate an anomaly to HITL to start.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — HITL QUEUE
# ══════════════════════════════════════════════════════════════════════════════
with t6:
    st.subheader("Human-in-the-Loop approval queue")

    if not HITL_FILE.exists() or HITL_FILE.stat().st_size == 0:
        st.info("No pending HITL items. Escalate an anomaly from the Anomalies tab to populate this queue.")
    else:
        items = []
        with open(HITL_FILE) as f:
            for line in f:
                try:
                    items.append(json.loads(line))
                except:
                    pass

        pending   = [x for x in items if x.get("status") == "PENDING"]
        resolved  = [x for x in items if x.get("status") != "PENDING"]

        st.metric("Pending approvals", len(pending))
        st.metric("Resolved",          len(resolved))

        if pending:
            st.divider()
            st.markdown("### ⏳ Pending items")
            for item in pending:
                with st.expander(f"🔴 {item.get('hitl_id')} — {item.get('service')} — {item.get('timestamp','')}"):
                    st.json(item)
                    col_a, col_r = st.columns(2)
                    with col_a:
                        if st.button("✅ Approve", key=f"appr_{item['hitl_id']}"):
                            item["status"]      = "APPROVED"
                            item["resolved_at"] = ts()
                            item["operator"]    = "prasenjit.roychoudhury"
                            # rewrite file
                            all_items = [x if x["hitl_id"] != item["hitl_id"] else item for x in items]
                            with open(HITL_FILE, "w") as f:
                                for rec in all_items:
                                    f.write(json.dumps(rec) + "\n")
                            write_audit({**item, "event_type": "HITL_APPROVED"})
                            st.success("Approved — agent will execute remediation.")
                            st.rerun()
                    with col_r:
                        if st.button("❌ Reject", key=f"rej_{item['hitl_id']}"):
                            item["status"]      = "REJECTED"
                            item["resolved_at"] = ts()
                            item["operator"]    = "prasenjit.roychoudhury"
                            all_items = [x if x["hitl_id"] != item["hitl_id"] else item for x in items]
                            with open(HITL_FILE, "w") as f:
                                for rec in all_items:
                                    f.write(json.dumps(rec) + "\n")
                            write_audit({**item, "event_type": "HITL_REJECTED"})
                            st.warning("Rejected — no action will be taken.")
                            st.rerun()

        if resolved:
            st.divider()
            st.markdown("### ✅ Resolved items")
            st.dataframe(pd.DataFrame(resolved), use_container_width=True)
