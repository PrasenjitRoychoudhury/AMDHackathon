"""AGENTS026 - SRE Operator Console (Streamlit version)"""
from pathlib import Path
from datetime import datetime, timezone
import json, uuid

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title='AGENTS026 SRE Console', page_icon='🚨', layout='wide')

ROOT = Path('/workspace/shared/agents026')
DATA = ROOT / 'data'
INCIDENTS_DIR = DATA / 'incidents'
EXEC_DIR = DATA / 'execution'
REPORTS_DIR = DATA / 'reports'
AUDIT_PATH = DATA / 'audit' / 'audit_log.jsonl'
PREV_PATH = DATA / 'prevention' / 'prev-001_prevention.json'
OPERATOR = 'prasenjit.roychoudhury'

# ---------- data access ----------
@st.cache_data(ttl=10)
def load_incidents():
    return json.load(open(INCIDENTS_DIR / 'incident_candidates.json', encoding='utf-8'))

@st.cache_data(ttl=10)
def load_metrics():
    return pd.read_csv(DATA / 'metrics.csv', parse_dates=['timestamp'])

def load_workflow(iid):
    fp = EXEC_DIR / f'{iid}_workflow.json'
    return json.load(open(fp, encoding='utf-8')) if fp.exists() else None

def save_workflow(bundle):
    json.dump(bundle, open(EXEC_DIR / f"{bundle['incident_id']}_workflow.json", 'w', encoding='utf-8'),
              default=str, indent=2)

def audit_event(event_type, incident_id, payload):
    rec = {'audit_id': f'aud-{uuid.uuid4().hex[:12]}',
           'timestamp_utc': datetime.now(timezone.utc).isoformat(),
           'event_type': event_type, 'incident_id': incident_id,
           'operator': OPERATOR, 'payload': payload}
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, default=str) + '\n')

EXECUTORS = {
    'rollback_deploy': lambda a: f"Simulated rollback of {a['target']} to {a.get('parameters',{}).get('version','previous')}",
    'rollback_config': lambda a: f"Simulated config rollback for {a['target']}",
    'disable_feature_flag': lambda a: f"Simulated flag disable on {a['target']}",
    'restart_service': lambda a: f"Simulated restart of {a['target']}",
    'scale_service': lambda a: f"Simulated scale of {a['target']}",
    'clear_cache': lambda a: f"Simulated cache clear for {a['target']}",
    'drain_node': lambda a: f"Simulated node drain for {a['target']}",
}

# ---------- layout ----------
st.title('🚨 AGENTS026 — SRE Operator Console')
st.caption('Autonomous Incident Diagnosis & Resolution · AMD Instinct MI300X · Qwen3-30B via vLLM')

incidents = load_incidents()
metrics_df = load_metrics()
incident_map = {x['incident_id']: x for x in incidents}

with st.sidebar:
    st.header('Incident selector')
    options = [f"{x['incident_id']} — {','.join(x['services'])} ({x['anomaly_type']})" for x in incidents]
    default_ix = next((i for i, x in enumerate(incidents) if x['incident_id'] == 'inc-014'), 0)
    sel = st.selectbox('Incident', options, index=default_ix)
    iid = sel.split(' ')[0]
    st.divider()
    st.metric('Total incidents', len(incidents))
    if AUDIT_PATH.exists():
        n_audit = sum(1 for _ in open(AUDIT_PATH, encoding='utf-8'))
        st.metric('Audit records', n_audit)

inc = incident_map[iid]
tab1, tab2, tab3, tab4 = st.tabs(['🚨 Incident', '🔍 RCA & Actions', '🛡️ Approval Queue', '📋 Governance'])

# ---------- Tab 1: incident ----------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    ms = inc.get('metric_summary', {})
    c1.metric('Service', inc['services'][0])
    c2.metric('CPU max', f"{ms.get('cpu_max',0):.1f}%")
    c3.metric('p95 latency max', f"{ms.get('latency_p95_max',0):.0f}ms")
    c4.metric('Error rate max', f"{ms.get('error_rate_max',0):.4f}")
    st.write(f"**Window:** {inc['start_time']} → {inc['end_time']} | **Anomaly:** `{inc['anomaly_type']}`")

    if inc.get('change_refs'):
        with st.expander('Recent change events (RCA breadcrumbs)', expanded=True):
            for c in inc['change_refs'][:4]:
                st.write('•', c)

    svc = inc['services'][0]
    s, e = pd.Timestamp(inc['start_time']), pd.Timestamp(inc['end_time'])
    sdf = metrics_df[(metrics_df.service == svc) &
                     (metrics_df.timestamp >= s - pd.Timedelta(minutes=40)) &
                     (metrics_df.timestamp <= e + pd.Timedelta(minutes=40))]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3))
    for ax, m in zip(axes, ['cpu_utilization', 'latency_p95_ms', 'error_rate']):
        ax.plot(sdf.timestamp, sdf[m], lw=1.2)
        ax.axvspan(s, e, color='red', alpha=0.15)
        ax.set_title(m, fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
    plt.tight_layout()
    st.pyplot(fig)

# ---------- Tab 2: RCA ----------
with tab2:
    bundle = load_workflow(iid)
    if not bundle:
        st.info(f'No workflow has been run for {iid} yet.')
    else:
        rca = bundle['rca_result']
        st.subheader(f'Root Cause Analysis — {iid}')
        st.write(f"**Hypothesis:** {rca['root_cause_hypothesis']}")
        cc1, cc2 = st.columns(2)
        cc1.metric('Probable trigger', rca.get('probable_trigger', '-'))
        cc2.metric('Confidence', f"{rca['confidence']:.0%}")
        st.write('**Evidence:**')
        for ev in rca['evidence']:
            st.write('•', ev)
        st.subheader('Action plan & status')
        adf = pd.DataFrame(bundle['action_plan']['actions'])
        st.dataframe(adf[['action_type', 'target', 'requires_approval', 'status']], use_container_width=True)
        rp = REPORTS_DIR / f'{iid}_report.md'
        if rp.exists():
            with st.expander('📄 Post-incident report'):
                st.markdown(rp.read_text(encoding='utf-8'))

# ---------- Tab 3: approval queue ----------
with tab3:
    bundle = load_workflow(iid)
    if not bundle:
        st.info('No workflow for this incident.')
    else:
        pending = [a for a in bundle['execution_result']['skipped_actions'] if a['status'] == 'skipped']
        if not pending:
            st.success('No actions pending approval for this incident ✅')
        for action in pending:
            with st.container(border=True):
                st.write(f"**{action['action_type']}** → `{action['target']}`")
                st.caption(action['description'])
                st.write(f"**Expected impact:** {action.get('expected_impact','-')}")
                b1, b2, _ = st.columns([1, 1, 4])
                if b1.button('✅ Approve', key=f"ok-{action['action_id']}"):
                    audit_event('human_decision', iid, {'action_id': action['action_id'], 'decision': 'approved'})
                    msg = EXECUTORS.get(action['action_type'], lambda a: 'no executor')(action)
                    action['status'] = 'executed'
                    audit_event('action_executed', iid, {'action_id': action['action_id'], 'result': msg})
                    bundle['execution_result']['executed_actions'].append(action)
                    bundle['execution_result']['skipped_actions'] = [
                        a for a in bundle['execution_result']['skipped_actions'] if a['action_id'] != action['action_id']]
                    save_workflow(bundle)
                    st.success(msg)
                    st.rerun()
                if b2.button('❌ Reject', key=f"no-{action['action_id']}"):
                    audit_event('human_decision', iid, {'action_id': action['action_id'], 'decision': 'rejected'})
                    action['status'] = 'rejected'
                    audit_event('action_rejected', iid, {'action_id': action['action_id']})
                    save_workflow(bundle)
                    st.warning('Action rejected by operator')
                    st.rerun()

# ---------- Tab 4: governance ----------
with tab4:
    if PREV_PATH.exists():
        p = json.load(open(PREV_PATH, encoding='utf-8'))
        st.subheader('🔮 Predictive Prevention (UC-5)')
        g1, g2, g3 = st.columns(3)
        g1.metric('Lead time', f"{p['lead_time_minutes']} min")
        g2.metric('Preventive action', p['preventive_action']['action_type'])
        g3.metric('Outcome', p['outcome'].replace('_', ' '))
    st.subheader('📋 Compliance Audit Trail (append-only)')
    if AUDIT_PATH.exists():
        adf = pd.DataFrame([json.loads(l) for l in open(AUDIT_PATH, encoding='utf-8')])
        st.dataframe(adf[['timestamp_utc', 'event_type', 'incident_id', 'operator']].iloc[::-1],
                     use_container_width=True, height=400)
    else:
        st.info('No audit records yet.')
