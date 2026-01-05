import os
import subprocess
from typing import Optional, Dict, Any

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from log_parser import parse_output
from storage import load_history, append_run

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="CARB Platform", layout="wide")

# -----------------------------
# CSS (Figma-like glass + deep blue + green accents)
# -----------------------------
st.markdown("""
<style>
:root{
  --bg0:#06111b;
  --bg1:#0a1b2a;
  --card: rgba(255,255,255,0.06);
  --card2: rgba(255,255,255,0.08);
  --border: rgba(255,255,255,0.10);
  --txt: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.65);
  --green:#2ee6a8;
  --blue:#4aa3ff;
  --amber:#f6b11a;
  --purple:#b36cff;
}

.stApp{
  background: radial-gradient(circle at 20% 10%, #0d2d3a 0%, var(--bg0) 40%, #040911 100%);
  color: var(--txt);
}

.block-container{padding-top: 1.2rem; max-width: 1400px;}
hr{border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 14px 0;}

.hero{
  background: linear-gradient(180deg, rgba(46,230,168,0.10) 0%, rgba(255,255,255,0.05) 100%);
  border:1px solid rgba(46,230,168,0.18);
  border-radius: 22px;
  padding: 22px;
}
.hero h1{margin:0; font-size: 44px; letter-spacing:-0.02em;}
.hero p{margin:8px 0 0 0; color: var(--muted); font-size: 16px; line-height: 1.5;}

.card{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 16px 16px;
  backdrop-filter: blur(10px);
}

.kpi-title{color: var(--muted); font-size: 14px; margin-bottom: 8px;}
.kpi-big{font-size: 42px; font-weight: 900; letter-spacing:-0.02em;}
.kpi-sub{color: var(--muted); font-size: 12px; margin-top: 6px;}

.pill{
  display:inline-block;
  padding:6px 10px;
  border-radius:999px;
  background: rgba(46,230,168,0.12);
  border: 1px solid rgba(46,230,168,0.22);
  color: var(--green);
  font-size: 12px;
  font-weight: 800;
}

.section-title{font-size: 20px; font-weight: 800; margin: 0 0 8px 0;}
.section-sub{color: var(--muted); font-size: 13px; margin: 0 0 10px 0;}

.stage{
  position: relative;
  height: 260px;
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,0.10);
  overflow: hidden;
  background: rgba(255,255,255,0.06);
}
.stage .top{
  height: 54%;
  background: linear-gradient(135deg, rgba(46,230,168,0.18), rgba(74,163,255,0.10));
}
.stage .bottom{
  height: 46%;
  padding: 14px 16px;
}
.stage .num{
  position:absolute; left:16px; top:14px;
  font-size: 38px; font-weight: 900; color: rgba(255,255,255,0.85);
}
.stage .icon{
  position:absolute; right:16px; top:16px;
  width:44px; height:44px; border-radius:14px;
  display:flex; align-items:center; justify-content:center;
  background: rgba(255,255,255,0.10);
  border: 1px solid rgba(255,255,255,0.10);
  font-size:22px;
}
.stage h3{margin:0 0 6px 0; font-size: 20px;}
.stage p{margin:0; color: var(--muted); font-size: 13px; line-height: 1.5;}
.badge{
  display:inline-block;
  margin-top:10px;
  padding:6px 10px;
  border-radius:999px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  color: rgba(255,255,255,0.78);
  font-size: 12px;
  font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Helpers
# -----------------------------
def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def fmt(x: Optional[float], digits=2):
    if x is None:
        return "—"
    return f"{x:.{digits}f}"

def rerun():
    # Streamlit renamed experimental_rerun -> rerun in newer versions
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def run_training(python_exe: str, workdir: str, script_path: str) -> Dict[str, Any]:
    """
    Runs your real MindSpore training script and returns parsed metrics.
    No logs are shown. We capture output internally for parsing.
    """
    full_script = os.path.join(workdir, script_path)
    if not os.path.exists(full_script):
        raise FileNotFoundError(f"Training script not found: {full_script}")

    # Use -u so prints flush correctly (even though we don't display them)
    proc = subprocess.Popen(
        [python_exe, "-u", script_path],
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line.rstrip("\n"))

    code = proc.wait()
    if code != 0:
        tail = "\n".join(lines[-60:])
        raise RuntimeError(f"Training exited with code {code}.\n\nLast output:\n{tail}")

    return parse_output(lines)


def plot_runs_big(df_runs: pd.DataFrame, metric: str, show_best_marker: bool = True):
    """
    Big main chart across runs.
    metric: "reduction_pct" or "served_carbon"
    """
    fig = go.Figure()

    y = df_runs[metric].astype(float)
    x = df_runs["run_index"].astype(int)

    name = "Carbon Reduction (%)" if metric == "reduction_pct" else "Served Carbon (lower is better)"

    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines+markers",
        name=name,
        hovertemplate="Run %{x}<br>Value: %{y:.3f}<extra></extra>"
    ))

    if show_best_marker and metric == "served_carbon":
        # Mark best (lowest)
        best_idx = y.idxmin()
        fig.add_trace(go.Scatter(
            x=[int(df_runs.loc[best_idx, "run_index"])],
            y=[float(df_runs.loc[best_idx, metric])],
            mode="markers",
            name="Global Best (min served carbon)",
            marker=dict(size=14)
        ))

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.86)"),
        xaxis=dict(title="Run #", gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(title="", gridcolor="rgba(255,255,255,0.07)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_single_run_eval_small(run: Dict[str, Any]):
    pts = run.get("eval_points", []) or []
    if not pts:
        st.info("No EVAL checkpoints detected in this run output.")
        return

    df = pd.DataFrame(pts).sort_values("ep")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["ep"],
        y=df["carbon"],
        mode="lines+markers",
        name="EVAL carbon",
        hovertemplate="Ep %{x}<br>Carbon %{y:.2f}<br>SLA %{customdata}<extra></extra>",
        customdata=df["sla"]
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.86)"),
        xaxis=dict(title="Episode checkpoint", gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(title="Carbon", gridcolor="rgba(255,255,255,0.07)"),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_action_mix_small(run: Dict[str, Any]):
    pts = run.get("eval_points", []) or []
    if not pts:
        st.info("No action mix detected (needs [ACTIONS] lines before [EVAL]).")
        return

    last = pts[-1]
    acts = last.get("actions")
    if not acts:
        st.info("No action mix detected in last checkpoint.")
        return

    total = acts["run"] + acts["delay"] + acts["batt"]
    if total <= 0:
        st.info("Action counts are zero.")
        return

    labels = ["Run Now", "Delay Flexible", "Use Battery"]
    values = [acts["run"], acts["delay"], acts["batt"]]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        hovertemplate="%{x}<br>%{y} selections<extra></extra>"
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.86)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(title="Count", gridcolor="rgba(255,255,255,0.07)"),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Sidebar (minimal + clean)
# -----------------------------
with st.sidebar:
    st.markdown("### Run Control")
    st.caption("Runs your real MindSpore training script and updates the dashboard.")
    workdir = st.text_input("Working directory (project root)", value="..")
    script_path = st.text_input("Training script", value="train_ms_dqn.py")
    python_exe = st.text_input("Python executable", value=r"..\venv\Scripts\python.exe")
    st.markdown("---")
    run_btn = st.button("Run Training (MindSpore)", type="primary")
    st.markdown("<div class='small-muted'>Tip: training may be quiet at start because predictor fitting happens before the first print.</div>", unsafe_allow_html=True)


# -----------------------------
# HERO
# -----------------------------
st.markdown("""
<div class="hero">
  <div class="pill">CARB — Carbon-Aware Resource Allocation Brain</div>
  <h1>CARB-X Carbon-Aware Resource Allocation Brain</h1>
  <p>
    CARB-X is a live AI decision system that learns how to reduce data-center 
    carbon emissions over time while guaranteeing zero SLA violations.
  </p>
</div>
""", unsafe_allow_html=True)

st.write("")


# -----------------------------
# Run training (no logs shown) + auto rerun
# -----------------------------
if run_btn:
    with st.status("Training is running… (MindSpore + Predictor + DQN)", expanded=False) as status:
        try:
            out = run_training(python_exe=python_exe, workdir=workdir, script_path=script_path)
            saved = append_run(out)
            status.update(label=f"Training finished. Run #{saved['run_index']} saved.", state="complete")
            st.success(f"Run #{saved['run_index']} saved. Dashboard updating…")
            rerun()
        except Exception as e:
            status.update(label="Training failed.", state="error")
            st.error(str(e))
            st.stop()

# -----------------------------
# Load history
# -----------------------------
hist = load_history()
if not hist:
    st.markdown('<div class="card">No runs yet. Use the sidebar to run training once.</div>', unsafe_allow_html=True)
    st.stop()

df_runs = pd.DataFrame(hist)
df_runs["run_index"] = range(1, len(df_runs) + 1)

latest = hist[-1]
baseline = safe_float(latest.get("baseline_carbon"), 0.0)
served = safe_float(latest.get("served_carbon"), 0.0)
sla = latest.get("served_sla")
reduction = latest.get("reduction_pct")
co2_saved = (baseline - served) if baseline and served else None

# -----------------------------
# KPI Row (Figma-like)
# -----------------------------
c1, c2, c3, c4, c5 = st.columns(5)

def kpi(col, title, big, sub, accent=None):
    color = f"color:{accent};" if accent else ""
    col.markdown(f"""
    <div class="card">
      <div class="kpi-title">{title}</div>
      <div class="kpi-big" style="{color}">{big}</div>
      <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

kpi(c1, "Carbon Reduction", f"{fmt(reduction, 2)}%", "vs. baseline scheduling", "var(--green)")
kpi(c2, "Total CO₂ Saved", fmt(co2_saved, 2), "baseline − served (relative units)", None)
kpi(c3, "SLA Violations", f"{sla if sla is not None else '—'}", "Hard constrained", None)
kpi(c4, "SLA Adherence", "100%" if sla == 0 else "Check", "All jobs on-time", None)

mix_txt = "—"
if latest.get("eval_points"):
    last = latest["eval_points"][-1]
    acts = last.get("actions")
    if acts:
        total = acts["run"] + acts["delay"] + acts["batt"]
        if total > 0:
            mix_txt = f"Run {acts['run']/total*100:.0f}% | Delay {acts['delay']/total*100:.0f}% | Battery {acts['batt']/total*100:.0f}%"
kpi(c5, "Active Strategy Mix", mix_txt, "from last checkpoint", None)

st.write("")

# -----------------------------
# Tabs: Dashboard / How it works
# -----------------------------
tab_dash, tab_story = st.tabs(["📈 Dashboard", "🌿 How CARB Reduces Carbon"])


with tab_dash:
    # Main big chart across runs (THIS is now the big one)
    st.markdown("<div class='card'><div class='section-title'>Learning Across Runs</div><div class='section-sub'>Global best serving improves over runs (carbon ↓ or reduction % ↑)</div></div>", unsafe_allow_html=True)
    st.write("")

    # Filters (interactive, judge-friendly)
    colA, colB, colC = st.columns([1, 1, 1.2])
    metric = colA.selectbox("Main metric", ["Carbon Reduction (%)", "Served Carbon (lower is better)"], index=0)
    show_best_marker = colB.toggle("Mark global best", value=True)
    show_table = colC.toggle("Show runs table", value=False)

    metric_key = "reduction_pct" if metric.startswith("Carbon Reduction") else "served_carbon"
    # Ensure numeric + drop Nones
    df_plot = df_runs.copy()
    df_plot[metric_key] = pd.to_numeric(df_plot[metric_key], errors="coerce")
    df_plot = df_plot.dropna(subset=[metric_key])

    plot_runs_big(df_plot, metric=metric_key, show_best_marker=show_best_marker)

    # Small panels: single-run details
    st.write("")
    st.markdown("<div class='card'><div class='section-title'>Single Run Details</div><div class='section-sub'>Choose a run to inspect learning checkpoints and action mix.</div></div>", unsafe_allow_html=True)
    st.write("")

    left, mid, right = st.columns([1.2, 1, 1])

    run_idx = left.selectbox("Select run", options=list(range(1, len(hist)+1)), index=len(hist)-1)
    sel = hist[run_idx-1]

    with mid:
        st.markdown("<div class='card'><div class='section-title'>Within-Run EVAL</div><div class='section-sub'>Real EVAL checkpoints (every 50 episodes)</div></div>", unsafe_allow_html=True)
        st.write("")
        plot_single_run_eval_small(sel)

    with right:
        st.markdown("<div class='card'><div class='section-title'>Action Mix</div><div class='section-sub'>Counts at last checkpoint</div></div>", unsafe_allow_html=True)
        st.write("")
        plot_action_mix_small(sel)

    # Promotion + serving clarity
    st.write("")
    promo = latest.get("promotion")
    loaded = latest.get("loaded_global_best")
    best_run = latest.get("best_run_carbon")

    st.markdown("<div class='card'><div class='section-title'>Global Best Serving</div><div class='section-sub'>Only promote if carbon is lower AND SLA=0.</div><hr/></div>", unsafe_allow_html=True)
    if loaded is not None:
        st.markdown(f"- Loaded previous global best: **{loaded:.2f}**")
    if best_run is not None:
        st.markdown(f"- Best within this run: **{best_run:.2f}** (SLA must be 0)")
    if promo:
        if promo["type"] == "new_global":
            st.success(f"✅ PROMOTED: New GLOBAL BEST = {promo['global_best']:.2f}")
        elif promo["type"] == "kept_global":
            st.info(f"ℹ️ Kept global best: {promo['global_best']:.2f} (run best: {promo['run_best']:.2f})")
        else:
            st.warning("⚠️ No valid best_run promoted.")
    else:
        st.info("Promotion message not detected in output (check training prints).")

    if show_table:
        st.write("")
        st.markdown("<div class='card'><div class='section-title'>Runs Table</div><div class='section-sub'>Export-ready history</div></div>", unsafe_allow_html=True)
        st.dataframe(df_runs[["run_index","timestamp","baseline_carbon","served_carbon","reduction_pct","served_sla","best_run_carbon"]], use_container_width=True)

    st.markdown("---")
    st.markdown("### 🎬 Live System Walkthrough")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.video("assets/carb_demo.mp4")

with tab_story:
    st.markdown("""
    <div class="card">
      <div class="section-title">How CARB Reduces Carbon Emissions</div>
      <div class="section-sub">A three-stage intelligent system that balances datacenter operations with environmental responsibility.</div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    s1, s2, s3 = st.columns(3)

    # ===== Stage 1 =====
    s1.markdown("""
    <div class="stage">
      <div class="top"
           style="
             background:
               linear-gradient(135deg, rgba(74,163,255,0.25), rgba(179,108,255,0.15)),
               url('assets/Forecast.png');
             background-size: cover;
             background-position: center;
           ">
      </div>

      <div class="num">01</div>
      <div class="icon">🌍</div>

      <div class="bottom">
        <h3>Forecast-Aware Signals</h3>
        <p>
          CARB uses your utilization predictor to build a forecast-aware RL state,
          enabling proactive decisions instead of reactive scheduling.
        </p>
        <div class="badge">Uses UtilPredictor1to1 (MindSpore)</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ===== Stage 2 =====
    s2.markdown("""
    <div class="stage">
      <div class="top"
           style="
             background:
               linear-gradient(135deg, rgba(74,163,255,0.16), rgba(179,108,255,0.10)),
               url('assets/data_center.png');
             background-size: cover;
             background-position: center;
           ">
      </div>

      <div class="num">02</div>
      <div class="icon">🧠</div>

      <div class="bottom">
        <h3>Intelligent Workload Scheduling</h3>
        <p>
          A DQN policy selects actions: run now, delay flexible workloads, or use battery.
          The objective is carbon minimization under safe actions.
        </p>
        <div class="badge">DQN + safe_actions constraint</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ===== Stage 3 =====
    s3.markdown("""
    <div class="stage">
      <div class="top"
           style="
             background:
               linear-gradient(135deg, rgba(46,230,168,0.18), rgba(255,255,255,0.06)),
               url('assets/sla.png');
             background-size: cover;
             background-position: center;
           ">
      </div>

      <div class="num">03</div>
      <div class="icon">🛡️</div>

      <div class="bottom">
        <h3>Zero SLA Violations</h3>
        <p>
          SLA is enforced as a hard constraint. CARB only serves policies that achieve
          lower carbon with zero SLA violations — legally compliant optimization.
        </p>
        <div class="badge">Global best serving (SLA = 0)</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.markdown("""
    <div class="card">
      <div class="section-title">What CARB Dashboard Proves</div>
      <div class="section-sub">
        This dashboard proves the AI is real: it trains, evaluates, promotes global-best policies,
        and improves carbon performance across runs while keeping SLA at zero.
      </div>
    </div>
    """, unsafe_allow_html=True)

