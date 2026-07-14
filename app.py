"""
PULSEGRID — Gig Economy Intelligence Platform
A premium, futuristic analytics & machine-learning cockpit for gig-worker
income data, built entirely with Streamlit + Plotly + scikit-learn.
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

warnings.filterwarnings("ignore")

APP_DIR = Path(__file__).parent
DATA_PATH = APP_DIR / "gig_worker_income_clean.csv"
MODEL_PATH = APP_DIR / "gig_worker_model.pkl"

# ════════════════════════════════════════════════════════════════════════
#  THEME TOKENS
# ════════════════════════════════════════════════════════════════════════
BG0 = "#05060B"
BG1 = "#0A0D1B"
BG2 = "#0F1428"
PANEL = "rgba(255,255,255,0.045)"
PANEL_HI = "rgba(255,255,255,0.07)"
BORDER = "rgba(255,255,255,0.09)"
BORDER_HI = "rgba(124,92,252,0.45)"
VIOLET = "#7C5CFC"
VIOLET_SOFT = "#A78BFA"
CYAN = "#22D3EE"
GOLD = "#FBBF24"
GREEN = "#34D399"
RED = "#FB7185"
TEXT = "#EAEBF9"
MUTED = "#8D93BE"
FAINT = "#5C6390"

COLORWAY = [VIOLET, CYAN, GOLD, GREEN, RED, VIOLET_SOFT, "#60A5FA", "#F472B6"]

CAT_COLS = ["city", "platform", "vehicle_type", "worker_type", "weather"]
NUM_COLS = ["age", "experience_years", "hours_worked", "orders_completed",
            "distance_km", "fuel_price", "customer_rating"]
BOOL_COLS = ["weekend", "festival"]
FEATURE_COLS = CAT_COLS + NUM_COLS + BOOL_COLS
TARGET = "net_income"

st.set_page_config(
    page_title="PulseGrid · Gig Economy Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ════════════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header[data-testid="stHeader"] {{background: transparent;}}

    .stApp {{
        background:
            radial-gradient(circle at 12% 8%, rgba(124,92,252,0.16) 0%, transparent 42%),
            radial-gradient(circle at 88% 4%, rgba(34,211,238,0.10) 0%, transparent 40%),
            radial-gradient(circle at 50% 100%, rgba(124,92,252,0.08) 0%, transparent 55%),
            linear-gradient(180deg, {BG0} 0%, {BG1} 45%, {BG2} 100%);
        background-attachment: fixed;
    }}

    h1, h2, h3, h4, .display-font {{
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.01em;
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(10,13,27,0.97) 0%, rgba(6,8,16,0.99) 100%);
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] * {{color: {TEXT};}}

    /* Tabs → premium pill nav */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 6px;
        backdrop-filter: blur(18px);
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 46px;
        border-radius: 12px;
        color: {MUTED};
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 0.86rem;
        letter-spacing: 0.04em;
        background: transparent;
        border: none;
        transition: all 0.25s ease;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, rgba(124,92,252,0.35), rgba(34,211,238,0.18));
        color: {TEXT} !important;
        box-shadow: 0 0 0 1px {BORDER_HI}, 0 8px 24px -8px rgba(124,92,252,0.6);
    }}
    .stTabs [data-baseweb="tab-highlight"] {{background: transparent;}}
    .stTabs [data-baseweb="tab-border"] {{background: transparent;}}

    /* Glass containers via st.container(key=...) */
    div[class*="st-key-panel_"] {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 18px;
        padding: 22px 24px 12px 24px;
        backdrop-filter: blur(18px);
        box-shadow: 0 4px 30px -12px rgba(0,0,0,0.5);
        animation: fadeInUp 0.6s ease both;
    }}
    div[class*="st-key-hero_"] {{
        border-radius: 24px;
        overflow: hidden;
        border: 1px solid {BORDER};
        margin-bottom: 4px;
    }}

    @keyframes fadeInUp {{
        from {{opacity: 0; transform: translateY(14px);}}
        to {{opacity: 1; transform: translateY(0);}}
    }}
    @keyframes pulseDot {{
        0% {{box-shadow: 0 0 0 0 rgba(52,211,153,0.65);}}
        70% {{box-shadow: 0 0 0 8px rgba(52,211,153,0);}}
        100% {{box-shadow: 0 0 0 0 rgba(52,211,153,0);}}
    }}
    @keyframes shimmerText {{
        0% {{background-position: 0% 50%;}}
        100% {{background-position: 200% 50%;}}
    }}

    .grad-text {{
        background: linear-gradient(90deg, {VIOLET_SOFT}, {CYAN}, {VIOLET_SOFT});
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shimmerText 6s linear infinite;
    }}

    .section-eyebrow {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.18em;
        color: {CYAN};
        text-transform: uppercase;
        margin-bottom: 2px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .section-eyebrow::before {{
        content: '';
        width: 7px; height: 7px; border-radius: 50%;
        background: {CYAN};
        box-shadow: 0 0 10px {CYAN};
    }}
    .section-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.28rem;
        font-weight: 600;
        color: {TEXT};
        margin: 0 0 14px 0;
    }}

    /* KPI cards */
    .kpi-card {{
        background: linear-gradient(160deg, rgba(255,255,255,0.055), rgba(255,255,255,0.015));
        border: 1px solid {BORDER};
        border-radius: 18px;
        padding: 18px 20px;
        position: relative;
        overflow: hidden;
        transition: transform 0.25s ease, border-color 0.25s ease;
        animation: fadeInUp 0.6s ease both;
        height: 132px;
    }}
    .kpi-card:hover {{
        transform: translateY(-3px);
        border-color: {BORDER_HI};
    }}
    .kpi-card::after {{
        content: '';
        position: absolute; top: -40%; right: -30%;
        width: 140px; height: 140px; border-radius: 50%;
        background: radial-gradient(circle, var(--glow, rgba(124,92,252,0.35)), transparent 70%);
        pointer-events: none;
    }}
    .kpi-label {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: {MUTED};
    }}
    .kpi-value {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.85rem;
        font-weight: 700;
        color: {TEXT};
        margin-top: 6px;
        line-height: 1.1;
    }}
    .kpi-delta {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        margin-top: 8px;
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 2px 8px;
        border-radius: 20px;
    }}
    .kpi-up {{color: {GREEN}; background: rgba(52,211,153,0.12);}}
    .kpi-down {{color: {RED}; background: rgba(251,113,133,0.12);}}
    .kpi-flat {{color: {GOLD}; background: rgba(251,191,36,0.12);}}

    .badge {{
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.08em;
        padding: 3px 10px;
        border-radius: 20px;
        border: 1px solid {BORDER};
        color: {MUTED};
        background: rgba(255,255,255,0.03);
    }}

    div[data-testid="stMetric"] {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 14px 16px;
    }}
    div[data-testid="stMetric"] label {{color: {MUTED} !important;}}

    .stButton>button, .stFormSubmitButton>button {{
        background: linear-gradient(135deg, {VIOLET}, #5B4CE6);
        color: white;
        border: none;
        border-radius: 12px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        letter-spacing: 0.03em;
        padding: 0.6rem 1.2rem;
        box-shadow: 0 8px 24px -10px rgba(124,92,252,0.75);
        transition: all 0.2s ease;
    }}
    .stButton>button:hover, .stFormSubmitButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 12px 28px -8px rgba(124,92,252,0.9);
    }}

    [data-testid="stDataFrame"] {{
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid {BORDER};
    }}

    hr {{border-color: {BORDER} !important;}}

    ::-webkit-scrollbar {{width: 9px; height: 9px;}}
    ::-webkit-scrollbar-track {{background: {BG0};}}
    ::-webkit-scrollbar-thumb {{background: #2A2F4A; border-radius: 10px;}}
    ::-webkit-scrollbar-thumb:hover {{background: {VIOLET};}}

    .insight-line {{
        display: flex; gap: 12px; align-items: flex-start;
        padding: 12px 0; border-bottom: 1px dashed {BORDER};
        font-size: 0.92rem; color: {TEXT}; line-height: 1.5;
    }}
    .insight-line:last-child {{border-bottom: none;}}
    .insight-tag {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem; color: {BG0};
        background: {CYAN}; padding: 3px 7px; border-radius: 6px;
        white-space: nowrap; margin-top: 2px; font-weight: 700;
    }}
    </style>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["weekend"] = df["weekend"].astype(bool)
    df["month"] = df["date"].dt.strftime("%b")
    df["month_num"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.day_name()
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["income_per_hour"] = (df["net_income"] / df["hours_worked"]).round(2)
    df["income_per_order"] = (df["net_income"] / df["orders_completed"]).round(2)
    df["cost_ratio"] = ((df["fuel_cost"] + df["maintenance_cost"]) / df["gross_income"]).round(4)
    return df


# ════════════════════════════════════════════════════════════════════════
#  MODEL — load if fitted, otherwise train a leakage-free pipeline
# ════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_model(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    X["weekend"] = X["weekend"].astype(int)
    X["festival"] = X["festival"].astype(int)
    y = df[TARGET]

    def build_pipeline():
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("num", StandardScaler(), NUM_COLS + BOOL_COLS),
        ])
        rf = RandomForestRegressor(
            n_estimators=260, max_depth=16, min_samples_leaf=2,
            n_jobs=-1, random_state=42,
        )
        return Pipeline([("pre", pre), ("rf", rf)])

    # attempt to reuse the shipped artifact if it is actually fitted
    loaded_ok = False
    pipe = None
    try:
        with open(MODEL_PATH, "rb") as f:
            candidate = pickle.load(f)

        check_is_fitted(candidate)
        pipe = candidate
        loaded_ok = True

    except Exception:
        loaded_ok = False

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if not loaded_ok:
        pipe = build_pipeline()
        pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, preds),
        "mae": mean_absolute_error(y_test, preds),
        "rmse": mean_squared_error(y_test, preds) ** 0.5,
    }

    # refit on the full dataset for best live predictions
    # Use the loaded/trained model directly
    final_pipe = pipe

# feature importance
    rf_final = final_pipe.named_steps["rf"]
    ohe = final_pipe.named_steps["pre"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CAT_COLS))
    all_names = cat_names + NUM_COLS + BOOL_COLS
    importance = pd.Series(rf_final.feature_importances_, index=all_names)

    grouped = {}
    for raw_col in FEATURE_COLS:
        cols = [n for n in all_names if n == raw_col or n.startswith(raw_col + "_")]
        grouped[raw_col] = importance[cols].sum()
    importance_df = (pd.Series(grouped).sort_values(ascending=False)
                      .rename_axis("feature").reset_index(name="importance"))

    return final_pipe, metrics, importance_df, loaded_ok


def predict_with_interval(pipe, row_df):
    X = row_df.copy()
    X["weekend"] = X["weekend"].astype(int)
    X["festival"] = X["festival"].astype(int)
    Xt = pipe.named_steps["pre"].transform(X)
    trees = pipe.named_steps["rf"].estimators_
    tree_preds = np.array([t.predict(Xt) for t in trees]).ravel()
    return float(np.mean(tree_preds)), float(np.percentile(tree_preds, 10)), float(np.percentile(tree_preds, 90))


# ════════════════════════════════════════════════════════════════════════
#  PLOTLY THEME
# ════════════════════════════════════════════════════════════════════════
def style_fig(fig, height=420, title=None, legend=True, scene=False):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT, size=12.5),
        colorway=COLORWAY,
        height=height,
        margin=dict(l=10, r=10, t=46 if title else 20, b=10),
        title=dict(text=title, font=dict(family="Space Grotesk, sans-serif", size=15, color=TEXT), x=0.01, y=0.97) if title else None,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=MUTED)) if legend else dict(),
        showlegend=legend,
        hoverlabel=dict(bgcolor=BG2, bordercolor=BORDER_HI, font=dict(family="Inter", color=TEXT, size=12)),
    )
    if not scene:
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)")
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)")
    else:
        axis = dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.08)",
                    color=MUTED, showbackground=True)
        fig.update_scenes(xaxis=axis, yaxis=axis, zaxis=axis)
    return fig


def panel_start(key, eyebrow=None, title=None):
    c = st.container(key=f"panel_{key}")
    with c:
        if eyebrow:
            st.markdown(f"<div class='section-eyebrow'>{eyebrow}</div>", unsafe_allow_html=True)
        if title:
            st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    return c


def kpi_card(label, value, delta=None, delta_dir="flat", glow=VIOLET, suffix=""):
    dir_class = {"up": "kpi-up", "down": "kpi-down", "flat": "kpi-flat"}[delta_dir]
    arrow = {"up": "▲", "down": "▼", "flat": "◆"}[delta_dir]
    delta_html = f"<div class='kpi-delta {dir_class}'>{arrow} {delta}</div>" if delta else ""
    glow_rgba = {
        VIOLET: "rgba(124,92,252,0.35)", CYAN: "rgba(34,211,238,0.3)",
        GOLD: "rgba(251,191,36,0.3)", GREEN: "rgba(52,211,153,0.3)", RED: "rgba(251,113,133,0.3)",
    }.get(glow, "rgba(124,92,252,0.35)")
    return f"""
    <div class="kpi-card" style="--glow:{glow_rgba};">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}<span style="font-size:0.95rem;color:{MUTED};">{suffix}</span></div>
        {delta_html}
    </div>
    """


def fmt_money(x):
    if abs(x) >= 1e7:
        return f"₹{x/1e7:.2f}Cr"
    if abs(x) >= 1e5:
        return f"₹{x/1e5:.2f}L"
    if abs(x) >= 1e3:
        return f"₹{x/1e3:.1f}K"
    return f"₹{x:.0f}"


def render_hero():
    c = st.container(key="hero_banner")
    with c:
        components.html(f"""
        <div style="width:100%;height:210px;position:relative;overflow:hidden;
             background:linear-gradient(135deg,{BG1} 0%,{BG2} 55%,#141a33 100%);
             font-family:'Space Grotesk',sans-serif;">
          <canvas id="net" style="position:absolute;inset:0;width:100%;height:100%;"></canvas>
          <div style="position:relative;z-index:2;height:100%;display:flex;flex-direction:column;
               justify-content:center;padding:0 42px;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:.22em;
                 color:{CYAN};display:flex;align-items:center;gap:8px;margin-bottom:10px;">
              <span style="width:8px;height:8px;border-radius:50%;background:{GREEN};
                    box-shadow:0 0 10px {GREEN};animation:pd 1.6s infinite;"></span>
              LIVE INTELLIGENCE FEED
            </div>
            <div style="font-size:40px;font-weight:700;letter-spacing:-0.02em;
                 background:linear-gradient(90deg,#fff,{VIOLET_SOFT} 45%,{CYAN});
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
              PULSEGRID
            </div>
            <div style="font-size:14.5px;color:#9aa1c9;margin-top:6px;max-width:560px;line-height:1.5;">
              Real-time earnings intelligence &amp; predictive modeling for gig-economy fleets
              across 10 cities and 5 platforms.
            </div>
          </div>
          <style>@keyframes pd{{0%{{box-shadow:0 0 0 0 rgba(52,211,153,.6)}}70%{{box-shadow:0 0 0 9px rgba(52,211,153,0)}}100%{{box-shadow:0 0 0 0 rgba(52,211,153,0)}}}}</style>
          <script>
          const cv = document.getElementById('net');
          const ctx = cv.getContext('2d');
          function size(){{cv.width = cv.clientWidth; cv.height = cv.clientHeight;}}
          size(); window.addEventListener('resize', size);
          const N = 46;
          const pts = Array.from({{length:N}}, () => ({{
              x: Math.random()*cv.width, y: Math.random()*cv.height,
              vx: (Math.random()-0.5)*0.35, vy: (Math.random()-0.5)*0.35
          }}));
          function tick(){{
              ctx.clearRect(0,0,cv.width,cv.height);
              for (const p of pts) {{
                  p.x += p.vx; p.y += p.vy;
                  if (p.x < 0 || p.x > cv.width) p.vx *= -1;
                  if (p.y < 0 || p.y > cv.height) p.vy *= -1;
              }}
              for (let i=0;i<N;i++) {{
                  for (let j=i+1;j<N;j++) {{
                      const dx = pts[i].x-pts[j].x, dy = pts[i].y-pts[j].y;
                      const d = Math.sqrt(dx*dx+dy*dy);
                      if (d < 130) {{
                          ctx.strokeStyle = 'rgba(124,92,252,' + (0.22*(1-d/130)) + ')';
                          ctx.lineWidth = 1;
                          ctx.beginPath(); ctx.moveTo(pts[i].x,pts[i].y); ctx.lineTo(pts[j].x,pts[j].y); ctx.stroke();
                      }}
                  }}
              }}
              for (const p of pts) {{
                  ctx.beginPath(); ctx.arc(p.x,p.y,1.6,0,7);
                  ctx.fillStyle = 'rgba(34,211,238,0.85)'; ctx.fill();
              }}
              requestAnimationFrame(tick);
          }}
          tick();
          </script>
        </div>
        """, height=212)


# ════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════
def sidebar_filters(df: pd.DataFrame):
    with st.sidebar:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;margin:2px 0 4px 0;">
            <div style="width:38px;height:38px;border-radius:11px;
                 background:linear-gradient(135deg,{VIOLET},{CYAN});
                 display:flex;align-items:center;justify-content:center;
                 font-family:'Space Grotesk';font-weight:700;color:white;font-size:17px;">◆</div>
            <div>
                <div style="font-family:'Space Grotesk';font-weight:700;font-size:17px;color:{TEXT};">PulseGrid</div>
                <div style="font-family:'JetBrains Mono';font-size:9.5px;letter-spacing:.14em;color:{MUTED};">GIG INTELLIGENCE OS</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<hr style='margin:10px 0 16px 0;'>", unsafe_allow_html=True)

        st.markdown("<div class='section-eyebrow'>DATA FILTERS</div>", unsafe_allow_html=True)

        min_d, max_d = df["date"].min().date(), df["date"].max().date()
        date_range = st.slider("Date range", min_value=min_d, max_value=max_d,
                                value=(min_d, max_d), format="DD MMM")

        cities = st.multiselect("Cities", sorted(df["city"].unique()), default=sorted(df["city"].unique()))
        platforms = st.multiselect("Platforms", sorted(df["platform"].unique()), default=sorted(df["platform"].unique()))

        col_a, col_b = st.columns(2)
        with col_a:
            vehicles = st.multiselect("Vehicle", sorted(df["vehicle_type"].unique()), default=sorted(df["vehicle_type"].unique()))
        with col_b:
            wtype = st.multiselect("Worker", sorted(df["worker_type"].unique()), default=sorted(df["worker_type"].unique()))

        c1, c2 = st.columns(2)
        with c1:
            weekend_only = st.checkbox("Weekends only")
        with c2:
            festival_only = st.checkbox("Festival days only")

        st.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)

        mask = (
            (df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1]) &
            (df["city"].isin(cities)) & (df["platform"].isin(platforms)) &
            (df["vehicle_type"].isin(vehicles)) & (df["worker_type"].isin(wtype))
        )
        if weekend_only:
            mask &= df["weekend"]
        if festival_only:
            mask &= df["festival"] == 1

        filtered = df[mask]

        st.markdown(f"""
        <div style="font-family:'JetBrains Mono';font-size:11px;color:{MUTED};line-height:1.8;">
        <span class="badge">{len(filtered):,} / {len(df):,} records</span><br><br>
        <span class="badge">{filtered['worker_id'].nunique()} active workers</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:11px;color:{FAINT};line-height:1.7;">
        PulseGrid v2.4 · Random-Forest core<br>
        Trained on {len(df):,} shift-level records<br>
        © 2026 PulseGrid Analytics
        </div>
        """, unsafe_allow_html=True)

        if len(filtered) == 0:
            st.error("No records match the current filters.")
            st.stop()

        return filtered


# ════════════════════════════════════════════════════════════════════════
#  PAGE — OVERVIEW
# ════════════════════════════════════════════════════════════════════════
def page_overview(df: pd.DataFrame, full_df: pd.DataFrame):
    dates_sorted = df["date"].sort_values().unique()
    mid = dates_sorted[len(dates_sorted) // 2] if len(dates_sorted) > 1 else dates_sorted[0]
    first_half = df[df["date"] < mid]
    second_half = df[df["date"] >= mid]

    def pct_delta(a, b):
        if a is None or b is None or a == 0 or pd.isna(a) or pd.isna(b):
            return None
        return (b - a) / a * 100

    total_income = df["net_income"].sum()
    avg_hourly = df["income_per_hour"].mean()
    total_orders = df["orders_completed"].sum()
    avg_rating = df["customer_rating"].mean()
    active_workers = df["worker_id"].nunique()
    avg_cancel = df["cancellation_rate"].mean()

    d_income = pct_delta(first_half["net_income"].sum(), second_half["net_income"].sum())
    d_hourly = pct_delta(first_half["income_per_hour"].mean(), second_half["income_per_hour"].mean())
    d_orders = pct_delta(first_half["orders_completed"].sum(), second_half["orders_completed"].sum())
    d_rating = pct_delta(first_half["customer_rating"].mean(), second_half["customer_rating"].mean())

    def dir_of(v):
        if v is None:
            return "flat"
        return "up" if v > 0.4 else ("down" if v < -0.4 else "flat")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.markdown(kpi_card("Total Net Income", fmt_money(total_income),
                              f"{d_income:+.1f}% vs prior period" if d_income is not None else None,
                              dir_of(d_income), VIOLET), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_card("Avg ₹ / Hour", f"₹{avg_hourly:,.0f}",
                              f"{d_hourly:+.1f}%" if d_hourly is not None else None,
                              dir_of(d_hourly), CYAN), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_card("Orders Completed", f"{total_orders:,.0f}",
                              f"{d_orders:+.1f}%" if d_orders is not None else None,
                              dir_of(d_orders), GOLD), unsafe_allow_html=True)
    with k4:
        st.markdown(kpi_card("Active Workers", f"{active_workers:,}",
                              f"of {full_df['worker_id'].nunique()} total", "flat", VIOLET_SOFT), unsafe_allow_html=True)
    with k5:
        st.markdown(kpi_card("Avg Rating", f"{avg_rating:.2f}",
                              f"{d_rating:+.1f}%" if d_rating is not None else None,
                              dir_of(d_rating), GREEN, suffix=" / 5"), unsafe_allow_html=True)
    with k6:
        st.markdown(kpi_card("Cancellation Rate", f"{avg_cancel:.2f}",
                              "lower is better", "flat" if avg_cancel < 6 else "down", RED, suffix="%"), unsafe_allow_html=True)

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    left, right = st.columns([2, 1])
    with left:
        panel_start("trend", "TIME SERIES", "Net Income Trend")
        daily = df.groupby("date").agg(net_income=("net_income", "sum"),
                                        orders=("orders_completed", "sum"),
                                        festival=("festival", "max")).reset_index()
        daily["rolling"] = daily["net_income"].rolling(7, min_periods=1).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["net_income"], mode="lines",
                                  line=dict(color="rgba(124,92,252,0.35)", width=1),
                                  fill="tozeroy", fillcolor="rgba(124,92,252,0.12)",
                                  name="Daily net income", hovertemplate="%{x|%d %b}<br>₹%{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["rolling"], mode="lines",
                                  line=dict(color=CYAN, width=2.6),
                                  name="7-day average", hovertemplate="%{x|%d %b}<br>₹%{y:,.0f} (avg)<extra></extra>"))
        fest = daily[daily["festival"] == 1]
        if len(fest):
            fig.add_trace(go.Scatter(x=fest["date"], y=fest["net_income"], mode="markers",
                                      marker=dict(color=GOLD, size=7, symbol="diamond",
                                                  line=dict(color=BG0, width=1)),
                                      name="Festival day"))
        style_fig(fig, height=360)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        panel_start("mix", "COMPOSITION", "Platform Market Share")
        mix = df.groupby("platform")["net_income"].sum().sort_values(ascending=False).reset_index()
        fig = go.Figure(go.Pie(labels=mix["platform"], values=mix["net_income"], hole=0.62,
                                marker=dict(colors=COLORWAY, line=dict(color=BG1, width=2)),
                                textfont=dict(color=TEXT, size=11.5), textinfo="label+percent"))
        fig.add_annotation(text=f"{fmt_money(mix['net_income'].sum())}<br><span style='font-size:10px;color:{MUTED}'>Total</span>",
                            showarrow=False, font=dict(size=15, color=TEXT, family="Space Grotesk"))
        style_fig(fig, height=360, legend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    c1, c2 = st.columns([1.3, 1])
    with c1:
        panel_start("city_bar", "GEOGRAPHY", "Net Income by City")
        city_g = df.groupby("city")["net_income"].sum().sort_values().reset_index()
        fig = go.Figure(go.Bar(
            x=city_g["net_income"], y=city_g["city"], orientation="h",
            marker=dict(color=city_g["net_income"], colorscale=[[0, "#3B2E8A"], [1, CYAN]],
                        line=dict(width=0)),
            text=[fmt_money(v) for v in city_g["net_income"]], textposition="outside",
            textfont=dict(color=MUTED, size=11),
            hovertemplate="%{y}<br>₹%{x:,.0f}<extra></extra>",
        ))
        style_fig(fig, height=360, legend=False)
        fig.update_xaxes(showticklabels=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with c2:
        panel_start("vehicle", "FLEET MIX", "Vehicle Type Distribution")
        veh = df.groupby("vehicle_type").agg(count=("worker_id", "count"),
                                               income=("net_income", "mean")).reset_index()
        fig = go.Figure(go.Bar(
            x=veh["vehicle_type"], y=veh["income"],
            marker=dict(color=[VIOLET, CYAN, GOLD][:len(veh)],
                        line=dict(color=BORDER_HI, width=1)),
            text=[f"₹{v:,.0f}" for v in veh["income"]], textposition="outside",
            textfont=dict(color=MUTED, size=11),
            hovertemplate="%{x}<br>Avg net ₹%{y:,.0f}<extra></extra>",
        ))
        style_fig(fig, height=360, legend=False, title=None)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    panel_start("galaxy", "3D VISUALIZATION", "Performance Galaxy — Hours × Distance × Income")
    sample = df.sample(min(2500, len(df)), random_state=7)
    fig = go.Figure(data=[go.Scatter3d(
        x=sample["hours_worked"], y=sample["distance_km"], z=sample["net_income"],
        mode="markers",
        marker=dict(
            size=(sample["orders_completed"] / sample["orders_completed"].max() * 9 + 2.5),
            color=sample["customer_rating"], colorscale=[[0, RED], [0.5, GOLD], [1, GREEN]],
            colorbar=dict(title=dict(text="Rating", font=dict(color=MUTED, size=10)),
                           tickfont=dict(color=MUTED, size=9), thickness=12, len=0.6),
            opacity=0.82, line=dict(width=0),
        ),
        hovertemplate="Hours: %{x}<br>Distance: %{y} km<br>Net income: ₹%{z:,.0f}<extra></extra>",
    )])
    fig.update_layout(scene=dict(
        xaxis_title="Hours Worked", yaxis_title="Distance (km)", zaxis_title="Net Income (₹)",
        camera=dict(eye=dict(x=1.55, y=1.55, z=0.9)),
    ))
    style_fig(fig, height=560, legend=False, scene=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ════════════════════════════════════════════════════════════════════════
#  PAGE — EXPLORE
# ════════════════════════════════════════════════════════════════════════
def page_explore(df: pd.DataFrame):
    panel_start("explore_controls", "INTERACTIVE EXPLORATION", "Build Your Own Scatter")
    c1, c2, c3, c4 = st.columns(4)
    numeric_options = ["hours_worked", "orders_completed", "distance_km", "net_income",
                        "gross_income", "tips", "bonus", "customer_rating", "cancellation_rate",
                        "income_per_hour", "age", "experience_years"]
    with c1:
        x_axis = st.selectbox("X axis", numeric_options, index=0)
    with c2:
        y_axis = st.selectbox("Y axis", numeric_options, index=3)
    with c3:
        color_by = st.selectbox("Color by", CAT_COLS + ["customer_rating"], index=1)
    with c4:
        size_by = st.selectbox("Size by", ["(none)"] + numeric_options, index=0)

    sample = df.sample(min(4000, len(df)), random_state=3)
    fig = px.scatter(
        sample, x=x_axis, y=y_axis, color=color_by,
        size=None if size_by == "(none)" else size_by,
        opacity=0.72, color_discrete_sequence=COLORWAY,
        color_continuous_scale=[[0, RED], [0.5, GOLD], [1, GREEN]] if color_by == "customer_rating" else None,
    )
    fig.update_traces(marker=dict(line=dict(width=0)))
    style_fig(fig, height=440)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    c1, c2 = st.columns(2)
    with c1:
        panel_start("corr", "STATISTICS", "Correlation Matrix")
        corr_cols = ["hours_worked", "orders_completed", "distance_km", "gross_income", "fuel_cost",
                     "maintenance_cost", "platform_commission", "tips", "bonus", "customer_rating",
                     "cancellation_rate", "net_income"]
        corr = df[corr_cols].corr().round(2)
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale=[[0, RED], [0.5, "#141a33"], [1, CYAN]], zmid=0,
            text=corr.values, texttemplate="%{text}", textfont=dict(size=9, color=TEXT),
            colorbar=dict(thickness=12, tickfont=dict(color=MUTED, size=9)),
        ))
        style_fig(fig, height=440, legend=False)
        fig.update_xaxes(tickfont=dict(size=9))
        fig.update_yaxes(tickfont=dict(size=9))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with c2:
        panel_start("dist", "DISTRIBUTIONS", "Metric Distribution by Segment")
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            metric = st.selectbox("Metric", numeric_options, index=3, key="dist_metric")
        with dcol2:
            segment = st.selectbox("Segment", CAT_COLS, index=1, key="dist_segment")
        fig = px.violin(df, x=segment, y=metric, color=segment, box=True, points=False,
                         color_discrete_sequence=COLORWAY)
        style_fig(fig, height=360, legend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    panel_start("raw", "RAW DATA", "Filtered Dataset")
    st.dataframe(df.sort_values("date", ascending=False).head(500), use_container_width=True, height=340,
                 hide_index=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download filtered CSV", csv, "pulsegrid_filtered.csv", "text/csv")


# ════════════════════════════════════════════════════════════════════════
#  PAGE — PREDICT
# ════════════════════════════════════════════════════════════════════════
def page_predict(df: pd.DataFrame, pipe, metrics, importance_df, loaded_ok):
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(kpi_card("Model R²", f"{metrics['r2']:.3f}", "Random Forest Regressor", "flat", VIOLET), unsafe_allow_html=True)
    with m2:
        st.markdown(kpi_card("Mean Abs. Error", f"₹{metrics['mae']:,.0f}", "avg deviation", "flat", CYAN), unsafe_allow_html=True)
    with m3:
        st.markdown(kpi_card("RMSE", f"₹{metrics['rmse']:,.0f}", "root mean sq. error", "flat", GOLD), unsafe_allow_html=True)
    with m4:
        st.markdown(kpi_card("Training Rows", f"{len(df):,}", "auto-trained" if not loaded_ok else "artifact loaded", "flat", GREEN), unsafe_allow_html=True)

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.15, 1])

    with left:
        panel_start("predict_form", "AI SHIFT SIMULATOR", "Predict Net Income")
        with st.form("predict_form"):
            f1, f2 = st.columns(2)
            with f1:
                city = st.selectbox("City", sorted(df["city"].unique()))
                platform = st.selectbox("Platform", sorted(df["platform"].unique()))
                vehicle_type = st.selectbox("Vehicle type", sorted(df["vehicle_type"].unique()))
                worker_type = st.selectbox("Worker type", sorted(df["worker_type"].unique()))
                weather = st.selectbox("Weather", sorted(df["weather"].unique()))
            with f2:
                age = st.slider("Age", 21, 55, 30)
                experience_years = st.slider("Experience (years)", 0.5, 15.0, 4.0, 0.1)
                hours_worked = st.slider("Planned hours worked", 3.0, 14.0, 8.0, 0.1)
                orders_completed = st.slider("Expected orders", 6, 57, 27)
                distance_km = st.slider("Expected distance (km)", 14.0, 310.0, 100.0, 1.0)

            f3, f4, f5 = st.columns(3)
            with f3:
                fuel_price = st.number_input("Fuel price (₹/L)", 95.0, 115.0, 104.5, 0.1)
            with f4:
                customer_rating = st.slider("Expected rating", 3.5, 5.0, 4.3, 0.1)
            with f5:
                st.write("")
                weekend = st.checkbox("Weekend shift")
                festival = st.checkbox("Festival day")

            submitted = st.form_submit_button("⚡ Run Prediction")

        if submitted:
            row = pd.DataFrame([{
                "city": city, "platform": platform, "vehicle_type": vehicle_type,
                "worker_type": worker_type, "weather": weather, "age": age,
                "experience_years": experience_years, "hours_worked": hours_worked,
                "orders_completed": orders_completed, "distance_km": distance_km,
                "fuel_price": fuel_price, "customer_rating": customer_rating,
                "weekend": weekend, "festival": festival,
            }])
            pred, lo, hi = predict_with_interval(pipe, row)
            st.session_state["last_pred"] = (pred, lo, hi, city, platform)

        if "last_pred" in st.session_state:
            pred, lo, hi, p_city, p_platform = st.session_state["last_pred"]
            avg_income = df["net_income"].mean()
            delta_pct = (pred - avg_income) / avg_income * 100

            gcol1, gcol2 = st.columns([1, 1])
            with gcol1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pred,
                    number=dict(prefix="₹", font=dict(size=30, color=TEXT, family="Space Grotesk")),
                    gauge=dict(
                        axis=dict(range=[0, max(df["net_income"].quantile(0.98), pred * 1.15)],
                                  tickfont=dict(color=MUTED, size=9)),
                        bar=dict(color=VIOLET, thickness=0.28),
                        bgcolor="rgba(255,255,255,0.03)",
                        borderwidth=0,
                        steps=[
                            dict(range=[0, avg_income], color="rgba(124,92,252,0.10)"),
                            dict(range=[avg_income, df["net_income"].quantile(0.98)], color="rgba(34,211,238,0.10)"),
                        ],
                        threshold=dict(line=dict(color=CYAN, width=3), thickness=0.8, value=avg_income),
                    ),
                ))
                style_fig(fig, height=260, legend=False)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            with gcol2:
                st.markdown(f"""
                <div style="padding-top:10px;">
                <div class="kpi-label">PREDICTED NET INCOME</div>
                <div class="kpi-value" style="font-size:2.1rem;">₹{pred:,.0f}</div>
                <div class="kpi-delta {'kpi-up' if delta_pct>=0 else 'kpi-down'}" style="margin-top:10px;">
                    {'▲' if delta_pct>=0 else '▼'} {delta_pct:+.1f}% vs dataset average
                </div>
                <div style="margin-top:16px;font-family:'JetBrains Mono';font-size:11.5px;color:{MUTED};line-height:1.9;">
                    80% confidence band<br>
                    <span style="color:{TEXT};font-size:14px;">₹{lo:,.0f} — ₹{hi:,.0f}</span>
                </div>
                </div>
                """, unsafe_allow_html=True)

    with right:
        panel_start("importance", "MODEL INTELLIGENCE", "What Drives Net Income")
        top_imp = importance_df.head(9).sort_values("importance")
        fig = go.Figure(go.Bar(
            x=top_imp["importance"], y=top_imp["feature"], orientation="h",
            marker=dict(color=top_imp["importance"], colorscale=[[0, "#3B2E8A"], [1, VIOLET]]),
            text=[f"{v*100:.1f}%" for v in top_imp["importance"]], textposition="outside",
            textfont=dict(color=MUTED, size=10.5),
        ))
        style_fig(fig, height=330, legend=False)
        fig.update_xaxes(showticklabels=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        if "last_pred" in st.session_state:
            pred, lo, hi, p_city, p_platform = st.session_state["last_pred"]
            panel_start("compare", "BENCHMARK", "Prediction vs Segment Averages")
            city_avg = df[df["city"] == p_city]["net_income"].mean()
            plat_avg = df[df["platform"] == p_platform]["net_income"].mean()
            overall_avg = df["net_income"].mean()
            comp = pd.DataFrame({
                "segment": ["Your prediction", f"{p_city} avg", f"{p_platform} avg", "Overall avg"],
                "value": [pred, city_avg, plat_avg, overall_avg],
            })
            fig = go.Figure(go.Bar(
                x=comp["segment"], y=comp["value"],
                marker=dict(color=[VIOLET, CYAN, GOLD, MUTED]),
                text=[f"₹{v:,.0f}" for v in comp["value"]], textposition="outside",
                textfont=dict(color=MUTED, size=10.5),
            ))
            style_fig(fig, height=300, legend=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ════════════════════════════════════════════════════════════════════════
#  PAGE — INSIGHTS
# ════════════════════════════════════════════════════════════════════════
def page_insights(df: pd.DataFrame):
    left, right = st.columns(2)

    with left:
        panel_start("leader_city", "LEADERBOARD", "Top Cities by ₹/Hour")
        rank = df.groupby("city")["income_per_hour"].mean().sort_values(ascending=False).reset_index()
        fig = go.Figure(go.Bar(
            x=rank["city"], y=rank["income_per_hour"],
            marker=dict(color=rank["income_per_hour"], colorscale=[[0, "#3B2E8A"], [1, VIOLET]]),
            text=[f"₹{v:,.0f}" for v in rank["income_per_hour"]], textposition="outside",
            textfont=dict(color=MUTED, size=10.5),
        ))
        style_fig(fig, height=320, legend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        panel_start("radar", "MULTI-METRIC VIEW", "Platform Comparison Radar")
        radar_cols = ["net_income", "customer_rating", "tips", "cancellation_rate"]
        agg = df.groupby("platform")[radar_cols].mean()
        norm = (agg - agg.min()) / (agg.max() - agg.min() + 1e-9)
        norm["cancellation_rate"] = 1 - norm["cancellation_rate"]
        labels = ["Net Income", "Rating", "Tips", "Reliability"]
        fig = go.Figure()
        for i, plat in enumerate(norm.index):
            vals = norm.loc[plat].tolist()
            fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=labels + [labels[0]],
                fill="toself", name=plat, opacity=0.55,
                line=dict(color=COLORWAY[i % len(COLORWAY)]),
            ))
        fig.update_layout(polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, showticklabels=False, gridcolor="rgba(255,255,255,0.08)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.08)", color=MUTED),
        ))
        style_fig(fig, height=320)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    panel_start("heatmap", "CROSS-TAB ANALYSIS", "Average Net Income — City × Platform")
    pivot = df.pivot_table(index="city", columns="platform", values="net_income", aggfunc="mean")
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale=[[0, "#141a33"], [0.5, VIOLET], [1, CYAN]],
        text=np.round(pivot.values, 0), texttemplate="₹%{text:,.0f}",
        textfont=dict(size=10, color=TEXT),
        colorbar=dict(thickness=12, tickfont=dict(color=MUTED, size=9)),
    ))
    style_fig(fig, height=380, legend=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    c1, c2 = st.columns(2)
    with c1:
        panel_start("surface", "3D SURFACE", "Income Surface — Experience × Hours")
        d = df.copy()
        d["exp_bucket"] = pd.cut(d["experience_years"], bins=8)
        d["hr_bucket"] = pd.cut(d["hours_worked"], bins=8)
        surf = d.pivot_table(index="exp_bucket", columns="hr_bucket", values="net_income",
                              aggfunc="mean", observed=False)
        x_labels = [round(iv.mid, 1) for iv in surf.columns]
        y_labels = [round(iv.mid, 1) for iv in surf.index]
        fig = go.Figure(data=[go.Surface(
            z=surf.values, x=x_labels, y=y_labels,
            colorscale=[[0, "#1e2650"], [0.5, VIOLET], [1, CYAN]],
            colorbar=dict(thickness=12, tickfont=dict(color=MUTED, size=9)),
        )])
        fig.update_layout(scene=dict(
            xaxis_title="Hours worked", yaxis_title="Experience (yrs)", zaxis_title="Avg net income",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.85)),
        ))
        style_fig(fig, height=440, legend=False, scene=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with c2:
        panel_start("segment3d", "3D VISUALIZATION", "Worker Segmentation")
        sample = df.sample(min(2200, len(df)), random_state=11)
        fig = go.Figure()
        for i, wt in enumerate(sample["worker_type"].unique()):
            sub = sample[sample["worker_type"] == wt]
            fig.add_trace(go.Scatter3d(
                x=sub["age"], y=sub["experience_years"], z=sub["net_income"],
                mode="markers", name=wt,
                marker=dict(size=3.2, color=COLORWAY[i % len(COLORWAY)], opacity=0.75),
                hovertemplate="Age: %{x}<br>Exp: %{y} yrs<br>Net: ₹%{z:,.0f}<extra></extra>",
            ))
        fig.update_layout(scene=dict(
            xaxis_title="Age", yaxis_title="Experience (yrs)", zaxis_title="Net income",
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.9)),
        ))
        style_fig(fig, height=440, scene=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    panel_start("auto_insights", "AUTO-GENERATED", "Key Business Insights")
    city_income = df.groupby("city")["income_per_hour"].mean()
    plat_income = df.groupby("platform")["net_income"].mean()
    veh_income = df.groupby("vehicle_type")["income_per_hour"].mean()
    best_city, worst_city = city_income.idxmax(), city_income.idxmin()
    best_plat = plat_income.idxmax()
    best_veh = veh_income.idxmax()
    weekend_lift = (df[df["weekend"]]["net_income"].mean() / df[~df["weekend"]]["net_income"].mean() - 1) * 100
    festival_lift = (df[df["festival"] == 1]["net_income"].mean() / df[df["festival"] == 0]["net_income"].mean() - 1) * 100
    rating_corr = df["customer_rating"].corr(df["net_income"])
    ft_income = df.groupby("worker_type")["income_per_hour"].mean()

    insights = [
        ("GEO", f"<b>{best_city}</b> leads all cities at ₹{city_income[best_city]:,.0f}/hour — "
                f"{(city_income[best_city]/city_income[worst_city]-1)*100:.0f}% higher than the lowest city, <b>{worst_city}</b>."),
        ("PLATFORM", f"<b>{best_plat}</b> generates the highest average net income per shift at "
                     f"{fmt_money(plat_income[best_plat])}, making it the top earning platform in the dataset."),
        ("FLEET", f"<b>{best_veh}</b> riders earn the most per hour on average (₹{veh_income[best_veh]:,.0f}/hr), "
                  f"suggesting vehicle choice materially affects efficiency."),
        ("TIMING", f"Weekend shifts pay <b>{weekend_lift:+.1f}%</b> {'more' if weekend_lift>=0 else 'less'} than weekdays, "
                   f"while festival days shift earnings by <b>{festival_lift:+.1f}%</b>."),
        ("QUALITY", f"Customer rating correlates at <b>{rating_corr:+.2f}</b> with net income — "
                    f"{'higher-rated shifts trend toward higher pay.' if rating_corr>0.05 else 'rating shows only a weak direct link to pay.'}"),
        ("WORKFORCE", f"<b>{ft_income.idxmax()}</b> workers out-earn <b>{ft_income.idxmin()}</b> workers by "
                      f"{(ft_income.max()/ft_income.min()-1)*100:.0f}% on a ₹/hour basis."),
    ]
    html = ""
    for tag, text in insights:
        html += f"<div class='insight-line'><span class='insight-tag'>{tag}</span><span>{text}</span></div>"
    st.markdown(html, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════
def main():
    inject_css()
    df = load_data()
    pipe, metrics, importance_df, loaded_ok = get_model(df)

    render_hero()
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    filtered = sidebar_filters(df)

    tab1, tab2, tab3, tab4 = st.tabs(["◆  OVERVIEW", "⬡  EXPLORE", "▲  PREDICT", "✦  INSIGHTS"])

    with tab1:
        page_overview(filtered, df)
    with tab2:
        page_explore(filtered)
    with tab3:
        page_predict(df, pipe, metrics, importance_df, loaded_ok)
    with tab4:
        page_insights(filtered)


if __name__ == "__main__":
    main()