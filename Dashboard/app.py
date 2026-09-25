"""
SentimentAI - Real-Time Sentiment Monitoring Dashboard
-------------------------------------------------------
Streamlit frontend for the EXISTING FastAPI sentiment backend.
UI layer only - backend contract unchanged:

    GET  /alerts
    GET  /analytics
    GET  /live?limit=50
    GET  /trends?minutes=180
    POST /predict
    GET  /model/performance

Run:
    streamlit run Dashboard/app.py
"""

import os
import time
from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="SentimentAI Dashboard",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# COLOUR PALETTE
# ============================================================

INK = "#050418"
GLASS = "rgba(255,255,255,0.045)"
GLASS_BORDER = "rgba(255,255,255,0.12)"
TEXT = "#ffffff"
MUTED = "#a6b1d6"

VIOLET = "#8b5cf6"
FUCHSIA = "#ec4899"
CYAN = "#22d3ee"
LIME = "#a3e635"
AMBER = "#fbbf24"
CORAL = "#fb7185"
MINT = "#34d399"
BLUE = "#3b82f6"

POS = "#22e08a"
NEG = "#ff4d6d"

RAINBOW = [
    VIOLET,
    CYAN,
    FUCHSIA,
    LIME,
    AMBER,
    CORAL,
    MINT,
    BLUE,
]

SENTIMENT_COLORS = {
    "Positive": POS,
    "Negative": NEG,
}


# ============================================================
# UI — COLOURFUL GLASS + AURORA BACKGROUND
# ============================================================

st.markdown(
    f"""
    <style>

      @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&display=swap');

      html, body, .stApp, [class*="css"] {{
          font-family: 'Outfit', system-ui, -apple-system, sans-serif;
      }}

      /* =====================================================
         ANIMATED AURORA BACKGROUND
         ===================================================== */

      @keyframes aurora {{
          0% {{
              background-position:
                  0% 50%,
                  100% 50%,
                  50% 0%,
                  0% 100%;
          }}

          50% {{
              background-position:
                  100% 50%,
                  0% 50%,
                  50% 100%,
                  100% 0%;
          }}

          100% {{
              background-position:
                  0% 50%,
                  100% 50%,
                  50% 0%,
                  0% 100%;
          }}
      }}

      .stApp {{
          background:
              radial-gradient(
                  900px 620px at 8% 4%,
                  rgba(139,92,246,.40),
                  transparent 62%
              ),
              radial-gradient(
                  880px 600px at 96% 8%,
                  rgba(34,211,238,.32),
                  transparent 62%
              ),
              radial-gradient(
                  820px 560px at 50% 100%,
                  rgba(236,72,153,.28),
                  transparent 62%
              ),
              radial-gradient(
                  760px 520px at 0% 92%,
                  rgba(163,230,53,.18),
                  transparent 62%
              ),
              {INK};

          background-size:
              200% 200%,
              200% 200%,
              200% 200%,
              200% 200%;

          animation: aurora 26s ease-in-out infinite;
          color: #e9edff;
      }}

      .block-container {{
          max-width: 1520px;
          padding-top: 1.1rem;
          padding-bottom: 3rem;
      }}

      #MainMenu,
      footer {{
          visibility: hidden;
      }}


      /* =====================================================
         PAGE ANIMATION
         ===================================================== */

      @keyframes riseIn {{
          from {{
              opacity: 0;
              transform: translateY(16px);
          }}

          to {{
              opacity: 1;
              transform: translateY(0);
          }}
      }}

      .block-container > div {{
          animation:
              riseIn .5s cubic-bezier(.22,.8,.3,1) both;
      }}


      /* =====================================================
         GLOW / SHIMMER
         ===================================================== */

      @keyframes pulseDot {{
          0% {{
              box-shadow:
                  0 0 0 0 rgba(34,224,138,.6);
          }}

          70% {{
              box-shadow:
                  0 0 0 10px rgba(34,224,138,0);
          }}

          100% {{
              box-shadow:
                  0 0 0 0 rgba(34,224,138,0);
          }}
      }}

      @keyframes shimmer {{
          0% {{
              background-position: 0% 50%;
          }}

          100% {{
              background-position: 200% 50%;
          }}
      }}


      /* =====================================================
         HEADINGS
         ===================================================== */

      h1 {{
          font-weight: 900 !important;
          letter-spacing: -.03em;

          background:
              linear-gradient(
                  100deg,
                  {VIOLET},
                  {CYAN},
                  {FUCHSIA},
                  {AMBER},
                  {VIOLET}
              );

          background-size: 200% auto;

          -webkit-background-clip: text;
          background-clip: text;

          -webkit-text-fill-color: transparent;

          animation:
              shimmer 7s linear infinite;
      }}

      h2,
      h3 {{
          color: {TEXT} !important;
          font-weight: 700 !important;
          letter-spacing: -.015em;
      }}

      p,
      label,
      .stCaption {{
          color: {MUTED};
      }}


      /* =====================================================
         SIDEBAR
         ===================================================== */

      section[data-testid="stSidebar"] {{
          background:
              linear-gradient(
                  185deg,
                  rgba(139,92,246,.20) 0%,
                  rgba(5,4,24,.92) 45%,
                  rgba(34,211,238,.14) 100%
              );

          border-right:
              1px solid {GLASS_BORDER};

          backdrop-filter: blur(18px);
      }}

      section[data-testid="stSidebar"] .block-container {{
          padding-top: 1.5rem;
      }}


      /* =====================================================
         BRAND
         ===================================================== */

      .brand {{
          display: flex;
          align-items: center;
          gap: 12px;
      }}

      .brand-badge {{
          width: 44px;
          height: 44px;
          border-radius: 14px;

          background:
              linear-gradient(
                  135deg,
                  {FUCHSIA},
                  {VIOLET} 45%,
                  {CYAN}
              );

          display: flex;
          align-items: center;
          justify-content: center;

          font-weight: 900;
          color: #fff;
          font-size: 19px;

          box-shadow:
              0 10px 26px rgba(139,92,246,.55);

          transition:
              transform .35s cubic-bezier(.22,.8,.3,1),
              box-shadow .35s ease;
      }}

      .brand:hover .brand-badge {{
          transform:
              rotate(-10deg)
              scale(1.10);

          box-shadow:
              0 14px 34px rgba(236,72,153,.65);
      }}

      .brand-name {{
          font-size: 1.15rem;
          font-weight: 900;
          line-height: 1.1;

          background:
              linear-gradient(
                  90deg,
                  #fff,
                  {CYAN}
              );

          -webkit-background-clip: text;
          background-clip: text;

          -webkit-text-fill-color: transparent;
      }}

      .brand-sub {{
          font-size: .72rem;
          color: {MUTED};
      }}


      /* =====================================================
         METRIC CARDS
         ===================================================== */

      div[data-testid="stMetric"] {{
          position: relative;
          overflow: hidden;

          background: {GLASS};

          border:
              1px solid {GLASS_BORDER};

          border-radius: 20px;

          padding:
              20px 22px 18px 22px;

          backdrop-filter: blur(14px);

          transition:
              transform .3s cubic-bezier(.22,.8,.3,1),
              box-shadow .3s ease,
              border-color .3s ease;
      }}

      div[data-testid="stMetric"]::before {{
          content: "";

          position: absolute;

          top: 0;
          left: 0;
          right: 0;

          height: 4px;

          background:
              linear-gradient(
                  90deg,
                  {VIOLET},
                  {CYAN},
                  {FUCHSIA}
              );

          background-size: 200% auto;

          animation:
              shimmer 5s linear infinite;
      }}

      div[data-testid="stMetric"]::after {{
          content: "";

          position: absolute;

          inset: 0;

          opacity: 0;

          background:
              radial-gradient(
                  520px 150px at 12% 0%,
                  rgba(139,92,246,.32),
                  transparent 70%
              );

          transition:
              opacity .35s ease;
      }}

      div[data-testid="stMetric"]:hover {{
          transform:
              translateY(-7px)
              scale(1.015);

          border-color:
              rgba(255,255,255,.28);

          box-shadow:
              0 22px 46px rgba(0,0,0,.55),
              0 0 32px rgba(139,92,246,.30);
      }}

      div[data-testid="stMetric"]:hover::after {{
          opacity: 1;
      }}

      div[data-testid="stMetricLabel"] {{
          color: {MUTED} !important;
          font-size: .74rem !important;
          font-weight: 700;
          letter-spacing: .10em;
          text-transform: uppercase;
      }}

      div[data-testid="stMetricValue"] {{
          color: {TEXT} !important;
          font-size: 2.0rem !important;
          font-weight: 900 !important;
          letter-spacing: -.02em;
      }}


      /* =====================================================
         PLOTLY GLASS
         ===================================================== */

      div[data-testid="stPlotlyChart"] {{
          background: {GLASS};

          border:
              1px solid {GLASS_BORDER};

          border-radius: 22px;

          padding: 10px;

          backdrop-filter: blur(14px);

          transition:
              transform .3s cubic-bezier(.22,.8,.3,1),
              box-shadow .3s ease,
              border-color .3s ease;
      }}

      div[data-testid="stPlotlyChart"]:hover {{
          transform:
              translateY(-6px);

          border-color:
              rgba(34,211,238,.45);

          box-shadow:
              0 24px 50px rgba(0,0,0,.55),
              0 0 36px rgba(34,211,238,.22);
      }}


      /* =====================================================
         KICKER
         ===================================================== */

      .kicker {{
          display: inline-block;

          font-size: .7rem;
          font-weight: 800;

          letter-spacing: .16em;

          text-transform: uppercase;

          color: #fff;

          padding: 5px 14px;

          border-radius: 999px;

          margin-bottom: .5rem;

          background:
              linear-gradient(
                  90deg,
                  {VIOLET},
                  {FUCHSIA},
                  {CYAN}
              );

          background-size: 200% auto;

          animation:
              shimmer 6s linear infinite;

          box-shadow:
              0 6px 18px rgba(139,92,246,.42);
      }}


      /* =====================================================
         BUTTONS
         ===================================================== */

      .stButton > button {{
          border-radius: 14px;

          border: 0;

          background:
              linear-gradient(
                  120deg,
                  {FUCHSIA},
                  {VIOLET} 50%,
                  {CYAN}
              );

          background-size: 200% auto;

          color: #fff;

          font-weight: 800;

          min-height: 44px;

          letter-spacing: .01em;

          box-shadow:
              0 8px 22px rgba(139,92,246,.40);

          transition:
              transform .22s ease,
              box-shadow .28s ease,
              background-position .5s ease;
      }}

      .stButton > button:hover {{
          background-position: right center;

          transform:
              translateY(-3px)
              scale(1.02);

          box-shadow:
              0 16px 34px rgba(236,72,153,.52);
      }}

      .stButton > button:active {{
          transform:
              translateY(0)
              scale(.98);
      }}


      /* =====================================================
         INPUTS
         ===================================================== */

      textarea,
      input {{
          color: {TEXT} !important;
      }}

      div[data-baseweb="textarea"] > div,
      div[data-baseweb="input"] > div {{
          background:
              rgba(255,255,255,.05) !important;

          border:
              1px solid {GLASS_BORDER} !important;

          border-radius:
              14px !important;

          transition:
              border-color .25s ease,
              box-shadow .25s ease;
      }}

      div[data-baseweb="textarea"] > div:hover,
      div[data-baseweb="input"] > div:hover,
      div[data-baseweb="textarea"] > div:focus-within,
      div[data-baseweb="input"] > div:focus-within {{
          border-color:
              rgba(34,211,238,.75) !important;

          box-shadow:
              0 0 0 4px rgba(34,211,238,.16);
      }}


      /* =====================================================
         SIDEBAR PIPELINE
         ===================================================== */

      .pipe-step {{
          display: flex;
          align-items: center;
          gap: 10px;

          border-radius: 12px;

          padding: 9px 12px;

          margin-bottom: 6px;

          font-size: .84rem;
          font-weight: 600;

          color: #eaf0ff;

          background:
              rgba(255,255,255,.05);

          border:
              1px solid {GLASS_BORDER};

          border-left:
              3px solid var(--accent, {VIOLET});

          transition:
              transform .25s ease,
              box-shadow .25s ease,
              background .25s ease;
      }}

      .pipe-step:hover {{
          transform:
              translateX(7px);

          background:
              rgba(255,255,255,.10);

          box-shadow:
              -6px 0 22px -6px
              var(--accent, {VIOLET});
      }}

      .pipe-arrow {{
          color:
              rgba(255,255,255,.28);

          font-size:
              .68rem;

          padding-left:
              18px;

          line-height:
              1.1;
      }}


      /* =====================================================
         BANNERS
         ===================================================== */

      .banner {{
          border-radius: 16px;

          padding: 14px 20px;

          font-weight: 700;

          font-size: .92rem;

          display: flex;

          align-items: center;

          gap: 10px;

          margin:
              .4rem 0 1.2rem 0;

          backdrop-filter:
              blur(10px);

          transition:
              transform .26s ease,
              box-shadow .26s ease;
      }}

      .banner:hover {{
          transform:
              translateY(-3px);

          box-shadow:
              0 16px 34px rgba(0,0,0,.45);
      }}

      .banner-ok {{
          background:
              linear-gradient(
                  90deg,
                  rgba(34,224,138,.20),
                  rgba(163,230,53,.12)
              );

          border:
              1px solid rgba(34,224,138,.45);

          color:
              #bbf7d0;
      }}

      .banner-bad {{
          background:
              linear-gradient(
                  90deg,
                  rgba(255,77,109,.22),
                  rgba(251,191,36,.12)
              );

          border:
              1px solid rgba(255,77,109,.50);

          color:
              #ffd1d9;
      }}

      .banner-info {{
          background:
              linear-gradient(
                  90deg,
                  rgba(139,92,246,.22),
                  rgba(34,211,238,.14)
              );

          border:
              1px solid rgba(139,92,246,.48);

          color:
              #ddd6fe;
      }}


      /* =====================================================
         STATUS
         ===================================================== */

      .status-line {{
          display: inline-flex;

          align-items: center;

          gap: 9px;

          color: #bbf7d0;

          font-size: .8rem;

          font-weight: 800;

          background:
              linear-gradient(
                  90deg,
                  rgba(34,224,138,.20),
                  rgba(34,211,238,.12)
              );

          border:
              1px solid rgba(34,224,138,.42);

          padding:
              6px 14px;

          border-radius:
              999px;
      }}

      .status-dot {{
          width: 9px;
          height: 9px;

          border-radius: 50%;

          background:
              {POS};

          display: inline-block;

          animation:
              pulseDot 2s infinite;
      }}


      /* =====================================================
         PREDICTION CHIPS
         ===================================================== */

      .chip {{
          display: inline-flex;

          align-items: center;

          gap: 10px;

          font-weight: 900;

          font-size: 1.08rem;

          padding:
              11px 22px;

          border-radius:
              999px;

          color:
              #050418;

          animation:
              riseIn .45s
              cubic-bezier(.22,.8,.3,1)
              both;
      }}

      .chip-pos {{
          background:
              linear-gradient(
                  120deg,
                  {POS},
                  {LIME}
              );

          box-shadow:
              0 10px 28px
              rgba(34,224,138,.45);
      }}

      .chip-neg {{
          background:
              linear-gradient(
                  120deg,
                  {NEG},
                  {AMBER}
              );

          box-shadow:
              0 10px 28px
              rgba(255,77,109,.45);
      }}


      /* =====================================================
         PROGRESS
         ===================================================== */

      div[data-testid="stProgress"] > div > div > div {{
          background:
              linear-gradient(
                  90deg,
                  {FUCHSIA},
                  {VIOLET},
                  {CYAN}
              ) !important;
      }}


      /* =====================================================
         DATAFRAME
         ===================================================== */

      div[data-testid="stDataFrame"] {{
          border:
              1px solid {GLASS_BORDER};

          border-radius:
              18px;

          overflow:
              hidden;

          transition:
              border-color .25s ease,
              box-shadow .25s ease;
      }}

      div[data-testid="stDataFrame"]:hover {{
          border-color:
              rgba(236,72,153,.45);

          box-shadow:
              0 0 30px
              rgba(236,72,153,.18);
      }}


      /* =====================================================
         TABS / MISC
         ===================================================== */

      button[data-baseweb="tab"] {{
          color:
              {MUTED};

          transition:
              color .2s ease;
      }}

      button[data-baseweb="tab"]:hover {{
          color:
              {CYAN};
      }}

      button[data-baseweb="tab"][aria-selected="true"] {{
          color:
              {FUCHSIA};
      }}

      hr {{
          border-color:
              rgba(255,255,255,.10) !important;
      }}


      /* =====================================================
         FOOTER
         ===================================================== */

      .footer {{
          text-align:
              center;

          font-size:
              .78rem;

          font-weight:
              600;

          padding-top:
              2rem;

          background:
              linear-gradient(
                  90deg,
                  {VIOLET},
                  {CYAN},
                  {FUCHSIA}
              );

          background-size:
              200% auto;

          animation:
              shimmer 8s linear infinite;

          -webkit-background-clip:
              text;

          background-clip:
              text;

          -webkit-text-fill-color:
              transparent;
      }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API HELPERS
# ============================================================

def api_get(
    path: str,
    **params: Any
) -> Optional[Dict[str, Any]]:
    """GET helper for the existing FastAPI backend."""

    try:
        response = requests.get(
            f"{API}{path}",
            params=params,
            timeout=10,
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException as exc:

        st.error(
            f"Backend connection error: {exc}"
        )

        return None

    except ValueError as exc:

        st.error(
            f"Invalid response from backend: {exc}"
        )

        return None


def api_post(
    path: str,
    payload: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """POST helper for the existing FastAPI backend."""

    try:
        response = requests.post(
            f"{API}{path}",
            json=payload,
            timeout=15,
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException as exc:

        st.error(
            f"Backend request failed: {exc}"
        )

        return None

    except ValueError as exc:

        st.error(
            f"Invalid response from backend: {exc}"
        )

        return None


# ============================================================
# PLOTLY THEME
# ============================================================

def plot_layout(
    fig: go.Figure,
    title: str = ""
) -> go.Figure:

    fig.update_layout(

        title=dict(
            text=title,
            font=dict(
                size=16,
                color=TEXT
            ),
            x=0.015,
            xanchor="left",
        ),

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font=dict(
            family="Outfit, sans-serif",
            color="#cfd7f5",
            size=12,
        ),

        margin=dict(
            l=18,
            r=18,
            t=56,
            b=18,
        ),

        hoverlabel=dict(
            bgcolor="rgba(10,8,34,.94)",
            bordercolor=CYAN,
            font=dict(
                color="#fff",
                family="Outfit",
            ),
        ),

        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(
                color="#cfd7f5"
            ),
        ),

        colorway=RAINBOW,

        transition=dict(
            duration=450,
            easing="cubic-in-out",
        ),
    )

    grid = "rgba(255,255,255,.08)"

    fig.update_xaxes(
        gridcolor=grid,
        zerolinecolor=grid,
        linecolor=grid,
    )

    fig.update_yaxes(
        gridcolor=grid,
        zerolinecolor=grid,
        linecolor=grid,
    )

    return fig


# ============================================================
# SECTION HELPER
# ============================================================

def section(
    kicker: str,
    title: str,
    caption: str = ""
) -> None:

    st.markdown(
        f'<div class="kicker">{kicker}</div>',
        unsafe_allow_html=True,
    )

    st.subheader(title)

    if caption:
        st.caption(caption)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="brand">'
        '<div class="brand-badge">S</div>'
        '<div>'
        '<div class="brand-name">SentimentAI</div>'
        '<div class="brand-sub">'
        'Real-Time Intelligence Platform'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown(
        '<div class="status-line">'
        '<span class="status-dot"></span>'
        'System Online'
        '</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Live Monitoring",
            "Analytics",
            "Model Performance",
        ],
    )

    st.divider()

    st.caption(
        "PROCESSING PIPELINE"
    )

    pipeline = [
        ("📡", "Kafka", FUCHSIA),
        ("⚙️", "Apache Spark", AMBER),
        ("🧠", "Spark MLlib", VIOLET),
        ("🚀", "FastAPI", CYAN),
        ("📊", "Dashboard", LIME),
    ]

    for index, (
        icon,
        name,
        color
    ) in enumerate(pipeline):

        st.markdown(
            f'<div class="pipe-step" '
            f'style="--accent:{color}">'
            f'<span>{icon}</span>'
            f'<span>{name}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if index < len(pipeline) - 1:

            st.markdown(
                '<div class="pipe-arrow">▼</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    st.caption("BACKEND")

    st.code(
        API,
        language="text"
    )

    st.caption(
        "Powered by Apache Spark + MLlib"
    )


# ============================================================
# TOP BAR / ALERT
# ============================================================

top_left, top_right = st.columns(
    [5, 1]
)

with top_left:

    st.markdown(
        "# Real-Time Sentiment Monitoring"
    )

    st.caption(
        "Monitor social-media sentiment, analyse incoming posts, "
        "and evaluate the machine-learning pipeline."
    )

with top_right:

    st.write("")

    if st.button(
        "↻  Refresh",
        use_container_width=True
    ):

        st.rerun()


# ============================================================
# ALERT
# ============================================================

alert_data = api_get(
    "/alerts"
)

if alert_data and alert_data.get("alert"):

    st.markdown(
        '<div class="banner banner-bad">'
        '🚨 Sentiment Alert — '
        f'{alert_data.get("message", "Unusual activity detected.")}'
        '</div>',
        unsafe_allow_html=True,
    )

else:

    st.markdown(
        '<div class="banner banner-ok">'
        '✨ Monitoring normal — no unusual '
        'sentiment activity detected.'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    analytics = api_get(
        "/analytics"
    )

    if analytics:

        counts_raw = analytics.get(
            "counts",
            {}
        )

        # ====================================================
        # STRICT BINARY FILTER
        # ====================================================

        positive_count = int(
            counts_raw.get(
                "positive",
                0
            ) or 0
        )

        negative_count = int(
            counts_raw.get(
                "negative",
                0
            ) or 0
        )

        total = (
            positive_count
            + negative_count
        )

        if total > 0:

            positive_percentage = round(
                positive_count / total * 100,
                2
            )

            negative_percentage = round(
                negative_count / total * 100,
                2
            )

        else:

            positive_percentage = 0
            negative_percentage = 0

        section(
            "Overview",
            "Current Sentiment Snapshot"
        )

        # ====================================================
        # ONLY 3 METRICS
        # ====================================================

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Total Posts",
            f"{total:,}"
        )

        c2.metric(
            "Positive",
            f"{positive_percentage}%"
        )

        c3.metric(
            "Negative",
            f"{negative_percentage}%"
        )

        st.write("")

        # ====================================================
        # BINARY DATAFRAME
        # ====================================================

        df = pd.DataFrame(
            {
                "Sentiment": [
                    "Positive",
                    "Negative"
                ],

                "Posts": [
                    positive_count,
                    negative_count
                ],
            }
        )

        left, right = st.columns(2)

        # ====================================================
        # DONUT
        # ====================================================

        with left:

            pie = px.pie(
                df,
                names="Sentiment",
                values="Posts",
                hole=0.60,
                color="Sentiment",
                color_discrete_map=SENTIMENT_COLORS,
            )

            pie.update_traces(

                textinfo="label+percent",

                textfont=dict(
                    color="#050418",
                    size=13,
                    family="Outfit",
                ),

                marker=dict(
                    line=dict(
                        color="rgba(5,4,24,.9)",
                        width=4,
                    )
                ),

                pull=[
                    0.035,
                    0.035
                ],

                hovertemplate=(
                    "<b>%{label}</b>"
                    "<br>Posts: %{value}"
                    "<br>Share: %{percent}"
                    "<extra></extra>"
                ),
            )

            pie.add_annotation(
                text=(
                    f"<b>{total:,}</b>"
                    "<br><span style='font-size:11px'>POSTS</span>"
                ),
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(
                    color=TEXT,
                    size=17,
                    family="Outfit",
                ),
            )

            st.plotly_chart(
                plot_layout(
                    pie,
                    "Sentiment Distribution"
                ),
                use_container_width=True,
            )

        # ====================================================
        # BAR CHART
        # ====================================================

        with right:

            bar = px.bar(
                df,
                x="Sentiment",
                y="Posts",
                color="Sentiment",
                color_discrete_map=SENTIMENT_COLORS,
                text="Posts",
            )

            bar.update_traces(
                marker_line_width=0,
                width=0.5,
                textposition="outside",
                textfont=dict(
                    color="#e9edff",
                    size=13,
                ),
                hovertemplate=(
                    "<b>%{x}</b>"
                    "<br>Posts: %{y}"
                    "<extra></extra>"
                ),
            )

            bar.update_layout(
                showlegend=False,
                bargap=0.45,
            )

            st.plotly_chart(
                plot_layout(
                    bar,
                    "Post Volume by Sentiment"
                ),
                use_container_width=True,
            )


    # ========================================================
    # AI ANALYZER
    # ========================================================

    st.divider()

    section(
        "AI Analysis",
        "Sentiment Analyzer",
        "Enter a message and send it to the trained model for prediction.",
    )

    text = st.text_area(
        "Message",
        value=(
            "The product quality is amazing!"
        ),
        height=110,
        placeholder=(
            "Type a social-media message..."
        ),
    )

    analyse_col, _ = st.columns(
        [1, 5]
    )

    with analyse_col:

        analyse = st.button(
            "✨ Analyse Sentiment",
            type="primary",
            use_container_width=True,
        )

    if analyse:

        if not text.strip():

            st.warning(
                "Please enter some text first."
            )

        else:

            with st.spinner(
                "Running sentiment prediction..."
            ):

                result = api_post(
                    "/predict",
                    {
                        "text": text,
                        "persist": False,
                    },
                )

            if result:

                sentiment = str(
                    result.get(
                        "sentiment",
                        "unknown"
                    )
                ).lower()

                confidence = float(
                    result.get(
                        "confidence",
                        0
                    )
                )

                # =================================================
                # STRICT BINARY RESULT
                # =================================================

                if sentiment not in [
                    "positive",
                    "negative"
                ]:

                    st.error(
                        "The model returned an unsupported sentiment label."
                    )

                else:

                    if sentiment == "positive":

                        chip_class = "chip-pos"
                        chip_icon = "😊"

                    else:

                        chip_class = "chip-neg"
                        chip_icon = "😠"

                    st.markdown(
                        f'<div class="chip {chip_class}">'
                        f'{chip_icon} '
                        f'Prediction: '
                        f'{sentiment.title()}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    st.write("")

                    st.progress(
                        min(
                            max(
                                confidence,
                                0.0
                            ),
                            1.0,
                        ),
                        text=(
                            "Model confidence: "
                            f"{confidence * 100:.2f}%"
                        ),
                    )

                    i1, i2, i3 = st.columns(3)

                    i1.metric(
                        "Language",
                        str(
                            result.get(
                                "language",
                                "en"
                            )
                        ),
                    )

                    hashtags = result.get(
                        "hashtags",
                        []
                    )

                    i2.metric(
                        "Hashtags",
                        (
                            ", ".join(
                                map(
                                    str,
                                    hashtags
                                )
                            )
                            if hashtags
                            else
                            "None detected"
                        ),
                    )

                    i3.metric(
                        "AI Engine",
                        "Spark MLlib"
                    )


# ============================================================
# LIVE MONITORING
# ============================================================

elif page == "Live Monitoring":

    section(
        "Live Data",
        "Live Sentiment Stream",
        "Recent posts received by the monitoring system.",
    )

    controls_left, _ = st.columns(
        [1, 4]
    )

    with controls_left:

        auto_refresh = st.toggle(
            "Auto-refresh",
            value=False
        )

    data = api_get(
        "/live",
        limit=50
    )

    if data and data.get("posts"):

        df = pd.DataFrame(
            data["posts"]
        )

        # ====================================================
        # STRICT BINARY FILTER
        # ====================================================

        if "sentiment" in df.columns:

            df["sentiment"] = (
                df["sentiment"]
                .astype(str)
                .str.lower()
            )

            df = df[
                df["sentiment"].isin(
                    [
                        "positive",
                        "negative"
                    ]
                )
            ]

            df["sentiment"] = (
                df["sentiment"]
                .str.upper()
            )

        wanted = [
            "timestamp",
            "source",
            "language",
            "sentiment",
            "confidence",
            "text",
        ]

        cols = [
            column
            for column in wanted
            if column in df.columns
        ]

        if "confidence" in df.columns:

            confidence_values = pd.to_numeric(
                df["confidence"],
                errors="coerce"
            )

            df["confidence"] = (
                confidence_values
                * 100
            ).round(2).astype(str) + "%"

        # ====================================================
        # BINARY SUMMARY
        # ====================================================

        if "sentiment" in df.columns:

            share = (
                df["sentiment"]
                .value_counts()
            )

            s1, s2 = st.columns(2)

            s1.metric(
                "Positive Posts",
                int(
                    share.get(
                        "POSITIVE",
                        0
                    )
                ),
            )

            s2.metric(
                "Negative Posts",
                int(
                    share.get(
                        "NEGATIVE",
                        0
                    )
                ),
            )

            st.write("")

        st.dataframe(
            df[cols],
            use_container_width=True,
            height=560,
            hide_index=True,
        )

        st.caption(
            f"{len(df)} recent binary sentiment posts detected"
        )

    else:

        st.markdown(
            '<div class="banner banner-info">'
            '⏳ Waiting for incoming posts…'
            '</div>',
            unsafe_allow_html=True,
        )

    if auto_refresh:

        time.sleep(5)

        st.rerun()


# ============================================================
# ANALYTICS
# ============================================================

elif page == "Analytics":

    section(
        "Analytics",
        "Sentiment Analytics",
        "Trends, topics, hashtags, and sentiment activity.",
    )

    analytics = api_get(
        "/analytics"
    )

    trends = api_get(
        "/trends",
        minutes=180
    )

    # ========================================================
    # BINARY SENTIMENT PIE
    # ========================================================

    if analytics:

        counts_raw = analytics.get(
            "counts",
            {}
        )

        positive_count = int(
            counts_raw.get(
                "positive",
                0
            ) or 0
        )

        negative_count = int(
            counts_raw.get(
                "negative",
                0
            ) or 0
        )

        df = pd.DataFrame(
            {
                "Sentiment": [
                    "Positive",
                    "Negative"
                ],

                "Posts": [
                    positive_count,
                    negative_count
                ],
            }
        )

        fig = px.pie(
            df,
            names="Sentiment",
            values="Posts",
            hole=0.60,
            color="Sentiment",
            color_discrete_map=SENTIMENT_COLORS,
        )

        fig.update_traces(
            textinfo="label+percent",

            textfont=dict(
                color="#050418",
                size=13,
                family="Outfit",
            ),

            marker=dict(
                line=dict(
                    color="rgba(5,4,24,.9)",
                    width=4
                )
            ),

            pull=[
                0.035,
                0.035
            ],
        )

        st.plotly_chart(
            plot_layout(
                fig,
                "Overall Sentiment Distribution"
            ),
            use_container_width=True,
        )


    # ========================================================
    # TIMELINE
    # ========================================================

    if trends:

        timeline = pd.DataFrame(
            trends.get(
                "sentiment_timeline",
                []
            )
        )

        if not timeline.empty:

            # STRICT BINARY FILTER

            timeline["sentiment"] = (
                timeline["sentiment"]
                .astype(str)
                .str.lower()
            )

            timeline = timeline[
                timeline["sentiment"].isin(
                    [
                        "positive",
                        "negative"
                    ]
                )
            ]

            if not timeline.empty:

                fig = px.area(
                    timeline,
                    x="minute",
                    y="count",
                    color="sentiment",
                    markers=True,
                    color_discrete_map={
                        "positive": POS,
                        "negative": NEG,
                    },
                )

                fig.update_traces(
                    line=dict(
                        width=3
                    ),

                    marker=dict(
                        size=7,
                        line=dict(
                            width=0
                        ),
                    ),

                    opacity=0.85,
                )

                st.plotly_chart(
                    plot_layout(
                        fig,
                        "Sentiment Activity — Last 180 Minutes"
                    ),
                    use_container_width=True,
                )


        # ====================================================
        # TOPICS + HASHTAGS
        # ====================================================

        keywords = pd.DataFrame(
            trends.get(
                "trending_topics",
                []
            )
        )

        hashtags = pd.DataFrame(
            trends.get(
                "hashtags",
                []
            )
        )

        left, right = st.columns(2)

        # ====================================================
        # TRENDING TOPICS
        # ====================================================

        with left:

            if not keywords.empty:

                fig = px.bar(
                    keywords.head(12),
                    x="count",
                    y="keyword",
                    orientation="h",
                    color="count",
                    text="count",
                    color_continuous_scale=[
                        VIOLET,
                        FUCHSIA,
                        AMBER
                    ],
                )

                fig.update_layout(
                    coloraxis_showscale=False,
                    yaxis=dict(
                        autorange="reversed"
                    ),
                )

                fig.update_traces(
                    textposition="outside",
                    textfont=dict(
                        color="#cfd7f5"
                    ),
                    marker_line_width=0,
                    hovertemplate=(
                        "<b>%{y}</b>"
                        "<br>Mentions: %{x}"
                        "<extra></extra>"
                    ),
                )

                st.plotly_chart(
                    plot_layout(
                        fig,
                        "🔥 Trending Topics"
                    ),
                    use_container_width=True,
                )

            else:

                st.markdown(
                    '<div class="banner banner-info">'
                    'No trending topics available.'
                    '</div>',
                    unsafe_allow_html=True,
                )


        # ====================================================
        # HASHTAGS
        # ====================================================

        with right:

            if not hashtags.empty:

                fig = px.bar(
                    hashtags.head(12),
                    x="count",
                    y="hashtag",
                    orientation="h",
                    color="count",
                    text="count",
                    color_continuous_scale=[
                        BLUE,
                        CYAN,
                        LIME
                    ],
                )

                fig.update_layout(
                    coloraxis_showscale=False,
                    yaxis=dict(
                        autorange="reversed"
                    ),
                )

                fig.update_traces(
                    textposition="outside",
                    textfont=dict(
                        color="#cfd7f5"
                    ),
                    marker_line_width=0,
                    hovertemplate=(
                        "<b>%{y}</b>"
                        "<br>Mentions: %{x}"
                        "<extra></extra>"
                    ),
                )

                st.plotly_chart(
                    plot_layout(
                        fig,
                        "＃ Top Hashtags"
                    ),
                    use_container_width=True,
                )

            else:

                st.markdown(
                    '<div class="banner banner-info">'
                    'No hashtags available.'
                    '</div>',
                    unsafe_allow_html=True,
                )


        # ====================================================
        # OPTIONAL WORD CLOUD
        # ====================================================

        if not keywords.empty:

            try:

                from wordcloud import WordCloud
                import matplotlib.pyplot as plt

                frequencies = dict(
                    zip(
                        keywords["keyword"],
                        keywords["count"]
                    )
                )

                wc = WordCloud(
                    width=1400,
                    height=450,
                    background_color=None,
                    mode="RGBA",
                    colormap="rainbow",
                )

                wc.generate_from_frequencies(
                    frequencies
                )

                fig, ax = plt.subplots(
                    figsize=(14, 4.5)
                )

                ax.imshow(
                    wc,
                    interpolation="bilinear"
                )

                ax.axis("off")

                fig.patch.set_alpha(0)

                st.subheader(
                    "Keyword Intelligence"
                )

                st.pyplot(
                    fig,
                    use_container_width=True,
                    transparent=True
                )

                plt.close(fig)

            except Exception:
                pass


# ============================================================
# MODEL PERFORMANCE
# ============================================================

else:

    section(
        "Machine Learning",
        "Model Performance",
        "Evaluation results returned by the existing FastAPI model endpoint.",
    )

    model_data = api_get(
        "/model/performance"
    )

    if model_data:

        best_model = model_data.get(
            "best_model",
            "Unknown"
        )

        st.markdown(
            f'<div class="banner banner-info">'
            f'🧠 Active AI Model: '
            f'<b>&nbsp;{best_model}</b>'
            f' &nbsp;·&nbsp; Engine: Spark MLlib'
            f'</div>',
            unsafe_allow_html=True,
        )

        rows = []

        for name, metrics in model_data.get(
            "models",
            {}
        ).items():

            rows.append(
                {
                    "Model": name,

                    "Accuracy": round(
                        metrics.get(
                            "accuracy",
                            0
                        ),
                        4,
                    ),

                    "Precision": round(
                        metrics.get(
                            "weightedPrecision",
                            0
                        ),
                        4,
                    ),

                    "Recall": round(
                        metrics.get(
                            "weightedRecall",
                            0
                        ),
                        4,
                    ),

                    "F1 Score": round(
                        metrics.get(
                            "f1",
                            0
                        ),
                        4,
                    ),

                    "Training Time (s)":
                        metrics.get(
                            "train_seconds",
                            0
                        ),
                }
            )

        df = pd.DataFrame(
            rows
        )

        if not df.empty:

            best_row = df[
                df["Model"] == best_model
            ]

            if not best_row.empty:

                row = best_row.iloc[0]

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Accuracy",
                    f"{row['Accuracy'] * 100:.2f}%"
                )

                c2.metric(
                    "Precision",
                    f"{row['Precision'] * 100:.2f}%"
                )

                c3.metric(
                    "Recall",
                    f"{row['Recall'] * 100:.2f}%"
                )

                c4.metric(
                    "F1 Score",
                    f"{row['F1 Score'] * 100:.2f}%"
                )

            st.write("")

            st.subheader(
                "Model Comparison"
            )

            display_df = df.copy()

            for column in [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
            ]:

                display_df[column] = (
                    (
                        display_df[column]
                        * 100
                    )
                    .round(2)
                    .astype(str)
                    + "%"
                )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )


            # =================================================
            # MODEL BAR CHART
            # =================================================

            chart_df = df.melt(
                id_vars="Model",

                value_vars=[
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score",
                ],

                var_name="Metric",

                value_name="Score",
            )

            fig = px.bar(
                chart_df,
                x="Model",
                y="Score",
                color="Metric",
                barmode="group",

                color_discrete_sequence=[
                    VIOLET,
                    CYAN,
                    FUCHSIA,
                    AMBER
                ],
            )

            fig.update_traces(
                marker_line_width=0
            )

            fig.update_yaxes(
                range=[
                    0,
                    1
                ],
                title="Score"
            )

            st.plotly_chart(
                plot_layout(
                    fig,
                    "Model Evaluation"
                ),
                use_container_width=True,
            )


            # =================================================
            # CONFUSION MATRIX
            # =================================================

            models = model_data.get(
                "models",
                {}
            )

            if best_model in models:

                cm = models[
                    best_model
                ].get(
                    "confusion_matrix",
                    []
                )

                # STRICTLY BINARY LABELS

                labels = [
                    "negative",
                    "positive"
                ]

                # Only display a confusion matrix if the
                # backend actually supplied valid binary data.

                if cm:

                    matrix = [
                        [0, 0],
                        [0, 0]
                    ]

                    valid_cm = False

                    for item in cm:

                        if (
                            isinstance(item, (list, tuple))
                            and len(item) >= 3
                        ):

                            i, j, count = item

                            try:

                                i = int(i)
                                j = int(j)

                                if (
                                    i in [0, 1]
                                    and j in [0, 1]
                                ):

                                    matrix[
                                        i
                                    ][
                                        j
                                    ] = count

                                    valid_cm = True

                            except (
                                ValueError,
                                TypeError
                            ):

                                pass

                    if valid_cm:

                        heatmap = go.Figure(
                            data=go.Heatmap(

                                z=matrix,

                                x=[
                                    "Negative",
                                    "Positive"
                                ],

                                y=[
                                    "Negative",
                                    "Positive"
                                ],

                                colorscale=[
                                    [
                                        0.0,
                                        "rgba(139,92,246,.18)"
                                    ],

                                    [
                                        0.5,
                                        FUCHSIA
                                    ],

                                    [
                                        1.0,
                                        AMBER
                                    ],
                                ],

                                showscale=False,

                                xgap=6,
                                ygap=6,

                                text=matrix,

                                texttemplate="%{text}",

                                textfont=dict(
                                    color="#ffffff",
                                    size=14,
                                ),

                                hovertemplate=(
                                    "Actual: %{y}"
                                    "<br>Predicted: %{x}"
                                    "<br>Count: %{z}"
                                    "<extra></extra>"
                                ),
                            )
                        )

                        heatmap.update_xaxes(
                            title="Predicted"
                        )

                        heatmap.update_yaxes(
                            title="Actual"
                        )

                        st.plotly_chart(
                            plot_layout(
                                heatmap,
                                "Confusion Matrix"
                            ),
                            use_container_width=True,
                        )

                else:

                    st.caption(
                        "Confusion matrix data is not available "
                        "in the current metrics.json."
                    )

    else:

        st.warning(
            "Model performance data is not available "
            "from the backend."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    '<div class="footer">'
    'SentimentAI · Apache Spark · Spark MLlib · '
    'FastAPI · Real-Time Sentiment Analytics'
    '</div>',
    unsafe_allow_html=True,
)