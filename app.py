"""Streamlit demo: rule-based reasoning over nuScenes Mini (JSON-only).

Run with:
    streamlit run app.py

The app is BEV-only — no camera image, no LiDAR rendering. Everything is
computed from the 13 JSON tables; no sensor files are ever read.

The right-hand panel renders the **dual-audience** output:
  - Driver HMI card  — instrument-cluster style, single plain-language line.
  - Engineer/OEM Trace card — 4-row structured grid with status indicators.
"""
from __future__ import annotations

import base64
import html
import io
import os

import matplotlib.pyplot as plt
import streamlit as st

from src.config import (
    DEFAULT_RULES_PATH,
    DEFAULT_TAXONOMY_PATH,
    Taxonomy,
    load_rules,
    load_taxonomy,
)
from src.pipeline import FrameResult, run_scene
from src.reasoning.narrate import (
    SIGNATURE_QUESTIONS,
    SignatureOutput,
    format_signature_output,
)
from src.store import NuMiniStore
from src.viz import plot_bev

DEFAULT_JSON_DIR = "./nuscenes-mini-JSON"


# =============================================================================
# Cached helpers
# =============================================================================


@st.cache_resource(show_spinner="Loading nuScenes JSON tables…")
def get_store(json_dir: str) -> NuMiniStore:
    return NuMiniStore(json_dir)


@st.cache_resource(show_spinner=False)
def get_taxonomy() -> Taxonomy:
    return load_taxonomy(DEFAULT_TAXONOMY_PATH)


@st.cache_resource(show_spinner=False)
def get_rules() -> dict:
    return load_rules(DEFAULT_RULES_PATH)


@st.cache_data(show_spinner="Computing scene…", max_entries=12)
def compute_scene_cached(json_dir: str, scene_token: str) -> list[FrameResult]:
    store = get_store(json_dir)
    return run_scene(store, scene_token, get_rules(), get_taxonomy())


# =============================================================================
# Styling — futurist HUD theme
# =============================================================================


_CSS = """
<style>
:root {
  --bg-0: #04070B;
  --bg-1: #0A1218;
  --bg-2: #0E1B26;
  --bg-3: #16242F;
  --border: #1F3441;
  --border-dim: #15242F;
  --text: #E6EEF2;
  --text-dim: #8FA3B0;
  --text-faint: #5A7280;
  --accent: #3FE0C5;
  --accent-dim: #25A691;
  --flag-bg: #3A2914;
  --flag-fg: #F0AD4E;
  --ok-fg: #5CB85C;
  --divider: #15242F;
  --glow: 0 0 18px rgba(63, 224, 197, 0.18);
}

/* Hide sidebar entirely */
[data-testid="stSidebar"], [data-testid="collapsedControl"] {
  display: none !important;
}

/* Make the page background match the futurist surface */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
  background: radial-gradient(1200px 700px at 20% -10%, #0A1A24 0%, var(--bg-0) 55%, #02050A 100%) !important;
  color: var(--text);
}

[data-testid="stHeader"] { background: transparent !important; }

.block-container {
  padding-top: 0.6rem !important;
  padding-bottom: 1.2rem !important;
  max-width: 1500px !important;
}

/* Streamlit's default text color */
.stMarkdown, .stMarkdown p, .stMarkdown span, .stCaption,
.stSelectbox label, .stSlider label, .stTextInput label, .stExpander {
  color: var(--text) !important;
}
.stCaption {
  color: var(--text-faint) !important;
  letter-spacing: 1.2px;
  text-transform: uppercase;
  font-size: 11px;
}

/* Inputs — futurist look */
.stSelectbox [data-baseweb="select"] > div,
.stTextInput input,
[data-baseweb="input"] {
  background: var(--bg-1) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
}
.stSlider [data-baseweb="slider"] {
  color: var(--accent) !important;
}
.stSlider [role="slider"] {
  background: var(--accent) !important;
  box-shadow: 0 0 8px rgba(63,224,197,0.55) !important;
}

/* ---------------- Hero (compact) ---------------- */
.hero {
  text-align: center;
  margin: 2px auto 12px auto;
  padding: 10px 22px 12px 22px;
  border: 1px solid var(--border-dim);
  border-radius: 12px;
  background: linear-gradient(180deg, rgba(63,224,197,0.04) 0%, rgba(4,7,11,0) 70%);
  position: relative;
  overflow: hidden;
}
.hero::before {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent 0%, var(--accent) 50%, transparent 100%);
  opacity: 0.55;
}
.hero h1 {
  margin: 0;
  font-size: 22px;
  font-weight: 300;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--text);
}
.hero h1 .prism {
  font-weight: 700;
  letter-spacing: 5px;
  background: linear-gradient(90deg, var(--accent) 0%, #6FE8D2 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-left: 8px;
  text-shadow: 0 0 12px rgba(63,224,197,0.25);
}
.hero .lede {
  margin: 6px auto 0 auto;
  max-width: 1000px;
  font-size: 13px;
  line-height: 1.45;
  color: var(--text);
  font-weight: 400;
}
.hero .lede .accent { color: var(--accent); font-weight: 600; }
.hero .lede p { margin: 2px 0; }
.hero .lede .small {
  color: var(--text-dim);
  font-size: 12px;
}

/* ---------------- Control strip (compact) ---------------- */
.controls-wrap {
  margin: 0 0 10px 0;
  padding: 8px 14px;
  border: 1px solid var(--border-dim);
  border-radius: 10px;
  background: linear-gradient(180deg, var(--bg-1) 0%, var(--bg-0) 100%);
}
.controls-wrap .controls-title {
  font-size: 10px;
  letter-spacing: 2.4px;
  color: var(--text-faint);
  text-transform: uppercase;
  margin-bottom: 2px;
}

/* Both wraps lock to the same fixed height so BEV and Reasoned Alert
   align exactly side-by-side. */
:root { --panel-height: 560px; }

/* ---------------- Reasoned Alert from PRISM layer (right column) ---- */
.alert-wrap {
  margin: 0 0 12px 0;
  padding: 14px 18px 16px 18px;
  border: 1px solid var(--border);
  border-radius: 12px;
  background:
    linear-gradient(160deg, var(--bg-2) 0%, var(--bg-1) 60%, var(--bg-0) 100%);
  box-shadow: 0 0 24px rgba(63,224,197,0.10);
  position: relative;
  height: var(--panel-height);
  display: flex;
  flex-direction: column;
  box-sizing: border-box;
}
.alert-wrap::before {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--accent), transparent);
  opacity: 0.65;
}
.alert-header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: 12px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--divider);
  gap: 12px;
}
.alert-header .title {
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 3.2px;
  color: var(--accent);
  text-transform: uppercase;
}
.alert-header .tag {
  font-size: 9.5px;
  letter-spacing: 2.4px;
  color: var(--text-faint);
  text-transform: uppercase;
  white-space: nowrap;
}
.alert-grid {
  display: flex;
  flex-direction: column;
  gap: 8px;
  flex: 1 1 auto;
  min-height: 0;
}
.alert-cell {
  background: var(--bg-1);
  border: 1px solid var(--border-dim);
  border-radius: 8px;
  padding: 8px 14px;
  display: flex;
  flex-direction: column;
  gap: 2px;
  position: relative;
  flex: 1 1 0;
  min-height: 0;
}
.alert-cell.reason-block { flex: 1.7 1 0; }
.alert-cell .field {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 2.6px;
  color: var(--accent);
  text-transform: uppercase;
}
.alert-cell .q {
  font-size: 10.5px;
  color: var(--text-faint);
  font-style: italic;
  line-height: 1.3;
  margin-bottom: 2px;
}
.alert-cell .v {
  font-size: 16px;
  font-weight: 600;
  color: var(--text);
  line-height: 1.4;
}
.alert-cell.reason-block .v {
  font-size: 14px;
  font-weight: 400;
  color: var(--text);
  line-height: 1.55;
  letter-spacing: 0.2px;
}
.alert-cell.risk-HIGH   .v { color: #FF7060; text-shadow: 0 0 12px rgba(255,112,96,0.35); }
.alert-cell.risk-MEDIUM .v { color: var(--flag-fg); text-shadow: 0 0 10px rgba(240,173,78,0.30); }
.alert-cell.risk-LOW    .v { color: var(--ok-fg);  text-shadow: 0 0 10px rgba(92,184,92,0.30); }
.alert-cell.risk-HIGH, .alert-cell.risk-MEDIUM, .alert-cell.risk-LOW {
  /* Risk value is a single word — display it on its own line */
}

/* ---------------- BEV wrap (locked height to match alert) ---------- */
.bev-wrap {
  padding: 6px;
  border: 1px solid var(--border-dim);
  border-radius: 10px;
  background: var(--bg-1);
  box-shadow: var(--glow);
  height: var(--panel-height);
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  box-sizing: border-box;
}
.bev-wrap img {
  display: block;
  margin: 0 auto;
  max-width: 100%;
  max-height: 100%;
  width: auto;
  height: auto;
  object-fit: contain;
}

/* ---------------- Footer ---------------- */
.prism-footer {
  text-align: center;
  margin-top: 10px;
  padding: 8px;
  font-size: 12px;
  color: var(--text-dim);
  border-top: 1px solid var(--border-dim);
}
.prism-footer .accent { color: var(--accent); font-weight: 600; }

/* Telemetry strip under the BEV (inline) */
.telemetry {
  margin-top: 6px;
  padding: 6px 10px;
  font-family: ui-monospace, "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace;
  font-size: 10.5px;
  color: var(--text-dim);
  letter-spacing: 1px;
  border: 1px solid var(--border-dim);
  border-radius: 5px;
  background: var(--bg-0);
}
.telemetry .lbl { color: var(--text-faint); }
.telemetry .v { color: var(--text); }
</style>
"""


# =============================================================================
# Section renderers
# =============================================================================


def _render_hero() -> str:
    return (
        '<div class="hero">'
        '<h1>ADAS Function Powered By <span class="prism">PRISM</span></h1>'
        '<div class="lede">'
        '<p><span class="accent">PRISM</span> bridges the gap between what an ADAS system knows and what humans understand.</p>'
        '<p class="small">For drivers, it builds trust through contextual explanations.</p>'
        '<p class="small">For engineers, it provides transparency through decision traces.</p>'
        '<p class="small">Together, these capabilities accelerate the adoption, validation, and usability of advanced driver assistance systems.</p>'
        "</div>"
        "</div>"
    )


def _fig_to_data_uri(fig, dpi: int = 130) -> str:
    """PNG-encode a matplotlib figure and return a data: URI.

    Used so we can embed the BEV inside our own `<div class="bev-wrap">`
    via a single st.markdown call — keeping the image as a real DOM
    descendant of the wrap so the height-locking CSS actually applies.
    """
    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", dpi=dpi,
        bbox_inches="tight", facecolor=fig.get_facecolor(),
    )
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _render_reasoned_alert(sig: SignatureOutput) -> str:
    """Render the 4-field 'Reasoned Alert from PRISM layer' panel."""
    rows = [
        ("CONTEXT", SIGNATURE_QUESTIONS["context"], sig.context, ""),
        ("RISK",    SIGNATURE_QUESTIONS["risk"],    sig.risk,    f"risk-{sig.risk}"),
        ("ACTION",  SIGNATURE_QUESTIONS["action"],  sig.action,  ""),
        ("REASON",  SIGNATURE_QUESTIONS["reason"],  sig.reason,  "reason-block"),
    ]
    cells = []
    for field_name, question, value, extra_class in rows:
        cls = f"alert-cell {extra_class}".strip()
        cells.append(
            f'<div class="{cls}">'
            f'<div class="field">{html.escape(field_name)}</div>'
            f'<div class="q">{html.escape(question)}</div>'
            f'<div class="v">{html.escape(value)}</div>'
            "</div>"
        )
    return (
        '<div class="alert-wrap">'
        '<div class="alert-header">'
        '<div class="title">◈ Reasoned Alert from PRISM layer</div>'
        '<div class="tag">Context · Risk · Action · Reason</div>'
        "</div>"
        f'<div class="alert-grid">{"".join(cells)}</div>'
        "</div>"
    )


def _render_telemetry(result: FrameResult, scene: dict, frame_idx: int, n_frames: int) -> str:
    return (
        '<div class="telemetry">'
        f'<span class="lbl">SCENE</span> <span class="v">{html.escape(scene["name"])}</span>  '
        f'<span class="lbl">FRAME</span> <span class="v">{frame_idx + 1:03d}/{n_frames:03d}</span>  '
        f'<span class="lbl">EGO</span> <span class="v">'
        f'x={result.ego.x:.1f} y={result.ego.y:.1f} yaw={result.ego.yaw:+.2f}rad</span>  '
        f'<span class="lbl">SPEED</span> <span class="v">{result.ego.speed:.1f} m/s</span>  '
        f'<span class="lbl">TS</span> <span class="v">{result.ego.timestamp}</span>'
        "</div>"
    )


def _render_footer() -> str:
    return (
        '<div class="prism-footer">'
        'Every decision is <span class="accent">inspectable</span> — '
        "trust for the driver, debuggability for the OEM."
        "</div>"
    )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    st.set_page_config(
        page_title="ADAS Function Powered by PRISM",
        layout="wide",
        page_icon="◆",
        initial_sidebar_state="collapsed",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    # -------- HERO ----------------------------------------------------------
    st.markdown(_render_hero(), unsafe_allow_html=True)

    # -------- Load store (default path; advanced override in expander) ------
    json_dir = DEFAULT_JSON_DIR
    with st.expander("⚙  Data source", expanded=False):
        json_dir = st.text_input(
            "JSON directory",
            value=DEFAULT_JSON_DIR,
            help="Folder with the 13 nuScenes JSON tables (or a parent of v1.0-mini).",
        )
        json_dir = os.path.expanduser(json_dir)

    try:
        store = get_store(json_dir)
    except FileNotFoundError as e:
        st.error(f"Could not load JSON tables from `{json_dir}`:\n\n{e}")
        st.stop()

    # -------- Control strip (scene selector only, no frame slider) ---------
    scenes = store.lists["scene"]
    scene_labels = [f"{s['name']}  —  {s['description']}" for s in scenes]

    scene_idx = st.selectbox(
        "◇ Scenario",
        range(len(scenes)),
        format_func=lambda i: scene_labels[i],
        label_visibility="collapsed",
    )
    scene = scenes[scene_idx]
    n_frames = scene["nbr_samples"]
    frame_idx = 0  # show the first keyframe of the selected scene

    # -------- Compute --------------------------------------------------------
    results = compute_scene_cached(json_dir, scene["token"])
    result = results[frame_idx]
    rules = get_rules()

    # -------- Main area: BEV (left) + Reasoned Alert (right) ---------------
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        # Render to PNG in-memory and embed inline so the <img> is a real
        # child of .bev-wrap (otherwise st.pyplot would render into its own
        # Streamlit block and CSS height locking wouldn't apply).
        fig = plot_bev(
            result.ego,
            result.objects,
            result.decision,
            rules,
            theme="dark",
            figsize=(7.0, 5.0),
            x_range=(-12, 45),
            y_range=(-22, 22),
            show_title=False,
        )
        bev_uri = _fig_to_data_uri(fig)
        plt.close(fig)
        st.markdown(
            f'<div class="bev-wrap"><img src="{bev_uri}" alt="BEV"/></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            _render_telemetry(result, scene, frame_idx, n_frames),
            unsafe_allow_html=True,
        )

    with col_right:
        sig = format_signature_output(result.narration)
        if sig is not None:
            st.markdown(_render_reasoned_alert(sig), unsafe_allow_html=True)

    st.markdown(_render_footer(), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
