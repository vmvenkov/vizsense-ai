import io
import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from PIL import Image

from core.rules_engine import RuleContext, analyze
from core.spec_parsers import detect_spec_type, parse_vegalite, parse_plotly
from core.fixers import apply_fix_actions_vegalite, make_spec_renderable_with_df


# -----------------------------
# Styling (UI polish)
# -----------------------------
st.set_page_config(
    page_title="VizSense AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
/* App background */
.stApp {
  background: radial-gradient(1200px 700px at 15% 10%, rgba(99, 102, 241, 0.10), transparent 60%),
              radial-gradient(900px 600px at 85% 15%, rgba(16, 185, 129, 0.10), transparent 55%),
              radial-gradient(900px 700px at 55% 95%, rgba(244, 63, 94, 0.08), transparent 55%),
              #0b1220;
  color: #e5e7eb;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
  border-right: 1px solid rgba(255,255,255,0.10);
}

/* Hide Streamlit “menu/footer” */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Headings */
h1, h2, h3 { letter-spacing: -0.02em; }
h1 { font-size: 2.05rem !important; }
h2 { font-size: 1.35rem !important; }
h3 { font-size: 1.10rem !important; }

/* Cards */
.vz-card {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 14px 14px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.20);
}
.vz-card strong { color: #ffffff; }
.vz-muted { color: rgba(229,231,235,0.80); }

/* Badges */
.vz-badge {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 0.85rem;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.06);
}
.vz-badge-high { border-color: rgba(16,185,129,0.35); background: rgba(16,185,129,0.10); }
.vz-badge-med  { border-color: rgba(245,158,11,0.35); background: rgba(245,158,11,0.10); }
.vz-badge-low  { border-color: rgba(244,63,94,0.35);  background: rgba(244,63,94,0.10); }

/* Buttons */
.stButton>button {
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.06);
}
.stButton>button:hover {
  border-color: rgba(99,102,241,0.45);
  box-shadow: 0 0 0 3px rgba(99,102,241,0.18);
}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Data profiling + type handling
# -----------------------------
@dataclass
class DataProfile:
    n_rows: int
    n_cols: int
    col_types: Dict[str, str]          # numeric/categorical/datetime
    cardinality: Dict[str, int]        # nunique
    missing_rate: float
    numeric_cols: list
    categorical_cols: list
    datetime_cols: list


def infer_column_type(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        sample = series.dropna().astype(str).head(80)
        if len(sample) > 0:
            parsed = pd.to_datetime(sample, errors="coerce", utc=False)
            if parsed.notna().mean() > 0.8:
                return "datetime"
    return "categorical"


def apply_user_types(df: pd.DataFrame, user_types: Dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for c, t in user_types.items():
        if t == "numeric":
            out[c] = pd.to_numeric(out[c], errors="coerce")
        elif t == "datetime":
            out[c] = pd.to_datetime(out[c], errors="coerce")
        else:
            out[c] = out[c].astype("string")
    return out


def profile_dataframe(df: pd.DataFrame, col_types: Dict[str, str]) -> DataProfile:
    n_rows, n_cols = df.shape
    missing_rate = float(df.isna().mean().mean()) if n_rows and n_cols else 0.0
    cardinality = {c: int(df[c].nunique(dropna=True)) for c in df.columns}

    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
    categorical_cols = [c for c, t in col_types.items() if t == "categorical"]
    datetime_cols = [c for c, t in col_types.items() if t == "datetime"]

    return DataProfile(
        n_rows=n_rows,
        n_cols=n_cols,
        col_types=col_types,
        cardinality=cardinality,
        missing_rate=missing_rate,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        datetime_cols=datetime_cols,
    )


# -----------------------------
# Image features (fallback mode)
# -----------------------------
@dataclass
class ImageFeatures:
    width: int
    height: int
    n_unique_colors_approx: int
    contrast: float
    edge_density: float


def image_features(pil_img: Image.Image) -> ImageFeatures:
    img = pil_img.convert("RGB")
    w, h = img.size
    arr = np.array(img).astype(np.float32) / 255.0

    small = img.resize((min(360, w), min(360, h)))
    q = small.quantize(colors=64, method=2)
    n_colors = len(q.getcolors() or [])

    lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    contrast = float(lum.std())

    gy, gx = np.gradient(lum)
    grad = np.sqrt(gx * gx + gy * gy)
    edge_density = float((grad > np.quantile(grad, 0.90)).mean())

    return ImageFeatures(
        width=w,
        height=h,
        n_unique_colors_approx=int(n_colors),
        contrast=contrast,
        edge_density=edge_density,
    )


# -----------------------------
# Suggested better chart (Altair)
# -----------------------------
def suggest_better_chart(df: pd.DataFrame, profile: DataProfile) -> Tuple[str, alt.Chart]:
    if profile.datetime_cols and profile.numeric_cols:
        x = profile.datetime_cols[0]
        y = profile.numeric_cols[0]
        desc = "Предложение: line chart (тенденция във времето)."
        ch = alt.Chart(df).mark_line().encode(
            x=alt.X(f"{x}:T", title=x),
            y=alt.Y(f"mean({y}):Q", title=f"mean({y})"),
            tooltip=[alt.Tooltip(f"{x}:T", title=x), alt.Tooltip(f"mean({y}):Q", title=f"mean({y})")]
        ).properties(title=f"{y} over {x} (mean)")
        return desc, ch

    if profile.categorical_cols and profile.numeric_cols:
        x = profile.categorical_cols[0]
        y = profile.numeric_cols[0]
        k = 10
        n = profile.cardinality.get(x, 0)
        if n > k:
            tmp = (df[[x, y]]
                   .dropna()
                   .groupby(x, as_index=False)[y].mean()
                   .sort_values(y, ascending=False))
            tmp["__cat__"] = np.where(tmp.index < k, tmp[x].astype(str), "Other")
            tmp2 = tmp.groupby("__cat__", as_index=False)[y].mean()
            desc = f"Предложение: bar chart с top-{k} + Other (за много категории)."
            ch = alt.Chart(tmp2).mark_bar().encode(
                x=alt.X("__cat__:N", sort="-y", title=x),
                y=alt.Y(f"{y}:Q", title=f"mean({y})", scale=alt.Scale(zero=True)),
                tooltip=[alt.Tooltip("__cat__:N", title=x), alt.Tooltip(f"{y}:Q", title=f"mean({y})")]
            ).properties(title=f"Top-{k} + Other: mean({y}) by {x}")
            return desc, ch

        desc = "Предложение: сортиран bar chart (с нулева база)."
        ch = alt.Chart(df).mark_bar().encode(
            x=alt.X(f"{x}:N", sort="-y", title=x),
            y=alt.Y(f"mean({y}):Q", title=f"mean({y})", scale=alt.Scale(zero=True)),
            tooltip=[alt.Tooltip(f"{x}:N", title=x), alt.Tooltip(f"mean({y}):Q", title=f"mean({y})")]
        ).properties(title=f"mean({y}) by {x}")
        return desc, ch

    if len(profile.numeric_cols) >= 2:
        x, y = profile.numeric_cols[:2]
        desc = "Предложение: scatter plot (връзка между две числови променливи)."
        ch = alt.Chart(df).mark_point().encode(
            x=alt.X(f"{x}:Q", title=x),
            y=alt.Y(f"{y}:Q", title=y),
            tooltip=[x, y]
        ).properties(title=f"{y} vs {x}")
        return desc, ch

    if len(profile.numeric_cols) == 1:
        x = profile.numeric_cols[0]
        desc = "Предложение: histogram (разпределение)."
        ch = alt.Chart(df).mark_bar().encode(
            x=alt.X(f"{x}:Q", bin=True, title=x),
            y=alt.Y("count():Q", title="count"),
            tooltip=[alt.Tooltip("count():Q", title="count")]
        ).properties(title=f"Histogram of {x}")
        return desc, ch

    desc = "Нямам достатъчно numeric/datetime колони за смислено автоматично предложение."
    ch = alt.Chart(pd.DataFrame({"note": [1]})).mark_text().encode(text=alt.value("Not enough columns"))
    return desc, ch


# -----------------------------
# Optional: render Plotly JSON
# -----------------------------
def try_render_plotly_json(fig_dict: dict) -> bool:
    try:
        import plotly.graph_objects as go  # type: ignore
        fig = go.Figure(fig_dict)
        st.plotly_chart(fig, use_container_width=True)
        return True
    except Exception:
        return False


# -----------------------------
# Helpers for UI
# -----------------------------
def badge_confidence(conf: str) -> str:
    cls = "vz-badge"
    if conf == "High":
        cls += " vz-badge-high"
    elif conf == "Medium":
        cls += " vz-badge-med"
    else:
        cls += " vz-badge-low"
    return f'<span class="{cls}">Confidence: <strong>{conf}</strong></span>'


def criterion_bar(label: str, value: int, max_value: int = 25):
    pct = int(round(100 * (value / max_value)))
    st.markdown(f"**{label}** — {value}/{max_value}")
    st.progress(pct)


def group_issues_by_criterion(issues) -> Dict[str, List]:
    out = {}
    for iss in issues:
        out.setdefault(iss.criterion, []).append(iss)
    # Stable ordering
    for k in list(out.keys()):
        out[k] = sorted(out[k], key=lambda x: (-x.penalty, x.id))
    return out


# -----------------------------
# Session state
# -----------------------------
for key in ("raw_spec", "spec_type", "analysis_result", "fixed_spec", "fixed_result"):
    if key not in st.session_state:
        st.session_state[key] = None


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
<div class="vz-card">
  <div style="display:flex;align-items:center;justify-content:space-between;gap:16px;">
    <div>
      <div style="font-size:0.95rem;opacity:0.85;">🧠 Data Mining Project</div>
      <div style="font-size:1.75rem;font-weight:800;line-height:1.1;">VizSense AI</div>
      <div class="vz-muted" style="margin-top:6px;">Анализира <b>как</b> са визуализирани данните и предлага по-ефективни решения.</div>
    </div>
    <div style="text-align:right;">
      <div class="vz-badge">Input: <strong>CSV + Chart</strong></div>
      <div style="height:8px;"></div>
      <div class="vz-badge">Output: <strong>Score + Fix + Better Viz</strong></div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")


# -----------------------------
# Sidebar: Data input
# -----------------------------
with st.sidebar:
    st.markdown("## 📦 Inputs")
    csv_file = st.file_uploader("1) Upload CSV", type=["csv"])

    df_raw: Optional[pd.DataFrame] = None
    df: Optional[pd.DataFrame] = None
    profile: Optional[DataProfile] = None
    user_types: Dict[str, str] = {}

    if csv_file is not None:
        try:
            content = csv_file.getvalue()
            try:
                df_raw = pd.read_csv(io.BytesIO(content))
            except Exception:
                df_raw = pd.read_csv(io.BytesIO(content), encoding="latin-1")

            inferred = {c: infer_column_type(df_raw[c]) for c in df_raw.columns}

            st.markdown("### 🧾 Column Types")
            with st.expander("Set / override types", expanded=True):
                for c in df_raw.columns:
                    user_types[c] = st.selectbox(
                        c,
                        options=["numeric", "categorical", "datetime"],
                        index=["numeric", "categorical", "datetime"].index(inferred[c]),
                        key=f"type_{c}",
                    )

            df = apply_user_types(df_raw, user_types)
            profile = profile_dataframe(df, user_types)

            st.markdown("### 🔎 Quick profile")
            st.caption(f"Rows: {profile.n_rows} | Cols: {profile.n_cols}")
            st.caption(f"Missing rate: {profile.missing_rate:.2%}")
            st.caption(f"Numeric: {len(profile.numeric_cols)} | Cat: {len(profile.categorical_cols)} | Time: {len(profile.datetime_cols)}")

        except Exception as e:
            st.error(f"CSV read error: {e}")

    st.markdown("---")
    chart_file = st.file_uploader("2) Upload Chart (JSON spec or image)", type=["json", "png", "jpg", "jpeg"])
    st.caption("Best: Vega-Lite/Altair JSON → full analysis + Auto-Fix.")


# -----------------------------
# Main tabs
# -----------------------------
tab_analyze, tab_suggest, tab_about = st.tabs(["🧪 Analyze", "✨ Suggested Viz", "ℹ️ About"])

with tab_analyze:
    if profile is None or df is None or chart_file is None:
        st.info("Качи **CSV** и **Chart** отляво (sidebar), за да започнем анализа.")
    else:
        file_name = (chart_file.name or "").lower()

        # Reset per-new upload
        st.session_state["fixed_spec"] = None
        st.session_state["fixed_result"] = None

        ctx: Optional[RuleContext] = None
        raw_spec: Optional[dict] = None
        spec_type: str = "unknown"

        # ---- JSON spec ----
        if file_name.endswith(".json"):
            try:
                raw_spec = json.loads(chart_file.getvalue().decode("utf-8"))
            except Exception:
                raw_spec = json.loads(chart_file.getvalue().decode("utf-8", errors="replace"))

            spec_type = detect_spec_type(raw_spec)
            st.session_state["raw_spec"] = raw_spec
            st.session_state["spec_type"] = spec_type

            if spec_type == "vegalite":
                chart_meta = parse_vegalite(raw_spec, profile)
                ctx = RuleContext(
                    data_profile=profile,
                    source_type="vegalite",
                    chart_meta=chart_meta,
                    raw_spec=raw_spec
                )
                st.success("Chart recognized: **Vega-Lite / Altair JSON** ✅")
                with st.expander("Preview uploaded chart", expanded=False):
                    try:
                        renderable = make_spec_renderable_with_df(raw_spec, df)
                        ch = alt.Chart.from_dict(renderable)
                        st.altair_chart(ch, use_container_width=True)
                    except Exception:
                        st.info("Preview failed (analysis still works). Showing JSON snippet:")
                        st.code(json.dumps(raw_spec, ensure_ascii=False, indent=2)[:3500])

            elif spec_type == "plotly":
                chart_meta = parse_plotly(raw_spec, profile)
                ctx = RuleContext(
                    data_profile=profile,
                    source_type="plotly",
                    chart_meta=chart_meta,
                    raw_spec=raw_spec
                )
                st.success("Chart recognized: **Plotly JSON** ✅")
                with st.expander("Preview uploaded chart (requires plotly)", expanded=False):
                    ok = try_render_plotly_json(raw_spec)
                    if not ok:
                        st.info("Install plotly to preview: `pip install plotly` (analysis still works).")

            else:
                ctx = RuleContext(
                    data_profile=profile,
                    source_type="unknown",
                    chart_meta={"mark": "unknown", "encoding": {}},
                    raw_spec=raw_spec
                )
                st.warning("JSON not recognized as Vega-Lite or Plotly. Analysis confidence will be low.")
                with st.expander("Show uploaded JSON (snippet)", expanded=False):
                    st.code(json.dumps(raw_spec, ensure_ascii=False, indent=2)[:4500])

        # ---- Image ----
        else:
            try:
                img = Image.open(chart_file)
                st.image(img, caption="Uploaded chart image", use_container_width=True)

                imgf = image_features(img)
                ctx = RuleContext(
                    data_profile=profile,
                    source_type="image",
                    chart_meta={"mark": "unknown", "encoding": {}},
                    image_meta={
                        "width": imgf.width,
                        "height": imgf.height,
                        "n_unique_colors_approx": imgf.n_unique_colors_approx,
                        "contrast": imgf.contrast,
                        "edge_density": imgf.edge_density,
                    },
                )
                st.caption(
                    f"Image features: {imgf.width}×{imgf.height}, colors~{imgf.n_unique_colors_approx}, "
                    f"contrast={imgf.contrast:.3f}, edge_density={imgf.edge_density:.3f}"
                )
            except Exception as e:
                st.error(f"Image read error: {e}")
                ctx = None

        if ctx is not None:
            # -------- BEFORE analysis --------
            result_before = analyze(ctx)
            st.session_state["analysis_result"] = result_before

            st.write("")
            st.markdown(
                f"""
<div class="vz-card">
  <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;">
    <div>
      <div style="font-size:1.1rem;font-weight:800;">BEFORE</div>
      <div class="vz-muted">Оценка на качената визуализация</div>
    </div>
    <div>{badge_confidence(result_before.confidence)}</div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

            colA, colB, colC, colD, colE = st.columns([1, 1, 1, 1, 1])
            colA.metric("Total (0–100)", result_before.total)
            colB.metric("Appropriateness", result_before.by_criterion["Appropriateness"])
            colC.metric("Readability", result_before.by_criterion["Readability"])
            colD.metric("Integrity", result_before.by_criterion["Integrity"])
            colE.metric("Accessibility", result_before.by_criterion["Accessibility"])

            with st.expander("Breakdown bars", expanded=False):
                criterion_bar("Appropriateness", result_before.by_criterion["Appropriateness"])
                criterion_bar("Readability", result_before.by_criterion["Readability"])
                criterion_bar("Integrity", result_before.by_criterion["Integrity"])
                criterion_bar("Accessibility", result_before.by_criterion["Accessibility"])

            # Issues, grouped
            st.subheader("Issues & recommendations")
            grouped = group_issues_by_criterion(result_before.issues)
            if not result_before.issues:
                st.success("No issues detected by current rules.")
            else:
                for crit in ["Appropriateness", "Readability", "Integrity", "Accessibility"]:
                    items = grouped.get(crit, [])
                    if not items:
                        continue
                    with st.expander(f"{crit} ({len(items)})", expanded=(crit in ("Integrity", "Appropriateness"))):
                        for iss in items:
                            text = f"**{iss.id}** — {iss.message}\n\n➡️ {iss.suggestion}  \nPenalty: **−{iss.penalty}**"
                            if iss.severity == "error":
                                st.error(text)
                            elif iss.severity == "warning":
                                st.warning(text)
                            else:
                                st.info(text)

            # -------- Auto-Fix (Vega-Lite only) --------
            st.write("")
            st.markdown("### 🛠️ Auto-Fix")

            if not result_before.fixes:
                st.info("No fix actions available for this input (or current rules did not trigger fixable issues).")
            else:
                if spec_type != "vegalite":
                    st.info("Auto-Fix currently applies only to **Vega-Lite/Altair JSON**. Fix ideas are listed below.")
                else:
                    with st.expander("Fixes to apply", expanded=False):
                        for fx in result_before.fixes:
                            st.write(f"- **{fx.id}**: {fx.description}")

                    apply_clicked = st.button("✅ Apply fixes & re-score (Vega-Lite)", type="primary")

                    if apply_clicked:
                        payloads = [fx.payload for fx in result_before.fixes]
                        fixed_spec = apply_fix_actions_vegalite(raw_spec, payloads)
                        st.session_state["fixed_spec"] = fixed_spec

                        # Re-analyze AFTER using the fixed spec
                        fixed_meta = parse_vegalite(fixed_spec, profile)
                        ctx_fixed = RuleContext(
                            data_profile=profile,
                            source_type="vegalite",
                            chart_meta=fixed_meta,
                            raw_spec=fixed_spec
                        )
                        result_after = analyze(ctx_fixed)
                        st.session_state["fixed_result"] = result_after

            # -------- AFTER section (if exists) --------
            if st.session_state["fixed_spec"] is not None and st.session_state["fixed_result"] is not None:
                fixed_spec = st.session_state["fixed_spec"]
                result_after = st.session_state["fixed_result"]

                st.write("")
                st.markdown(
                    """
<div class="vz-card">
  <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;">
    <div>
      <div style="font-size:1.1rem;font-weight:800;">AFTER</div>
      <div class="vz-muted">Оценка след автоматичните поправки</div>
    </div>
  </div>
</div>
""",
                    unsafe_allow_html=True,
                )

                # Before/After comparison
                b = result_before
                a = result_after
                delta_total = a.total - b.total

                c1, c2, c3, c4, c5, c6 = st.columns([1.2, 1, 1, 1, 1, 1])
                c1.metric("Total (0–100)", a.total, delta=delta_total)
                c2.metric("Appropriateness", a.by_criterion["Appropriateness"], delta=a.by_criterion["Appropriateness"] - b.by_criterion["Appropriateness"])
                c3.metric("Readability", a.by_criterion["Readability"], delta=a.by_criterion["Readability"] - b.by_criterion["Readability"])
                c4.metric("Integrity", a.by_criterion["Integrity"], delta=a.by_criterion["Integrity"] - b.by_criterion["Integrity"])
                c5.metric("Accessibility", a.by_criterion["Accessibility"], delta=a.by_criterion["Accessibility"] - b.by_criterion["Accessibility"])
                c6.markdown(badge_confidence(a.confidence), unsafe_allow_html=True)

                left_v, right_v = st.columns([1, 1], gap="large")
                with left_v:
                    st.subheader("Fixed chart preview")
                    try:
                        renderable_fixed = make_spec_renderable_with_df(fixed_spec, df)
                        ch2 = alt.Chart.from_dict(renderable_fixed)
                        st.altair_chart(ch2, use_container_width=True)
                    except Exception as e:
                        st.info(f"Preview failed: {e}")
                        st.code(json.dumps(fixed_spec, ensure_ascii=False, indent=2)[:4500])

                with right_v:
                    st.subheader("What improved?")
                    # Summarize changes by comparing number of issues & total penalties
                    before_count = len(b.issues)
                    after_count = len(a.issues)
                    st.markdown(
                        f"""
<div class="vz-card">
  <div><strong>Issues:</strong> {before_count} → {after_count}</div>
  <div class="vz-muted" style="margin-top:6px;">
    Това сравнение показва ефекта от Auto-Fix върху правилата, които използваме.
  </div>
</div>
""",
                        unsafe_allow_html=True,
                    )

                    with st.expander("AFTER issues (grouped)", expanded=False):
                        grouped2 = group_issues_by_criterion(a.issues)
                        if not a.issues:
                            st.success("No issues detected after fixes.")
                        else:
                            for crit in ["Appropriateness", "Readability", "Integrity", "Accessibility"]:
                                items = grouped2.get(crit, [])
                                if not items:
                                    continue
                                st.markdown(f"**{crit} ({len(items)})**")
                                for iss in items:
                                    st.write(f"- {iss.id}: −{iss.penalty} | {iss.message}")

                st.subheader("Download fixed spec")
                fixed_bytes = json.dumps(fixed_spec, ensure_ascii=False, indent=2).encode("utf-8")
                st.download_button(
                    "⬇️ Download fixed Vega-Lite JSON",
                    data=fixed_bytes,
                    file_name="vizsense_fixed_chart.json",
                    mime="application/json",
                )

            # Convenience: download uploaded JSON (if any)
            if raw_spec is not None:
                with st.expander("Download uploaded chart JSON", expanded=False):
                    raw_bytes = json.dumps(raw_spec, ensure_ascii=False, indent=2).encode("utf-8")
                    st.download_button(
                        "⬇️ Download uploaded JSON",
                        data=raw_bytes,
                        file_name="vizsense_uploaded_chart.json",
                        mime="application/json",
                    )


with tab_suggest:
    if profile is None or df is None:
        st.info("Качи CSV и задай типове, за да генерирам примерна по-добра визуализация.")
    else:
        st.markdown(
            """
<div class="vz-card">
  <div style="font-size:1.1rem;font-weight:800;">✨ Suggested visualization</div>
  <div class="vz-muted">Автоматично предложение според профила на данните (heuristic).</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.write("")
        desc, better = suggest_better_chart(df, profile)
        st.info(desc)
        st.altair_chart(better, use_container_width=True)

        st.markdown("### Save")
        better_json = json.dumps(better.to_dict(), ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button(
            "⬇️ Download suggested JSON (Vega-Lite)",
            data=better_json,
            file_name="vizsense_suggested_chart.json",
            mime="application/json",
        )
        better_html = better.to_html().encode("utf-8")
        st.download_button(
            "⬇️ Download suggested HTML",
            data=better_html,
            file_name="vizsense_suggested_chart.html",
            mime="text/html",
        )

        with st.expander("Data preview", expanded=False):
            st.dataframe(df.head(50), use_container_width=True)


with tab_about:
    st.markdown(
        """
<div class="vz-card">
  <div style="font-size:1.1rem;font-weight:800;">ℹ️ About VizSense AI</div>
  <div class="vz-muted" style="margin-top:6px;">
    VizSense AI анализира не само данните, а <b>визуализационните решения</b>:
    тип графика, оси, агрегации, скали, цветове, легенда, четимост и потенциални “антипатърни”.
  </div>
  <div style="margin-top:10px;">
    <span class="vz-badge">Inputs: CSV + (Vega-Lite / Plotly / Image)</span>
    <span class="vz-badge" style="margin-left:8px;">Outputs: Score + Issues + Fixes + Suggested Viz</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

st.caption(
    "Tip: За full pipeline (Analyze → Fix → Re-score → Export) използвай **Vega-Lite/Altair JSON**."
)
