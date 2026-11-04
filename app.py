import base64
import json
import time

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Teen Mental Health AI",
    page_icon="🧠",
    layout="wide",
)

# ------------------------------------------------------------------
# Styling: midnight blue, one memorable element (the result panel)
# ------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;700;800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
    --bg: #071426;
    --panel: rgba(16, 40, 68, 0.72);
    --panel-solid: #0F2742;
    --line: rgba(94, 174, 230, 0.22);
    --text: #EAF4FF;
    --muted: #8FA9C4;
    --accent: #35C6F4;
    --accent-deep: #0878D1;
}

html, body, .stApp, [data-testid="stMarkdownContainer"] {
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
}

.stApp {
    background:
        linear-gradient(rgba(94, 174, 230, 0.07) 1px, transparent 1px),
        linear-gradient(90deg, rgba(94, 174, 230, 0.07) 1px, transparent 1px),
        radial-gradient(ellipse at 12% 8%, rgba(20, 100, 190, 0.26), transparent 42%),
        radial-gradient(ellipse at 88% 85%, rgba(0, 180, 220, 0.12), transparent 42%),
        linear-gradient(135deg, #071426, #0B1D35, #102B49);
    background-size: 40px 40px, 40px 40px, auto, auto, auto;
    background-attachment: fixed;
    color: var(--text);
}

header[data-testid="stHeader"] { background: transparent; }

.block-container {
    max-width: 1180px;
    padding-top: 2.2rem;
    padding-bottom: 3rem;
}

/* Hero */
.hero-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
    font-size: clamp(2.2rem, 4.4vw, 3.4rem);
    line-height: 1.05;
    color: #F0F7FF;
    margin: 0 0 0.9rem 0;
}
.hero-sub {
    font-size: 1.12rem;
    color: #BFD9F2;
    max-width: 52ch;
    line-height: 1.6;
    margin: 0;
}
.hero-meta {
    margin-top: 1.1rem;
    color: var(--muted);
    font-size: 0.92rem;
}

/* Panels (st.container(border=True)) */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--panel);
    border: 1px solid var(--line) !important;
    border-radius: 16px;
    padding: 0.4rem 0.6rem 0.6rem 0.6rem;
}
.panel-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 1.12rem;
    color: #8CCBFF;
    margin: 0.2rem 0 0.1rem 0;
}
.panel-hint {
    color: var(--muted);
    font-size: 0.88rem;
    margin: 0 0 0.6rem 0;
}

/* Labels and inputs */
label p, [data-testid="stWidgetLabel"] p {
    color: #CFE3F6 !important;
    font-weight: 500;
    font-size: 0.93rem;
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #142B46;
    border: 1px solid #31577D;
    border-radius: 10px;
}
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within {
    border-color: var(--accent);
    box-shadow: 0 0 0 2px rgba(53, 198, 244, 0.25);
}

/* Predict button */
div.stButton > button {
    background: linear-gradient(90deg, #0878D1, #00A6D6);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.85rem 1.5rem;
    font-weight: 700;
    font-size: 1.02rem;
    width: 100%;
    transition: box-shadow 0.25s ease, background 0.25s ease;
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #0065B8, #008DBB);
    box-shadow: 0 0 22px rgba(0, 166, 214, 0.38);
    color: white;
}
div.stButton > button:focus-visible {
    outline: 2px solid #fff;
    outline-offset: 2px;
}

/* Result panel */
.result-empty {
    border: 1.5px dashed rgba(94, 174, 230, 0.35);
    border-radius: 14px;
    padding: 2.2rem 1.4rem;
    text-align: center;
    color: var(--muted);
    line-height: 1.6;
}
.result-empty strong { color: var(--text); font-weight: 600; }

.result-head {
    display: flex;
    align-items: center;
    gap: 1.3rem;
    margin: 0.4rem 0 1.2rem 0;
}
.ring {
    --pct: 50;
    --c: #35C6F4;
    flex: 0 0 auto;
    width: 118px;
    height: 118px;
    border-radius: 50%;
    background:
        radial-gradient(closest-side, #0F2742 76%, transparent 78% 100%),
        conic-gradient(var(--c) calc(var(--pct) * 1%), rgba(143, 169, 196, 0.2) 0);
    display: grid;
    place-items: center;
}
.ring span {
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
    font-size: 1.55rem;
    color: #F0F7FF;
}
.verdict-label { color: var(--muted); font-size: 0.85rem; margin-bottom: 0.15rem; }
.verdict {
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
    font-size: 1.9rem;
    line-height: 1.1;
    color: var(--c, #35C6F4);
}
.verdict-note { color: #BFD9F2; font-size: 0.93rem; margin-top: 0.4rem; line-height: 1.5; }

.bars { margin: 0.2rem 0 1rem 0; }
.bar-row { display: grid; grid-template-columns: 96px 1fr 52px; align-items: center; gap: 0.7rem; margin: 0.5rem 0; }
.bar-name { color: #CFE3F6; font-size: 0.92rem; }
.bar-track { height: 10px; border-radius: 99px; background: rgba(143, 169, 196, 0.18); overflow: hidden; }
.bar-fill { height: 100%; border-radius: 99px; }
.bar-val { color: #CFE3F6; font-size: 0.9rem; text-align: right; font-variant-numeric: tabular-nums; }
.bar-row.top .bar-name, .bar-row.top .bar-val { color: #fff; font-weight: 600; }

.chips { display: flex; flex-wrap: wrap; gap: 0.45rem; margin: 0.3rem 0 0.9rem 0; }
.chip {
    background: rgba(53, 198, 244, 0.1);
    border: 1px solid rgba(53, 198, 244, 0.28);
    color: #CFE9FB;
    border-radius: 99px;
    padding: 0.22rem 0.7rem;
    font-size: 0.84rem;
}

.care-note {
    border-left: 3px solid rgba(53, 198, 244, 0.6);
    background: rgba(17, 67, 105, 0.45);
    border-radius: 0 10px 10px 0;
    padding: 0.7rem 0.9rem;
    color: #BFD9F2;
    font-size: 0.88rem;
    line-height: 1.55;
}

/* Feature importance rows */
.fi-row { display: grid; grid-template-columns: 210px 1fr 52px; align-items: center; gap: 0.8rem; margin: 0.55rem 0; }
.fi-name { color: #CFE3F6; font-size: 0.93rem; }
.fi-track { height: 10px; border-radius: 99px; background: rgba(143, 169, 196, 0.18); overflow: hidden; }
.fi-fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #0878D1, #35C6F4); }
.fi-val { color: #CFE3F6; font-size: 0.9rem; text-align: right; font-variant-numeric: tabular-nums; }
@media (max-width: 640px) { .fi-row { grid-template-columns: 130px 1fr 46px; } }

hr { border-color: rgba(112, 180, 230, 0.2); }

@media (prefers-reduced-motion: reduce) {
    * { transition: none !important; animation: none !important; }
}
@media (max-width: 640px) {
    .result-head { flex-direction: column; align-items: flex-start; }
    .bar-row { grid-template-columns: 80px 1fr 46px; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Load model and animation
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("depression_risk_model.pkl")


@st.cache_data
def load_animation(path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return None


model = load_model()
robot_animation = load_animation("assets/Anima Bot.json")
brain_animation = load_animation("assets/Brain.json")

# Readable names for the model's input columns (also used to group the
# one-hot encoded columns back to their original feature).
FEATURE_LABELS = {
    "age": "Age",
    "gender": "Gender",
    "daily_social_media_hours": "Daily social media hours",
    "platform_usage": "Main platform",
    "sleep_hours": "Sleep hours",
    "screen_time_before_sleep": "Screen time before sleep",
    "academic_performance": "Academic performance",
    "physical_activity": "Physical activity",
    "social_interaction_level": "Social interaction",
    "stress_level": "Stress level",
    "anxiety_level": "Anxiety level",
}

# Fill these from your evaluation on the test set, for example:
# MODEL_METRICS = {"Accuracy": "0.00", "Macro F1": "0.00"}
MODEL_METRICS = {}


@st.cache_data
def get_feature_importance(_model):
    """Feature importance per original column, or None if unavailable."""
    try:
        importances = _model.steps[-1][1].feature_importances_
        names = None
        for _, step in _model.steps[:-1]:
            if hasattr(step, "transform") and hasattr(step, "get_feature_names_out"):
                names = list(step.get_feature_names_out())
                break
        if names is None or len(names) != len(importances):
            return None

        totals = {}
        for name, value in zip(names, importances):
            base = str(name).split("__", 1)[-1]
            matches = [c for c in FEATURE_LABELS if base == c or base.startswith(c + "_")]
            key = max(matches, key=len) if matches else base
            totals[key] = totals.get(key, 0.0) + float(value)
        return sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    except Exception:
        return None


def render_lottie(animation, size=220):
    """Render a Lottie animation (parsed JSON) inside the page."""
    encoded = base64.b64encode(json.dumps(animation).encode("utf-8")).decode("utf-8")
    html = f"""
<style>html, body {{ margin: 0; padding: 0; overflow: hidden; background: transparent; }}</style>
<script src="https://unpkg.com/@lottiefiles/lottie-player@2.0.4/dist/lottie-player.js"></script>
<div style="width:100%;height:{size}px;display:flex;justify-content:center;align-items:center;background:transparent;">
  <lottie-player
    src="data:application/json;base64,{encoded}"
    background="transparent" speed="1"
    style="width:{size}px;height:{size}px;" loop autoplay>
  </lottie-player>
</div>
"""
    embed_html(html, size + 10)


def embed_html(html, height):
    if hasattr(st, "iframe"):
        # Newer Streamlit: components.html is deprecated in favour of st.iframe
        st.iframe(html, height=height)
    else:
        components.html(html, height=height, scrolling=False)


def scroll_to_result():
    """Smooth-scroll the page to the result panel."""
    embed_html(
        """
<script>
setTimeout(function () {
    try {
        var el = window.parent.document.getElementById("result-anchor");
        if (el) { el.scrollIntoView({ behavior: "smooth", block: "start" }); }
    } catch (e) {}
}, 150);
</script>
""",
        1,
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def risk_color(label) -> str:
    """Pick a colour from the class name so it works with any label set."""
    text = str(label).lower()
    if any(k in text for k in ("low", "minimal", "none", "no ")):
        return "#3DDBB0"
    if any(k in text for k in ("mod", "med", "mild")):
        return "#F2B84B"
    if any(k in text for k in ("high", "severe")):
        return "#FF7A7A"
    return "#35C6F4"


def risk_note(label) -> str:
    color = risk_color(label)
    if color == "#3DDBB0":
        return "These inputs look like profiles the model usually groups as lower risk."
    if color == "#F2B84B":
        return "These inputs look like profiles the model groups in the middle range."
    if color == "#FF7A7A":
        return "These inputs look like profiles the model usually groups as higher risk."
    return "This is the category the model matched to these inputs."


# ------------------------------------------------------------------
# Hero
# ------------------------------------------------------------------
hero_left, hero_right = st.columns([2.2, 1], vertical_alignment="center")

with hero_left:
    st.markdown(
        """
<h1 class="hero-title">Teen Mental Health AI</h1>
<p class="hero-sub">Enter a teenager's social media habits, sleep, and lifestyle.
The model estimates which depression risk category the profile resembles.</p>
<p class="hero-meta">Trained on 2,500 teen survey responses. Classes were balanced with SMOTE during training.</p>
""",
        unsafe_allow_html=True,
    )

with hero_right:
    if robot_animation is not None:
        render_lottie(robot_animation, 220)

st.write("")

# ------------------------------------------------------------------
# Main layout: inputs on the left, result on the right
# ------------------------------------------------------------------
form_col, result_col = st.columns([1.3, 1], gap="large")

with form_col:
    with st.container(border=True):
        st.markdown('<div class="panel-title">About the teen</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", min_value=13, max_value=19, value=16)
        with c2:
            gender = st.selectbox("Gender", ["male", "female"])

    with st.container(border=True):
        st.markdown('<div class="panel-title">Social media</div>', unsafe_allow_html=True)
        daily_social_media_hours = st.slider(
            "Daily social media hours",
            min_value=0.0, max_value=24.0, value=5.5, step=0.5,
        )
        platform_usage = st.selectbox(
            "Main platform",
            ["Instagram", "TikTok", "Snapchat", "Facebook", "Other"],
        )

    with st.container(border=True):
        st.markdown('<div class="panel-title">Sleep</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            sleep_hours = st.slider(
                "Sleep hours per night",
                min_value=0.0, max_value=24.0, value=6.0, step=0.5,
            )
        with c2:
            screen_time_before_sleep = st.slider(
                "Screen time before sleep (hours)",
                min_value=0.0, max_value=12.0, value=2.0, step=0.5,
            )

    with st.container(border=True):
        st.markdown('<div class="panel-title">School and lifestyle</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            academic_performance = st.slider(
                "Academic performance (1 to 5)",
                min_value=1.0, max_value=5.0, value=3.0, step=1.0,
            )
        with c2:
            physical_activity = st.slider(
                "Physical activity (0 to 7)",
                min_value=0.0, max_value=7.0, value=1.0, step=1.0,
            )
        with c3:
            social_interaction_level = st.selectbox(
                "Social interaction", ["low", "medium", "high"]
            )

    with st.container(border=True):
        st.markdown('<div class="panel-title">Wellbeing</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            stress_level = st.slider("Stress level (1 to 10)", 1, 10, 7)
        with c2:
            anxiety_level = st.slider("Anxiety level (1 to 10)", 1, 10, 8)

    predict_clicked = st.button("Predict depression risk", type="primary")


# ------------------------------------------------------------------
# Prediction (stored in session_state so it survives widget changes)
# ------------------------------------------------------------------
def run_prediction():
    new_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "daily_social_media_hours": daily_social_media_hours,
        "platform_usage": platform_usage,
        "sleep_hours": sleep_hours,
        "screen_time_before_sleep": screen_time_before_sleep,
        "academic_performance": academic_performance,
        "physical_activity": physical_activity,
        "social_interaction_level": social_interaction_level,
        "stress_level": stress_level,
        "anxiety_level": anxiety_level,
    }])

    prediction = model.predict(new_data)[0]
    probabilities = model.predict_proba(new_data)[0]

    st.session_state["result"] = {
        "prediction": str(prediction),
        "classes": [str(c) for c in model.classes_],
        "probabilities": [float(p) for p in probabilities],
        "chips": [
            f"{age} years old",
            f"{daily_social_media_hours:g}h on {platform_usage}",
            f"{sleep_hours:g}h sleep",
            f"{screen_time_before_sleep:g}h screens before bed",
            f"Stress {stress_level}/10",
            f"Anxiety {anxiety_level}/10",
        ],
    }


# ------------------------------------------------------------------
# Result panel
# ------------------------------------------------------------------
with result_col:
    st.markdown(
        '<div id="result-anchor" style="scroll-margin-top:24px;"></div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.markdown('<div class="panel-title">Prediction</div>', unsafe_allow_html=True)

        if predict_clicked:
            loading_started = time.monotonic()
            loading_placeholder = st.empty()
            try:
                with st.spinner("Analyzing your answers..."):
                    with loading_placeholder.container():
                        if brain_animation is not None:
                            render_lottie(brain_animation, 180)
                        st.markdown(
                            '<div class="panel-hint" style="text-align:center;">'
                            "Analyzing the profile...</div>",
                            unsafe_allow_html=True,
                        )
                    run_prediction()
                    time.sleep(max(0, 1.2 - (time.monotonic() - loading_started)))
            finally:
                loading_placeholder.empty()
            scroll_to_result()

        result = st.session_state.get("result")

        if result is None:
            st.markdown(
                """
<div class="result-empty">
<strong>No prediction yet</strong><br>
Fill in the sections on the left, then select
<em>Predict depression risk</em>. The result and the model's scores will appear here.
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            label = result["prediction"]
            color = risk_color(label)
            probs = result["probabilities"]
            classes = result["classes"]
            top_pct = max(probs) * 100

            bars_html = ""
            for cls, p in zip(classes, probs):
                is_top = cls == label
                bars_html += (
                    f'<div class="bar-row{" top" if is_top else ""}">'
                    f'<div class="bar-name">{cls}</div>'
                    f'<div class="bar-track"><div class="bar-fill" '
                    f'style="width:{p * 100:.1f}%;background:{risk_color(cls)};"></div></div>'
                    f'<div class="bar-val">{p * 100:.0f}%</div>'
                    f"</div>"
                )

            chips_html = "".join(f'<span class="chip">{c}</span>' for c in result["chips"])

            st.markdown(
                f"""
<div class="result-head" style="--c:{color};">
<div class="ring" style="--pct:{top_pct:.0f};--c:{color};"><span>{top_pct:.0f}%</span></div>
<div>
<div class="verdict-label">Predicted risk category</div>
<div class="verdict">{label}</div>
<div class="verdict-note">{risk_note(label)}</div>
</div>
</div>
<div class="bars">{bars_html}</div>
<div class="chips">{chips_html}</div>
<div class="care-note">
This is a model estimate, not a medical diagnosis. It should not be used to assess
an individual's mental health or replace professional evaluation. If you are worried
about a teenager's wellbeing, talk to a school counselor or a health professional.
</div>
""",
                unsafe_allow_html=True,
            )

# ------------------------------------------------------------------
# What drives the model
# ------------------------------------------------------------------
st.write("")
with st.container(border=True):
    st.markdown('<div class="panel-title">What drives the model</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-hint">How much each input contributed to the Random Forest\'s '
        "predictions overall. This shows what the model relies on, not what causes "
        "depression.</div>",
        unsafe_allow_html=True,
    )

    importance = get_feature_importance(model)

    if importance is None:
        st.markdown(
            '<div class="panel-hint">Feature importance is not available for this model pipeline.</div>',
            unsafe_allow_html=True,
        )
    else:
        top_value = max(v for _, v in importance) or 1.0
        rows = ""
        for key, value in importance:
            label = FEATURE_LABELS.get(key, key)
            rows += (
                '<div class="fi-row">'
                f'<div class="fi-name">{label}</div>'
                f'<div class="fi-track"><div class="fi-fill" style="width:{value / top_value * 100:.1f}%;"></div></div>'
                f'<div class="fi-val">{value * 100:.0f}%</div>'
                "</div>"
            )
        st.markdown(f'<div>{rows}</div>', unsafe_allow_html=True)

    if MODEL_METRICS:
        chips = "".join(
            f'<span class="chip">{name}: {value}</span>' for name, value in MODEL_METRICS.items()
        )
        st.markdown(
            f'<div class="chips" style="margin-top:0.8rem;">{chips}'
            '<span class="chip">Measured on the held-out test set</span></div>',
            unsafe_allow_html=True,
        )