import math
import streamlit as st
from utils.state import init_state_obj, init_state_value, QS_INPUT, QS_WEIGHTS, QS_MAX, QS_DELTA, QS_COST, MAX_RU

st.set_page_config(page_title="QS Configurator — Page 1", layout="centered")
st.title("QS Ranking — Page 1: Editable Inputs")
st.caption("Усі поля без верхніх обмежень. Значення одразу зберігаються у st.session_state.")

def get_cost_str(x) -> str:
    if isinstance(x, (float, int)) and math.isinf(float(x)):
        return "inf"
    return str(x)

def parse_cost_str(s: str, fallback):
    s = (s or "").strip().lower()
    if s in ("inf", "+inf", "infinity"):
        return float("inf")
    try:
        return float(s)
    except Exception:
        return fallback

init_state_obj("QS_INPUT", QS_INPUT)
init_state_obj("QS_WEIGHTS", QS_WEIGHTS)
init_state_obj("QS_MAX", QS_MAX)
init_state_obj("QS_DELTA", QS_DELTA)
init_state_obj("QS_COST", QS_COST)
init_state_value("MAX_RU", MAX_RU)

st.header("🧭 Global")
st.session_state["MAX_RU"] = st.number_input(
    "MAX_RU",
    value=float(st.session_state["MAX_RU"]),
    step=1.0,
    key="max_ru_input",
)

st.header("📥 QS_INPUT (значення показників)")
for k in QS_INPUT.keys():
    st.session_state["QS_INPUT"][k] = round(
        st.number_input(
            f"{k}",
            value=float(st.session_state["QS_INPUT"][k]),
            step=0.1,
            format="%.2f",
            key=f"input_{k}",
        ), 2
    )

st.header("⚖️ QS_WEIGHTS (ваги)")
for k in QS_WEIGHTS.keys():
    st.session_state["QS_WEIGHTS"][k] = round(
        st.number_input(
            f"Weight {k}",
            value=float(st.session_state["QS_WEIGHTS"][k]),
            step=0.01,
            format="%.2f",
            key=f"weight_{k}",
        ), 2
    )

# --- QS_MAX ---
# st.header("🔝 QS_MAX (максимальні значення показників)")
# for k in QS_MAX.keys():
#     st.session_state["QS_MAX"][k] = int(
#         st.number_input(
#             f"Max {k}",
#             value=int(st.session_state["QS_MAX"][k]),
#             step=1,
#             format="%d",
#             key=f"max_{k}",
#         )
#     )

st.header("🔼 QS_DELTA (дельти)")
for k in QS_DELTA.keys():
    st.session_state["QS_DELTA"][k] = round(
        st.number_input(
            f"Delta {k}",
            value=float(st.session_state["QS_DELTA"][k]),
            step=0.1,
            format="%.2f",
            key=f"delta_{k}",
        ), 2
    )

# --- QS_COST ---
st.header("💰 QS_COST (вартість; підтримує 'inf')")
for k in QS_COST.keys():
    current = st.session_state["QS_COST"][k]
    new_str = st.text_input(
        f"Cost {k}",
        value="inf" if (isinstance(current, float) and math.isinf(current)) else str(int(current)),
        key=f"cost_{k}",
    )
    st.session_state["QS_COST"][k] = float("inf") if new_str.strip().lower() == "inf" else int(new_str)

st.divider()
st.subheader("📊 Поточні значення (SessionState)")
st.json({
    "MAX_RU": st.session_state["MAX_RU"],
    "QS_INPUT": st.session_state["QS_INPUT"],
    "QS_WEIGHTS": st.session_state["QS_WEIGHTS"],
    # "QS_MAX": st.session_state["QS_MAX"],
    "QS_DELTA": st.session_state["QS_DELTA"],
    "QS_COST": st.session_state["QS_COST"],
})
if st.button("▶️ Перейти до генетичного алгоритму"):
    st.switch_page("pages/genetic_optimizer.py")
    
print(st.session_state)