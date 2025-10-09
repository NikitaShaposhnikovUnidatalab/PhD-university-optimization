import math
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.state import init_state_obj, init_state_value, QS_INPUT, QS_WEIGHTS, QS_MAX, QS_DELTA, QS_COST, MAX_RU

st.set_page_config(
    page_title="QS Ranking Configurator", 
    page_icon="⚙️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚙️ Налаштування параметрів QS рейтингу")
st.markdown("**Введіть поточні значення показників, ваги та обмеження для оптимізації на 2026 рік**")

print("⚙️ Користувач завантажив сторінку налаштувань")

init_state_obj("QS_INPUT", QS_INPUT)
init_state_obj("QS_WEIGHTS", QS_WEIGHTS)
init_state_obj("QS_MAX", QS_MAX)
init_state_obj("QS_DELTA", QS_DELTA)
init_state_obj("QS_COST", QS_COST)
init_state_value("MAX_RU", MAX_RU)

st.markdown("---")
st.subheader("💰 Загальний бюджет")
st.markdown("**Встановіть загальний бюджет ресурсів (RU) для покращення показників**")

col1, col2 = st.columns([1, 2])
with col1:
    new_budget = st.number_input(
        "Бюджет (RU):",
        value=float(st.session_state["MAX_RU"]),
        step=10.0,
        min_value=0.0,
        help="Загальний бюджет ресурсів для покращення всіх показників",
        key="max_ru_input",
    )
    if new_budget != st.session_state["MAX_RU"]:
        print(f"💰 Користувач змінив бюджет з {st.session_state['MAX_RU']} на {new_budget} RU")
        st.session_state["MAX_RU"] = new_budget

with col2:
    st.metric("Поточний бюджет", f"{st.session_state['MAX_RU']:,} RU")
    if st.session_state["MAX_RU"] > 0:
        st.success("✅ Бюджет встановлено")
    else:
        st.warning("⚠️ Встановіть бюджет більше 0")

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

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Поточні значення показників (2025 рік)",
    "⚖️ Ваги показників (формула QS score)",
    "🔼 Максимальні покращення на один рік (Delta)",
    "💰 Вартість покращення (Cost)"
])

indicator_descriptions = {
    "AR": "Academic Reputation - Репутація в академічному середовищі",
    "ER": "Employer Reputation - Репутація серед роботодавців", 
    "FSR": "Faculty Student Ratio - Співвідношення викладачів до студентів",
    "CPF": "Citations per Faculty - Цитування на викладача",
    "IFR": "International Faculty Ratio - Частка іноземних викладачів",
    "ISR": "International Student Ratio - Частка іноземних студентів",
    "IRN": "International Research Network - Міжнародна дослідницька мережа",
    "EO": "Employment Outcomes - Результати працевлаштування",
    "SUS": "Sustainability - Сталість розвитку"
}

with tab1:
    st.markdown("**Введіть поточні значення показників QS рейтингу**")
    
    col1, col2 = st.columns(2)
    
    for i, k in enumerate(QS_INPUT.keys()):
        col = col1 if i % 2 == 0 else col2
        
        with col:
            new_value = st.number_input(
                f"**{k}** - {indicator_descriptions.get(k, k)}",
                value=float(st.session_state["QS_INPUT"][k]),
                step=0.1,
                min_value=0.0,
                format="%.2f",
                help=indicator_descriptions.get(k, f"Поточне значення показника {k}"),
                key=f"input_{k}",
            )
            if round(new_value, 2) != st.session_state["QS_INPUT"][k]:
                print(f"📊 Користувач змінив показник {k} з {st.session_state['QS_INPUT'][k]} на {round(new_value, 2)}")
                st.session_state["QS_INPUT"][k] = round(new_value, 2)

with tab2:
    st.markdown("**Встановіть важливість кожного показника (сума повинна дорівнювати 1.0)**")
    
    current_sum = sum(float(st.session_state["QS_WEIGHTS"][k]) for k in QS_WEIGHTS.keys())
    st.info(f"**Поточна сума ваг: {current_sum:.2f}** {'✅' if abs(current_sum - 1.0) < 0.01 else '⚠️ (має бути 1.0)'}")
    
    col1, col2 = st.columns(2)
    
    for i, k in enumerate(QS_WEIGHTS.keys()):
        col = col1 if i % 2 == 0 else col2
        
        with col:
            new_weight = st.number_input(
                f"**{k}** - {indicator_descriptions.get(k, k)}",
                value=float(st.session_state["QS_WEIGHTS"][k]),
                step=0.01,
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
                help=f"Вага показника {k} (0.0 - 1.0)",
                key=f"weight_{k}",
            )
            if round(new_weight, 3) != st.session_state["QS_WEIGHTS"][k]:
                print(f"⚖️ Користувач змінив вагу показника {k} з {st.session_state['QS_WEIGHTS'][k]} на {round(new_weight, 3)}")
                st.session_state["QS_WEIGHTS"][k] = round(new_weight, 3)

with tab3:
    st.markdown("**Вкажіть максимальне можливе покращення для кожного показника**")
    
    col1, col2 = st.columns(2)
    
    for i, k in enumerate(QS_DELTA.keys()):
        col = col1 if i % 2 == 0 else col2
        
        with col:
            new_delta = st.number_input(
                f"**{k}** - {indicator_descriptions.get(k, k)}",
                value=float(st.session_state["QS_DELTA"][k]),
                step=0.1,
                min_value=0.0,
                format="%.2f",
                help=f"Максимальне покращення показника {k}",
                key=f"delta_{k}",
            )
            if round(new_delta, 2) != st.session_state["QS_DELTA"][k]:
                print(f"🔼 Користувач змінив delta показника {k} з {st.session_state['QS_DELTA'][k]} на {round(new_delta, 2)}")
                st.session_state["QS_DELTA"][k] = round(new_delta, 2)

with tab4:
    st.markdown("**Встановіть вартість покращення кожного показника (в RU). Введіть 'inf' для показників, які неможливо покращити**")
    
    col1, col2 = st.columns(2)
    
    for i, k in enumerate(QS_COST.keys()):
        col = col1 if i % 2 == 0 else col2
        
        with col:
            current = st.session_state["QS_COST"][k]
            new_str = st.text_input(
                f"**{k}** - {indicator_descriptions.get(k, k)}",
                value="inf" if (isinstance(current, float) and math.isinf(current)) else str(int(current)),
                help=f"Вартість покращення показника {k} (в RU). Введіть 'inf' якщо неможливо покращити",
                key=f"cost_{k}",
            )
            new_cost = float("inf") if new_str.strip().lower() == "inf" else int(new_str)
            if new_cost != st.session_state["QS_COST"][k]:
                print(f"💰 Користувач змінив вартість показника {k} з {st.session_state['QS_COST'][k]} на {new_cost}")
                st.session_state["QS_COST"][k] = new_cost

st.markdown("---")
st.subheader("📋 Підсумок налаштувань")

col1, col2, col3 = st.columns(3)

with col1:
    budget = st.session_state["MAX_RU"]
    st.metric("Бюджет", f"{budget:,} RU", delta=None)

with col2:
    weights_sum = sum(float(st.session_state["QS_WEIGHTS"][k]) for k in QS_WEIGHTS.keys())
    st.metric("Сума ваг", f"{weights_sum:.3f}", delta=f"{weights_sum - 1.0:.3f}")

with col3:
    eligible_count = sum(1 for k in QS_INPUT.keys() if float(st.session_state["QS_DELTA"][k]) > 0 and float(st.session_state["QS_COST"][k]) != float("inf"))
    st.metric("Придатних показників", eligible_count)

if abs(weights_sum - 1.0) > 0.01:
    st.warning("⚠️ **Сума ваг не дорівнює 1.0!** Рекомендується виправити перед запуском оптимізації.")
if budget <= 0:
    st.error("❌ **Бюджет має бути більше 0!**")
if eligible_count < 2:
    st.warning("⚠️ **Недостатньо придатних показників!** Потрібно мінімум 2 для оптимізації.")

st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Переглянути дані", use_container_width=True):
        print("📊 Користувач натиснув кнопку 'Переглянути дані' - відображення поточних налаштувань")
        st.json({
            "MAX_RU": st.session_state["MAX_RU"],
            "QS_INPUT": st.session_state["QS_INPUT"],
            "QS_WEIGHTS": st.session_state["QS_WEIGHTS"],
            "QS_DELTA": st.session_state["QS_DELTA"],
            "QS_COST": st.session_state["QS_COST"],
        })

with col2:
    if st.button("🚀 Запустити оптимізацію", type="primary", use_container_width=True):
        print("🚀 Користувач натиснув кнопку 'Запустити оптимізацію' на сторінці налаштувань")
        if budget > 0 and eligible_count >= 2:
            print(f"✅ Валідація пройшла успішно: бюджет={budget}, придатних показників={eligible_count}")
            st.success("✅ **Параметри налаштовано! Переходимо до оптимізації...**")
            st.switch_page("pages/genetic_optimizer.py")
        else:
            print(f"❌ Валідація не пройшла: бюджет={budget}, придатних показників={eligible_count}")
            st.error("❌ **Виправте помилки перед запуском оптимізації!**")

with col3:
    if st.button("🏠 На головну", use_container_width=True):
        print("🏠 Користувач натиснув кнопку 'На головну' - повернення на головну сторінку")
        st.switch_page("main.py")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>⚙️ QS Ranking Configurator | Налаштування параметрів для оптимізації на 2026 рік</p>
</div>
""", unsafe_allow_html=True)