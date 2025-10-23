import streamlit as st
import sys
import os

# Додаємо батьківську директорію до шляху для імпорту модулів
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.state import init_state_obj, init_state_value, QS_INPUT, QS_WEIGHTS, QS_MAX, QS_DELTA, QS_COST, MAX_RU

st.set_page_config(
    page_title="Калькулятор покращення рейтингу університету", 
    page_icon="🎯", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Зберігаємо режим in session state
if "app_mode" not in st.session_state:
    st.session_state["app_mode"] = "simple"

st.title("🎯 Калькулятор QS рейтингу")
st.caption("Оптимізація показників університету до 2026 року")

print("🎯 Користувач завантажив головну сторінку")
print(f"📊 Поточний стан сесії: {list(st.session_state.keys())}")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    if st.button("⚙️ Налаштування", type="primary", use_container_width=True, help="Введіть показники, ваги та бюджет"):
        print("🔧 Перехід на налаштування")
        st.switch_page("pages/1_Налаштування.py")

with col2:
    if st.button("🧮 Розрахунок", type="secondary", use_container_width=True, help="Запустіть оптимізацію"):
        print("🚀 Ініціалізація та перехід на розрахунок")
        init_state_obj("QS_INPUT", QS_INPUT)
        init_state_obj("QS_WEIGHTS", QS_WEIGHTS)
        init_state_obj("QS_MAX", QS_MAX)
        init_state_obj("QS_DELTA", QS_DELTA)
        init_state_obj("QS_COST", QS_COST)
        init_state_value("MAX_RU", MAX_RU)
        st.switch_page("pages/2_Розрахунок.py")

with st.expander("ℹ️ Про систему"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Що робить:**
        - Оптимізує показники QS рейтингу
        - Шукає найкращі комбінації
        - Розраховує в межах бюджету
        """)
    with col2:
        st.markdown("""
        **Методи:**
        - Лінійне програмування
        - Топ-N стратегії
        - AI аналіз результатів
        """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎯 Калькулятор покращення рейтингу університету | Розроблено для покращення позицій університетів у світових рейтингах</p>
</div>
""", unsafe_allow_html=True)
