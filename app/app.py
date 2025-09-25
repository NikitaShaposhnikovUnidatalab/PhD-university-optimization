import streamlit as st
from utils.state import init_state_obj, init_state_value, QS_INPUT, QS_WEIGHTS, QS_MAX, QS_DELTA, QS_COST, MAX_RU
st.set_page_config(page_title="QS Optimizer", layout="centered")

st.title("🎓 QS Ranking Optimizer")
st.markdown("""
Ласкаво просимо!  
Цей застосунок дозволяє налаштувати показники QS Ranking і запустити генетичний алгоритм для оптимізації.
""")

st.divider()
print(st.session_state)
if st.button("⚙️ Налаштувати параметри"):
    st.switch_page("pages/input_config.py")

if st.button("🚀 Запустити оптимізацію"):
    init_state_obj("QS_INPUT", QS_INPUT)
    init_state_obj("QS_WEIGHTS", QS_WEIGHTS)
    init_state_obj("QS_MAX", QS_MAX)
    init_state_obj("QS_DELTA", QS_DELTA)
    init_state_obj("QS_COST", QS_COST)
    init_state_value("MAX_RU", MAX_RU)
    print(st.session_state)
    
    st.switch_page("pages/genetic_optimizer.py")

st.divider()