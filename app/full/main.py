import streamlit as st
import sys
import os

# Додаємо батьківську директорію до шляху для імпорту модулів
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.state import init_state_obj, init_state_value, QS_INPUT, QS_WEIGHTS, QS_MAX, QS_DELTA, QS_COST, MAX_RU

st.set_page_config(
    page_title="QS Ranking Optimizer", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Зберігаємо режим in session state
if "app_mode" not in st.session_state:
    st.session_state["app_mode"] = "full"

st.title("🎓 QS Ranking Optimizer")
st.markdown("**Система оптимізації рейтингу університету для покращення позицій у QS World University Rankings**")

print("🎓 Користувач завантажив головну сторінку")
print(f"📊 Поточний стан сесії: {list(st.session_state.keys())}")

st.markdown("---")

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    ### 🎯 Про застосунок
    
    Цей інструмент допоможе вам:
    - **Оптимізувати показники** QS рейтингу для 2026 року
    - **Знайти найкращі стратегії** покращення в межах бюджету
    - **Порівняти різні підходи** до оптимізації
    - **Візуалізувати результати** та аналізувати ефективність
    """)
    
    st.markdown("""
### 📊 Доступні методи оптимізації:
- **Генетичний алгоритм** - знаходить оптимальні рішення
- **Лінійне програмування** - математично точний підхід
- **Топ-N стратегії** - автоматичний пошук найкращих комбінацій
    """)

with col2:
    st.info("""
    **🚀 Швидкий старт:**
    
    1. Налаштуйте параметри
    2. Оберіть метод оптимізації
    3. Запустіть аналіз
    4. Отримайте рекомендації
    """)

st.markdown("---")

st.subheader("🎯 Почати роботу")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### ⚙️ Налаштування")
    st.markdown("Введіть поточні значення показників, ваги та обмеження")
    if st.button("⚙️ Налаштувати параметри", type="primary", use_container_width=True):
        print("🔧 Користувач натиснув кнопку 'Налаштувати параметри' - перехід на сторінку конфігурації")
        st.switch_page("pages/_input_config.py")

with col2:
    st.markdown("### 🚀 Оптимізація")
    st.markdown("Запустіть оптимізацію з налаштованими параметрами")
    if st.button("🚀 Запустити оптимізацію", type="secondary", use_container_width=True):
        print("🚀 Користувач натиснув кнопку 'Запустити оптимізацію' - ініціалізація стану та перехід на сторінку оптимізації")
        print(f"📊 Ініціалізовано стан з параметрами: MAX_RU={MAX_RU}")
        init_state_obj("QS_INPUT", QS_INPUT)
        init_state_obj("QS_WEIGHTS", QS_WEIGHTS)
        init_state_obj("QS_MAX", QS_MAX)
        init_state_obj("QS_DELTA", QS_DELTA)
        init_state_obj("QS_COST", QS_COST)
        init_state_value("MAX_RU", MAX_RU)
        print(f"📊 Поточний стан сесії: {list(st.session_state.keys())}")
        st.switch_page("pages/2_Оптимізація.py")


st.markdown("---")
st.subheader("✨ Можливості системи")

features_col1, features_col2 = st.columns(2)

with features_col1:
    st.markdown("""
    **🎯 Генетичний алгоритм:**
    - Оптимізує всі показники одночасно
    - Знаходить глобальний оптимум
    - Враховує обмеження бюджету
    
    **🔧 Лінійне програмування:**
    - Математично точний підхід
    - Гарантовано оптимальне рішення
    - Швидкий розрахунок
    """)

with features_col2:
    st.markdown("""
    **🏆 Топ-N стратегії:**
    - Автоматичний пошук комбінацій
    - Аналіз найкращих варіантів
    - Детальне порівняння результатів
    
    **📈 Візуалізація:**
    - Графіки динаміки покращення
    - Heatmap стратегій
    - Аналіз внеску показників
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎓 QS Ranking Optimizer | Розроблено для покращення позицій університетів у світових рейтингах</p>
</div>
""", unsafe_allow_html=True)

