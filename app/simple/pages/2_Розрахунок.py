import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import combinations
import time
import sys
import os
from datetime import datetime

app_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if app_root not in sys.path:
    sys.path.insert(0, app_root)

from top_n_optimizer import run_top_n_lp_optimization
from genetic_optimizer import compute_total_ru, save_experiment_to_session
from lp import optimize_qs_pulp

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

st.set_page_config(
    page_title="QS Ranking Optimizer - Simple", 
    page_icon="🎯", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🧮 Розрахунок оптимізації")
st.caption("Знайдіть найкраще рішення для покращення QS рейтингу до 2026")

print("🎯 Користувач завантажив сторінку оптимізації (Simple)")
print(f"📊 Поточний стан сесії: {list(st.session_state.keys())}")


required_keys = ["QS_INPUT", "QS_WEIGHTS", "QS_MAX", "QS_DELTA", "QS_COST", "MAX_RU"]

if not all(k in st.session_state for k in required_keys):
    st.error("❌ **Дані ще не введені!**")
    st.markdown("""
    **Щоб почати роботу:**
    1. Перейдіть на сторінку **"⚙️ Налаштування параметрів"** (у меню зліва)
    2. Введіть поточні значення показників, ваги та обмеження
    3. Поверніться на цю сторінку для запуску оптимізації
    """)
    st.stop()
    
QS_INPUT = st.session_state["QS_INPUT"]
QS_WEIGHTS = st.session_state["QS_WEIGHTS"]
QS_MAX = st.session_state["QS_MAX"]
QS_DELTA = st.session_state["QS_DELTA"]
QS_COST = st.session_state["QS_COST"]
MAX_RU = st.session_state["MAX_RU"]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("💰 Бюджет", f"{MAX_RU:,} ресурсних одиниць")
with col2:
    eligible_count = sum(1 for k in QS_INPUT.keys() if float(QS_DELTA.get(k, 0.0)) > 0 and float(QS_COST.get(k, 0.0)) != float("inf"))
    st.metric("📊 Показників", eligible_count)
with col3:
    current_qs = sum(float(QS_INPUT[k]) * float(QS_WEIGHTS[k]) for k in QS_INPUT.keys())
    st.metric("⭐ Поточний бал", f"{current_qs:.2f}")

# Словник з описами показників
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

st.markdown("---")

# CSS для більших вкладок
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        width: 100%;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        width: 50%;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 6px 6px 0px 0px;
        gap: 1px;
        padding: 8px 16px;
        font-size: 15px;
        font-weight: 500;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🧮 Автоматично", "🎯 Вибрати показники"])

with tab1:
    # Функція для отримання доступних показників
    def get_eligible_indicators():
        eligible = []
        for key in QS_INPUT.keys():
            if (float(QS_DELTA.get(key, 0.0)) > 0.0 and 
                float(QS_COST.get(key, 0.0)) != float("inf")):
                eligible.append(key)
        return eligible
    
    eligible = get_eligible_indicators()
    
    if len(eligible) < 2:
        st.error("❌ Недостатньо показників (мінімум 2)")
        st.info("💡 Налаштуйте показники на сторінці 'Налаштування'")
    else:
        with st.expander(f"📋 Доступно {len(eligible)} показників", expanded=False):
            for k in eligible:
                desc = indicator_descriptions.get(k, k)
                st.markdown(f"• **{k}** — {desc}")
        
        # Створюємо список опцій
        options = list(range(2, min(len(eligible), 10)))
        options.append(len(eligible))
        
        def format_option(num):
            if num == len(eligible):
                return f"Всі показники ({num}) — оптимізувати все одночасно"
            else:
                return f"{num} показників — знайти топ-{num} найкращих комбінацій"
        
        # За замовчуванням топ-3, якщо доступно
        default_index = options.index(3) if 3 in options else 0
        
        selected_count = st.selectbox(
            "Оберіть стратегію оптимізації",
            options=options,
            format_func=format_option,
            index=default_index
        )
        
        if selected_count < len(eligible):
            total_combinations = len(list(combinations(eligible, selected_count)))
            st.caption(f"Буде перевірено {total_combinations} комбінацій")
        
        if st.button("🚀 Розрахувати", type="primary", use_container_width=True, key="lp_optimize"):
            if selected_count == len(eligible):
                # Оптимізація всіх показників
                print("🧮 Користувач запустив LP-оптимізацію всіх показників (Simple)")
                selected = [k for k, d in QS_DELTA.items() if float(d) > 0]
                print(f"📊 Параметри: бюджет={MAX_RU}, обраних показників={len(selected)}")
                start_time = time.time()
                x_2026, qs_score_lp, df_lp = optimize_qs_pulp(
                    QS_INPUT=QS_INPUT,
                    QS_WEIGHTS=QS_WEIGHTS,
                    QS_MAX=QS_MAX,
                    QS_DELTA=QS_DELTA,
                    QS_COST=QS_COST,
                    MAX_RU=MAX_RU,
                    selected_indicators=selected,
                )

                deltas = {k: float(x_2026[k]) - float(QS_INPUT[k]) for k in QS_INPUT.keys()}
                ru_used = sum(
                    (deltas[k] * float(QS_COST[k])) if QS_COST[k] < float("inf") else 0.0
                    for k in QS_INPUT.keys()
                )

                # Додаємо розшифровку назв показників
                df_lp['Показник'] = df_lp['Показник'].apply(lambda x: f"{x} - {indicator_descriptions.get(x, x)}")
                
                elapsed_time_lp_full = time.time() - start_time
                print(f"✅ LP-оптимізація завершена за {elapsed_time_lp_full:.1f}с, QS Score: {qs_score_lp:.2f}")
                
                experiment = save_experiment_to_session(
                    algorithm="LP",
                    current_qs=current_qs,
                    qs_score=float(qs_score_lp),
                    ru_used=ru_used,
                    execution_time=elapsed_time_lp_full,
                    solution_details={"solution": x_2026, "algorithm": "LP"},
                    comparison_metrics={
                        "improvement": float(qs_score_lp) - current_qs,
                        "improvement_percent": ((float(qs_score_lp) - current_qs) / current_qs * 100) if current_qs > 0 else 0,
                        "efficiency": (float(qs_score_lp) - current_qs) / ru_used if ru_used > 0 else 0,
                        "budget_utilization": ru_used / MAX_RU if MAX_RU > 0 else 0,
                        "current_qs": current_qs
                    },
                    QS_INPUT=QS_INPUT,
                    solution=[float(x_2026[k]) for k in QS_INPUT.keys()]
                )
                
                st.session_state["last_lp_experiment"] = experiment
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⭐ Новий бал", f"{qs_score_lp:.2f}", delta=f"+{qs_score_lp - current_qs:.2f}")
                with col2:
                    improvement = ((qs_score_lp - current_qs) / current_qs * 100) if current_qs > 0 else 0
                    st.metric("📈 Зростання", f"{improvement:.1f}%")
                with col3:
                    st.metric("💰 Витрачено", f"{ru_used:,.0f} ресурсних одиниць")
                
                with st.expander("📊 Деталі", expanded=True):
                    st.dataframe(df_lp, use_container_width=True)
            else:
                # Топ-N комбінації
                print(f"🏆 Користувач запустив топ-N LP-оптимізацію: {selected_count} показників з {len(eligible)} доступних")
                run_top_n_lp_optimization(eligible, selected_count, QS_INPUT, QS_WEIGHTS, QS_MAX, QS_DELTA, QS_COST, MAX_RU, current_qs)
        
        st.markdown("---")
        
        if st.button("🤖 AI аналіз результатів", type="secondary", use_container_width=True, key="ai_analyze"):
            print("🧠 Користувач запустив AI аналіз результату оптимізації")
            with st.spinner("🤖 Аналізуємо результати та готуємо рекомендації..."):
                try:
                    import sys
                    from pathlib import Path
                    sys.path.insert(0, str(Path(__file__).parent.parent))
                    from llm import generate_qs_insights
                    
                    # Використовуємо збережений експеримент
                    if "last_lp_experiment" in st.session_state:
                        experiment = st.session_state["last_lp_experiment"]
                        insights_result = generate_qs_insights(experiment, current_qs, MAX_RU)
                        st.session_state["last_insights"] = insights_result
                        st.success("✅ **Аналіз завершено! Ось наші рекомендації:**")
                    elif "last_lp_topn_experiment" in st.session_state:
                        experiment = st.session_state["last_lp_topn_experiment"]
                        insights_result = generate_qs_insights(experiment, current_qs, MAX_RU)
                        st.session_state["last_insights"] = insights_result
                        st.success("✅ **Аналіз завершено! Ось наші рекомендації:**")
                    else:
                        st.error("❌ **Немає даних для аналізу**")
                        st.info("💡 Спочатку натисніть кнопку 'Розрахувати оптимальне рішення'")
                
                except Exception as e:
                    st.error(f"❌ **Виникла помилка:** {str(e)}")
                    st.info("💡 Зверніться до адміністратора системи")

        if "last_insights" in st.session_state:
            insights = st.session_state["last_insights"]
            
            if insights["status"] == "success":
                st.subheader("💡 Рекомендації та поради")
                st.markdown(insights.get("text", "Відповідь відсутня"))
                    
            elif insights["status"] == "no_api":
                st.error("❌ **Розумний помічник не налаштовано**")
                st.info("💡 Зверніться до адміністратора для налаштування")
            elif insights["status"] == "error":
                st.error(f"❌ **Помилка аналізу:** {insights.get('text', 'Невідома помилка')}")
            elif insights["status"] == "empty":
                st.warning("⚠️ Не вдалося отримати рекомендації")

with tab2:
    all_keys = list(QS_INPUT.keys())
    default_selected = [k for k in all_keys if float(QS_DELTA.get(k, 0.0)) > 0]
    if "SELECTED_INDICATORS" not in st.session_state:
        st.session_state["SELECTED_INDICATORS"] = default_selected

    # Показуємо всі доступні показники з описами в expander
    with st.expander(f"📋 Доступно {len(all_keys)} показників", expanded=False):
        for k in all_keys:
            desc = indicator_descriptions.get(k, k)
            st.markdown(f"• **{k}** — {desc}")
    
    st.markdown("")
    
    # Створюємо словник з форматованими назвами
    formatted_options = {k: f"{k} - {indicator_descriptions.get(k, k).split(' - ')[0]}" for k in all_keys}
    
    st.multiselect(
        "Оберіть показники для оптимізації",
        options=all_keys,
        format_func=lambda x: formatted_options[x],
        key="SELECTED_INDICATORS"
    )
    
    selected_keys = st.session_state.get("SELECTED_INDICATORS", [])

    if not selected_keys:
        st.warning("⚠️ Оберіть хоча б один показник")
    else:
        st.caption(f"Обрано: {len(selected_keys)} із {len(all_keys)}")
        
        if st.button("🚀 Розрахувати", type="primary", use_container_width=True):
            print(f"🧮 Користувач запустив LP-оптимізацію обраних показників: {selected_keys}")
            start_time = time.time()
            x_2026, qs_score_lp, df_lp = optimize_qs_pulp(
                QS_INPUT=QS_INPUT,
                QS_WEIGHTS=QS_WEIGHTS,
                QS_MAX=QS_MAX,
                QS_DELTA=QS_DELTA,
                QS_COST=QS_COST,
                MAX_RU=MAX_RU,
                selected_indicators=selected_keys,
            )
            
            # Додаємо розшифровку назв показників
            df_lp['Показник'] = df_lp['Показник'].apply(lambda x: f"{x} - {indicator_descriptions.get(x, x)}")
            
            print(f"✅ LP-оптимізація обраних показників завершена, QS Score: {qs_score_lp:.2f}")

            deltas = {k: float(x_2026[k]) - float(QS_INPUT[k]) for k in QS_INPUT.keys()}
            ru_used = sum(
                (deltas[k] * float(QS_COST[k])) if QS_COST[k] < float("inf") else 0.0
                for k in QS_INPUT.keys()
            )
            
            experiment = save_experiment_to_session(
                algorithm="LP_Selected",
                current_qs=current_qs,
                qs_score=float(qs_score_lp),
                ru_used=ru_used,
                execution_time=time.time() - start_time,
                solution_details={"solution": x_2026, "algorithm": "LP"},
                comparison_metrics={
                    "improvement": float(qs_score_lp) - current_qs,
                    "improvement_percent": ((float(qs_score_lp) - current_qs) / current_qs * 100) if current_qs > 0 else 0,
                    "efficiency": (float(qs_score_lp) - current_qs) / ru_used if ru_used > 0 else 0,
                    "budget_utilization": ru_used / MAX_RU if MAX_RU > 0 else 0,
                    "current_qs": current_qs
                },
                QS_INPUT=QS_INPUT,
                solution=[float(x_2026[k]) for k in QS_INPUT.keys()]
            )

            st.session_state["last_lp_selected_experiment"] = experiment
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⭐ Новий бал", f"{qs_score_lp:.2f}", delta=f"+{qs_score_lp - current_qs:.2f}")
            with col2:
                improvement = ((qs_score_lp - current_qs) / current_qs * 100) if current_qs > 0 else 0
                st.metric("📈 Зростання", f"{improvement:.1f}%")
            with col3:
                st.metric("💰 Витрачено", f"{ru_used:,.0f} ресурсних одиниць")

            with st.expander("📊 Деталі", expanded=True):
                st.dataframe(df_lp, use_container_width=True)

    st.markdown("---")
    
    if st.button("🤖 AI аналіз результатів", type="secondary", use_container_width=True):
        print("🧠 Користувач запустив AI аналіз результату LP оптимізації обраних показників")
        with st.spinner("🤖 Аналізуємо результати та готуємо рекомендації..."):
            try:
                import sys
                from pathlib import Path
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from llm import generate_qs_insights
                
                # Використовуємо збережений експеримент
                if "last_lp_selected_experiment" in st.session_state:
                    experiment = st.session_state["last_lp_selected_experiment"]
                    insights_result = generate_qs_insights(experiment, current_qs, MAX_RU)
                    st.session_state["last_insights_lp_selected"] = insights_result
                    st.success("✅ **Аналіз завершено! Ось наші рекомендації:**")
                else:
                    st.error("❌ **Немає даних для аналізу**")
                    st.info("💡 Спочатку натисніть кнопку 'Розрахувати для обраних показників'")
                
            except Exception as e:
                st.error(f"❌ **Виникла помилка:** {str(e)}")
                st.info("💡 Зверніться до адміністратора системи")
    
    # Відображаємо рекомендації
    if "last_insights_lp_selected" in st.session_state:
        insights = st.session_state["last_insights_lp_selected"]
        
        if insights["status"] == "success":
            st.subheader("💡 Рекомендації та поради")
            st.markdown(insights.get("text", "Відповідь відсутня"))
                
        elif insights["status"] == "no_api":
            st.error("❌ **Розумний помічник не налаштовано**")
            st.info("💡 Зверніться до адміністратора для налаштування")
        elif insights["status"] == "error":
            st.error(f"❌ **Помилка аналізу:** {insights.get('text', 'Невідома помилка')}")
        elif insights["status"] == "empty":
            st.warning("⚠️ Не вдалося отримати рекомендації")
