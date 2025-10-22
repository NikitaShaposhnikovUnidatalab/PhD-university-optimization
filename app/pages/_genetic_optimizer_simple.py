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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

# Приховуємо sidebar повністю
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        display: none;
    }
    [data-testid="collapsedControl"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎯 QS Ranking Optimizer (Simple)")
st.markdown("**Оптимізація рейтингу університету за допомогою лінійного програмування**")
st.markdown("📅 **Ціль:** Покращення QS рейтингу на 2026 рік в межах доступного бюджету")

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

st.markdown("---")
st.subheader("📋 Поточні дані")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Бюджет RU", f"{MAX_RU:,}")
with col2:
    eligible_count = sum(1 for k in QS_INPUT.keys() if float(QS_DELTA.get(k, 0.0)) > 0 and float(QS_COST.get(k, 0.0)) != float("inf"))
    st.metric("Придатних показників", eligible_count)
with col3:
    current_qs = sum(float(QS_INPUT[k]) * float(QS_WEIGHTS[k]) for k in QS_INPUT.keys())
    st.metric("Поточний QS Score", f"{current_qs:.2f}")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Оптимізація всіх показників", 
    "🎯 Оптимізація обраних показників", 
    "🏆 Топ 3-5 показників",
    "📈 Результати експериментів"
])

with tab1:
    st.subheader("📊 Оптимізація всіх показників (LP)")
    st.markdown("**Що це робить:** Використовує математичний метод лінійного програмування для знаходження оптимального рішення.")
    st.markdown("""
    - ✅ Гарантовано знаходить глобальний оптимум
    - ✅ Швидко та точно
    - ✅ Оптимізує всі доступні показники одночасно
    """)

    if st.button("🧮 Запустити LP-оптимізацію", type="primary", use_container_width=True):
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

        st.success("✅ **LP-оптимізація завершена!**")
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
        
        # Зберігаємо експеримент для AI аналізу
        st.session_state["last_lp_experiment"] = experiment
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("QS Score (LP)", f"{qs_score_lp:.2f}", delta=f"{qs_score_lp - current_qs:.2f}")
        with col2:
            st.metric("Витрати RU", f"{ru_used:.0f}", delta=f"{ru_used - MAX_RU:.0f}")
        with col3:
            efficiency = (qs_score_lp - current_qs) / ru_used if ru_used > 0 else 0
            st.metric("Ефективність", f"{efficiency:.3f}")
        with col4:
            improvement = ((qs_score_lp - current_qs) / current_qs * 100) if current_qs > 0 else 0
            st.metric("Покращення", f"{improvement:.1f}%")
        with col5:
            st.metric("Час обчислення", f"{elapsed_time_lp_full:.1f}с")
        
        st.subheader("📊 Результати LP-оптимізації")
        st.dataframe(df_lp, use_container_width=True)
        
        st.subheader("📈 Візуалізація результатів LP")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        indicators = list(QS_INPUT.keys())
        values_2025 = [float(QS_INPUT[k]) for k in indicators]
        values_2026 = [float(x_2026[k]) for k in indicators]
        
        x_pos = np.arange(len(indicators))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, values_2025, width, label='2025', color='#2E86AB', alpha=0.8)
        bars2 = ax1.bar(x_pos + width/2, values_2026, width, label='2026 (LP)', color='#E63946', alpha=0.8)
        
        ax1.set_xlabel('Показники', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Значення', fontsize=12, fontweight='bold')
        ax1.set_title('Порівняння показників: 2025 vs 2026 (LP)', fontsize=14, fontweight='bold', pad=20)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(indicators, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        improvements = [values_2026[i] - values_2025[i] for i in range(len(indicators))]
        colors = ['#2ECC71' if imp > 0 else '#E74C3C' for imp in improvements]
        
        bars = ax2.bar(indicators, improvements, color=colors, alpha=0.8)
        ax2.set_xlabel('Показники', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Приріст', fontsize=12, fontweight='bold')
        ax2.set_title('Приріст показників (LP)', fontsize=14, fontweight='bold', pad=20)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.3),
                    f'{imp:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
        
        costs = []
        efficiencies = []
        for k in indicators:
            delta = float(x_2026[k]) - float(QS_INPUT[k])
            if QS_COST[k] < float("inf") and delta > 0:
                cost = float(QS_COST[k]) * delta
                efficiency = delta / cost if cost > 0 else 0
            else:
                cost = 0
                efficiency = 0
            costs.append(cost)
            efficiencies.append(efficiency)
        
        non_zero_indices = [i for i, cost in enumerate(costs) if cost > 0]
        if non_zero_indices:
            filtered_indicators = [indicators[i] for i in non_zero_indices]
            filtered_efficiencies = [efficiencies[i] for i in non_zero_indices]
            filtered_costs = [costs[i] for i in non_zero_indices]
            
            scatter = ax3.scatter(filtered_costs, filtered_efficiencies, 
                                c=filtered_efficiencies, cmap='viridis', s=100, alpha=0.7)
            ax3.set_xlabel('Витрати RU', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Ефективність (приріст/витрати)', fontsize=12, fontweight='bold')
            ax3.set_title('Ефективність витрат по показниках (LP)', fontsize=14, fontweight='bold', pad=20)
            ax3.grid(True, alpha=0.3)
            
            for i, (cost, eff, ind) in enumerate(zip(filtered_costs, filtered_efficiencies, filtered_indicators)):
                ax3.annotate(ind, (cost, eff), xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            cbar = plt.colorbar(scatter, ax=ax3)
            cbar.set_label('Ефективність', fontsize=11, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'Немає витрат RU', ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Ефективність витрат по показниках (LP)', fontsize=14, fontweight='bold', pad=20)
        
        if any(cost > 0 for cost in costs):
            non_zero_costs = [costs[i] for i in range(len(costs)) if costs[i] > 0]
            non_zero_indicators = [indicators[i] for i in range(len(costs)) if costs[i] > 0]
            
            wedges, texts, autotexts = ax4.pie(non_zero_costs, labels=non_zero_indicators, autopct='%1.1f%%', 
                                              startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, len(non_zero_costs))))
            ax4.set_title('Розподіл витрат RU по показниках (LP)', fontsize=14, fontweight='bold', pad=20)
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            ax4.text(0.5, 0.5, 'Немає витрат RU', ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Розподіл витрат RU по показниках (LP)', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.clf()
    
    # AI Аналіз секція
    st.markdown("---")
    st.subheader("🤖 AI Аналіз результатів")
    
    if st.button("🧠 Генерувати AI інсайт (LP)", type="primary", use_container_width=True):
        print("🧠 Користувач запустив AI аналіз результату LP оптимізації")
        with st.spinner("🤖 AI аналізує результат LP оптимізації..."):
            try:
                import sys
                from pathlib import Path
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from llm import generate_qs_insights
                
                # Використовуємо збережений експеримент
                if "last_lp_experiment" in st.session_state:
                    experiment = st.session_state["last_lp_experiment"]
                    insights_result = generate_qs_insights(experiment, current_qs, MAX_RU)
                    st.session_state["last_insights_lp"] = insights_result
                    st.success("✅ **AI аналіз LP завершено!**")
                else:
                    st.error("❌ **Немає даних LP експерименту**")
                    st.info("💡 Спочатку запустіть LP оптимізацію")
                
            except Exception as e:
                st.error(f"❌ **Помилка генерації інсайтів:** {str(e)}")
                st.info("💡 Перевірте налаштування API ключа Google Gemini")
    
    # Відображаємо інсайти
    if "last_insights_lp" in st.session_state:
        insights = st.session_state["last_insights_lp"]
        
        if insights["status"] == "success":
            st.subheader("💡 AI Аналіз та Рекомендації (LP)")
            st.markdown(insights.get("text", "Відповідь відсутня"))
                
        elif insights["status"] == "no_api":
            st.error("❌ **API ключ Google Gemini не налаштовано**")
            st.info("💡 Додайте GOOGLE_API_KEY в файл .env")
        elif insights["status"] == "error":
            st.error(f"❌ **Помилка аналізу:** {insights.get('text', 'Невідома помилка')}")
        elif insights["status"] == "empty":
            st.warning("⚠️ Отримано порожню відповідь від LLM")

with tab2:
    st.subheader("🎯 Вибір показників для покращення (LP)")
    st.markdown("""
    **Що це робить:** Дозволяє вам вручну обрати конкретні показники для покращення.
    - ✅ Повний контроль над тим, що оптимізувати
    - ✅ Можна тестувати різні комбінації
    - ✅ Підходить для стратегічного планування
    """)

    all_keys = list(QS_INPUT.keys())
    default_selected = [k for k in all_keys if float(QS_DELTA.get(k, 0.0)) > 0]
    if "SELECTED_INDICATORS" not in st.session_state:
        st.session_state["SELECTED_INDICATORS"] = default_selected

    new_selection = st.multiselect(
        "🔍 Оберіть показники для покращення:",
        options=all_keys,
        default=st.session_state["SELECTED_INDICATORS"],
        key="SELECTED_INDICATORS",
        help="Виберіть один або кілька показників для оптимізації. Інші залишаться незмінними."
    )
    selected_keys = list(new_selection) or []
    
    if set(selected_keys) != set(st.session_state.get("SELECTED_INDICATORS", [])):
        print(f"🎯 Користувач змінив вибір показників: {selected_keys}")
        st.session_state["SELECTED_INDICATORS"] = selected_keys

    if not selected_keys:
        st.warning("⚠️ **Оберіть хоча б один показник для оптимізації!**")
    else:
        st.info(f"📊 **Обрано показників:** {len(selected_keys)} з {len(all_keys)}")
        
        if st.button("🧮 Запустити LP (обрані)", type="primary", use_container_width=True):
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

            # Зберігаємо експеримент для AI аналізу
            st.session_state["last_lp_selected_experiment"] = experiment

            st.success("✅ **LP-оптимізація (обрані) завершена!**")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("QS Score (LP)", f"{qs_score_lp:.2f}", delta=f"{qs_score_lp - current_qs:.2f}")
            with col2:
                st.metric("Витрати RU", f"{ru_used:.0f}", delta=f"{ru_used - MAX_RU:.0f}")
            with col3:
                efficiency = (qs_score_lp - current_qs) / ru_used if ru_used > 0 else 0
                st.metric("Ефективність", f"{efficiency:.3f}")
            with col4:
                improvement = ((qs_score_lp - current_qs) / current_qs * 100) if current_qs > 0 else 0
                st.metric("Покращення", f"{improvement:.1f}%")

            st.subheader("📊 Результати LP-оптимізації (обрані)")
            st.dataframe(df_lp, use_container_width=True)

    # AI Аналіз секція для табу 2
    st.markdown("---")
    st.subheader("🤖 AI Аналіз результатів (обрані показники)")
    
    if st.button("🧠 Генерувати AI інсайт (LP обрані)", type="primary", use_container_width=True):
        print("🧠 Користувач запустив AI аналіз результату LP оптимізації обраних показників")
        with st.spinner("🤖 AI аналізує результат LP оптимізації обраних показників..."):
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
                    st.success("✅ **AI аналіз LP (обрані) завершено!**")
                else:
                    st.error("❌ **Немає даних LP експерименту обраних показників**")
                    st.info("💡 Спочатку запустіть LP оптимізацію обраних показників")
                
            except Exception as e:
                st.error(f"❌ **Помилка генерації інсайтів:** {str(e)}")
                st.info("💡 Перевірте налаштування API ключа Google Gemini")
    
    # Відображаємо інсайти
    if "last_insights_lp_selected" in st.session_state:
        insights = st.session_state["last_insights_lp_selected"]
        
        if insights["status"] == "success":
            st.subheader("💡 AI Аналіз та Рекомендації (LP обрані)")
            st.markdown(insights.get("text", "Відповідь відсутня"))
                
        elif insights["status"] == "no_api":
            st.error("❌ **API ключ Google Gemini не налаштовано**")
            st.info("💡 Додайте GOOGLE_API_KEY в файл .env")
        elif insights["status"] == "error":
            st.error(f"❌ **Помилка аналізу:** {insights.get('text', 'Невідома помилка')}")
        elif insights["status"] == "empty":
            st.warning("⚠️ Отримано порожню відповідь від LLM")

with tab3:
    st.subheader("🏆 Топ стратегій: Автоматичний пошук найкращих комбінацій (LP)")
    st.markdown("""
    **Що це робить:** Система автоматично перебирає всі можливі комбінації з N показників і знаходить найкращі стратегії.
    - 🔍 Перебирає всі можливі комбінації
    - 🏆 Показує топ-3 найкращі стратегії
    - 📊 Детальний аналіз кожної стратегії
    - ⚡ Швидко знаходить оптимальні рішення
    """)

    def get_eligible_indicators():
        eligible = []
        for key in QS_INPUT.keys():
            if (float(QS_DELTA.get(key, 0.0)) > 0.0 and 
                float(QS_COST.get(key, 0.0)) != float("inf")):
                eligible.append(key)
        return eligible

    eligible = get_eligible_indicators()

    if len(eligible) < 2:
        st.error("❌ **Недостатньо придатних показників для покращення!**")
        st.markdown(f"""
        **Потрібно мінімум 2 показники з:**
        - Delta > 0 (можна покращити)
        - Скінченна вартість (не ∞)
        
        **Доступні показники:** {', '.join(eligible) if eligible else 'Немає'}
        
        **Рішення:** Перейдіть на сторінку налаштувань і змініть параметри показників.
        """)
    else:
        st.success(f"✅ **Знайдено {len(eligible)} придатних показників для покращення**")
        st.markdown(f"**Доступні показники:** {', '.join(eligible)}")
        
        st.markdown("**⚙️ Налаштування пошуку:**")
        num_indicators = st.selectbox(
            "Кількість показників для покращення:",
            options=list(range(2, min(len(eligible) + 1, 6))),
            index=1,
            help=f"Доступно придатних показників: {len(eligible)}",
            key="topn_num_indicators"
        )
        
        total_combinations = len(list(combinations(eligible, num_indicators)))
        st.info(f"📊 **Буде перевірено {total_combinations} комбінацій показників**")
        
        if st.button("🧮 Запустити топ-N оптимізацію (LP)", type="primary", use_container_width=True):
            print(f"🏆 Користувач запустив топ-N LP-оптимізацію: {num_indicators} показників з {len(eligible)} доступних")
            run_top_n_lp_optimization(eligible, num_indicators, QS_INPUT, QS_WEIGHTS, QS_MAX, QS_DELTA, QS_COST, MAX_RU, current_qs)

    # AI Аналіз секція для табу 3
    st.markdown("---")
    st.subheader("🤖 AI Аналіз результатів (топ стратегії)")
    
    if st.button("🧠 Генерувати AI інсайт (LP топ-N)", type="primary", use_container_width=True):
        print("🧠 Користувач запустив AI аналіз результату LP топ-N оптимізації")
        with st.spinner("🤖 AI аналізує результат LP топ-N оптимізації..."):
            try:
                import sys
                from pathlib import Path
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from llm import generate_qs_insights
                
                # Використовуємо збережений експеримент
                if "last_lp_topn_experiment" in st.session_state:
                    experiment = st.session_state["last_lp_topn_experiment"]
                    insights_result = generate_qs_insights(experiment, current_qs, MAX_RU)
                    st.session_state["last_insights_lp_topn"] = insights_result
                    st.success("✅ **AI аналіз LP (топ-N) завершено!**")
                else:
                    st.error("❌ **Немає даних LP експерименту топ-N**")
                    st.info("💡 Спочатку запустіть LP топ-N оптимізацію")
                
            except Exception as e:
                st.error(f"❌ **Помилка генерації інсайтів:** {str(e)}")
                st.info("💡 Перевірте налаштування API ключа Google Gemini")
    
    # Відображаємо інсайти
    if "last_insights_lp_topn" in st.session_state:
        insights = st.session_state["last_insights_lp_topn"]
        
        if insights["status"] == "success":
            st.subheader("💡 AI Аналіз та Рекомендації (LP топ-N)")
            st.markdown(insights.get("text", "Відповідь відсутня"))
                
        elif insights["status"] == "no_api":
            st.error("❌ **API ключ Google Gemini не налаштовано**")
            st.info("💡 Додайте GOOGLE_API_KEY в файл .env")
        elif insights["status"] == "error":
            st.error(f"❌ **Помилка аналізу:** {insights.get('text', 'Невідома помилка')}")
        elif insights["status"] == "empty":
            st.warning("⚠️ Отримано порожню відповідь від LLM")