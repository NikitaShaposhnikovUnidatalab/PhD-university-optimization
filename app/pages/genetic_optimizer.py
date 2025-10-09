import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import combinations
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from top_n_optimizer import run_top_n_ga_optimization, run_top_n_lp_optimization
from genetic_optimizer import run_optimization, plot_progress, get_top_solutions, compute_total_ru
from lp import optimize_qs_pulp

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

st.set_page_config(
    page_title="QS Ranking Optimizer", 
    page_icon="🎯", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎯 QS Ranking Optimizer")
st.markdown("**Оптимізація рейтингу університету за допомогою генетичних алгоритмів**")
st.markdown("📅 **Ціль:** Покращення QS рейтингу на 2026 рік в межах доступного бюджету")

print("🎯 Користувач завантажив сторінку оптимізації")
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

tab1, tab2, tab3 = st.tabs([
    "📊 Оптимізація всіх показників", 
    "🎯 Оптимізація обраних показників", 
    "🏆 Топ 3-5 показників"
])

with tab1:
    st.subheader("📊 Оптимізація всіх показників")
    st.markdown("**Що це робить:** Оптимізує всі доступні показники одночасно, щоб отримати максимальний QS Score в межах бюджету.")
    st.markdown("**⚙️ Налаштування генетичного алгоритму:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        auto_find_params = st.checkbox(
            "🔍 Автоматично шукати оптимальні параметри",
            value=True,
            help="Система автоматично знайде найкращі параметри для вашої задачі"
        )
        
        if "prev_auto_find_params" not in st.session_state:
            st.session_state["prev_auto_find_params"] = auto_find_params
            print(f"📊 Оновлений стан сесії prev_auto_find_params: {st.session_state['prev_auto_find_params']}")
            
        if auto_find_params != st.session_state["prev_auto_find_params"]:
            print(f"🔧 Користувач змінив режим пошуку параметрів з {st.session_state['prev_auto_find_params']} на {auto_find_params}")
            st.session_state["prev_auto_find_params"] = auto_find_params
            print(f"📊 Оновлений стан сесії prev_auto_find_params: {st.session_state['prev_auto_find_params']}")
        
        if auto_find_params:
            n_trials = 30
        else:
            st.markdown("**🔧 Ручне налаштування параметрів:**")
            num_generations = st.slider("Кількість поколінь:", 100, 1000, 400)
            sol_per_pop = st.slider("Розмір популяції:", 20, 200, 60)
            
            if "prev_num_generations" not in st.session_state:
                st.session_state["prev_num_generations"] = num_generations
            if num_generations != st.session_state["prev_num_generations"]:
                print(f"🔧 Користувач змінив кількість поколінь з {st.session_state['prev_num_generations']} на {num_generations}")
                st.session_state["prev_num_generations"] = num_generations
                print(f"📊 Оновлений стан сесії prev_num_generations: {st.session_state['prev_num_generations']}")
                
            if "prev_sol_per_pop" not in st.session_state:
                st.session_state["prev_sol_per_pop"] = sol_per_pop
                print(f"📊 Оновлений стан сесії prev_sol_per_pop: {st.session_state['prev_sol_per_pop']}")
            
            if sol_per_pop != st.session_state["prev_sol_per_pop"]:
                print(f"🔧 Користувач змінив розмір популяції з {st.session_state['prev_sol_per_pop']} на {sol_per_pop}")
                st.session_state["prev_sol_per_pop"] = sol_per_pop
                print(f"📊 Оновлений стан сесії prev_sol_per_pop: {st.session_state['prev_sol_per_pop']}")
    
    with col2:
        if auto_find_params:
            st.info("""
            **🎯 Автоматичний пошук параметрів:**
            - Система сама знайде найкращі налаштування
            - Оптимізує: покоління, популяцію, батьків, мутації
            - Гарантує найкращі результати
            """)
        else:
            num_parents_mating = st.slider("Кількість батьків:", 5, 50, 24)
            mutation_percent_genes = st.slider("Відсоток мутацій:", 5, 50, 20)
            
            if "prev_num_parents_mating" not in st.session_state:
                st.session_state["prev_num_parents_mating"] = num_parents_mating
                print(f"📊 Оновлений стан сесії prev_num_parents_mating: {st.session_state['prev_num_parents_mating']}")
                
            if num_parents_mating != st.session_state["prev_num_parents_mating"]:
                print(f"🔧 Користувач змінив кількість батьків з {st.session_state['prev_num_parents_mating']} на {num_parents_mating}")
                st.session_state["prev_num_parents_mating"] = num_parents_mating
                print(f"📊 Оновлений стан сесії prev_num_parents_mating: {st.session_state['prev_num_parents_mating']}")
                
            if "prev_mutation_percent_genes" not in st.session_state:
                st.session_state["prev_mutation_percent_genes"] = mutation_percent_genes
            if mutation_percent_genes != st.session_state["prev_mutation_percent_genes"]:
                print(f"🔧 Користувач змінив відсоток мутацій з {st.session_state['prev_mutation_percent_genes']} на {mutation_percent_genes}")
                st.session_state["prev_mutation_percent_genes"] = mutation_percent_genes
                print(f"📊 Оновлений стан сесії prev_mutation_percent_genes: {st.session_state['prev_mutation_percent_genes']}")
                
            st.info("""
            **⚙️ Ручне налаштування:**
            - Ви самі контролюєте параметри
            - Швидший запуск
            - Потребує досвіду в налаштуванні GA
            """)

    if st.button("🚀 Запустити GA-оптимізацію", type="primary", use_container_width=True):
        print("🧬 Користувач запустив GA-оптимізацію всіх показників")
        print(f"📊 Параметри: бюджет={MAX_RU}, показників={len(QS_INPUT)}")
        print(f"🔍 Автоматичний пошук параметрів: {auto_find_params}")
        if auto_find_params:
            print(f"📊 Параметри пошуку: експериментів={n_trials}")
        else:
            print(f"📊 Ручні параметри: поколінь={num_generations}, популяція={sol_per_pop}, батьки={num_parents_mating}, мутації={mutation_percent_genes}%")
        
        start_time = time.time()
        
        if auto_find_params:
            ga = run_optimization(
                QS_INPUT, QS_WEIGHTS, QS_MAX, QS_DELTA, QS_COST, MAX_RU,
                auto_find_params=True,
                n_trials=n_trials,
                verbose=True
            )
        else:
            ga = run_optimization(
                QS_INPUT, QS_WEIGHTS, QS_MAX, QS_DELTA, QS_COST, MAX_RU,
                auto_find_params=False,
                num_generations=num_generations,
                sol_per_pop=sol_per_pop,
                num_parents_mating=num_parents_mating,
                mutation_percent_genes=mutation_percent_genes,
                verbose=True
            )
        solution, qs_score, _ = ga.best_solution()
        elapsed_time_ga_full = time.time() - start_time
        print(f"✅ GA-оптимізація завершена за {elapsed_time_ga_full:.1f}с, QS Score: {qs_score:.2f}")
        result = dict(zip(QS_INPUT.keys(), solution))

        total_ru = compute_total_ru(QS_INPUT, QS_COST, solution)

        st.success("✅ **Оптимізація завершена!**")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("QS Score", f"{qs_score:.2f}", delta=f"{qs_score - current_qs:.2f}")
        with col2:
            st.metric("Витрати RU", f"{total_ru:.0f}", delta=f"{total_ru - MAX_RU:.0f}")
        with col3:
            efficiency = (qs_score - current_qs) / total_ru if total_ru > 0 else 0
            st.metric("Ефективність", f"{efficiency:.3f}", help="QS Score на одиницю RU")
        with col4:
            improvement = ((qs_score - current_qs) / current_qs * 100) if current_qs > 0 else 0
            st.metric("Покращення", f"{improvement:.1f}%")
        with col5:
            st.metric("Час обчислення", f"{elapsed_time_ga_full:.1f}с")
        
        st.subheader("📊 Детальні результати")
        ru_spent = [
            (float(solution[i]) - float(QS_INPUT[k])) * float(QS_COST[k])
            if float(QS_COST[k]) != float("inf") and (float(solution[i]) - float(QS_INPUT[k])) > 0
            else 0
            for i, k in enumerate(QS_INPUT.keys())
        ]
        result_df = pd.DataFrame({
            "Показник": list(QS_INPUT.keys()),
            "2025": [float(QS_INPUT[k]) for k in QS_INPUT.keys()],
            "2026 (оптимізовано)": [float(solution[i]) for i in range(len(solution))],
            "Приріст": [float(solution[i]) - float(QS_INPUT[list(QS_INPUT.keys())[i]]) for i in range(len(solution))],
            "Витрати RU": ru_spent
        })
        st.dataframe(result_df, use_container_width=True)
        
        st.subheader("📈 Візуалізація результатів GA")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        indicators = list(QS_INPUT.keys())
        values_2025 = [float(QS_INPUT[k]) for k in indicators]
        values_2026 = [float(solution[i]) for i in range(len(solution))]
        
        x_pos = np.arange(len(indicators))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, values_2025, width, label='2025', color='#2E86AB', alpha=0.8)
        bars2 = ax1.bar(x_pos + width/2, values_2026, width, label='2026 (GA)', color='#E63946', alpha=0.8)
        
        ax1.set_xlabel('Показники', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Значення', fontsize=12, fontweight='bold')
        ax1.set_title('Порівняння показників: 2025 vs 2026 (GA)', fontsize=14, fontweight='bold', pad=20)
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
        ax2.set_title('Приріст показників (GA)', fontsize=14, fontweight='bold', pad=20)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.3),
                    f'{imp:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
        
        costs = []
        efficiencies = []
        for i, k in enumerate(indicators):
            delta = float(solution[i]) - float(QS_INPUT[k])
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
            ax3.set_title('Ефективність витрат по показниках (GA)', fontsize=14, fontweight='bold', pad=20)
            ax3.grid(True, alpha=0.3)
            
            for i, (cost, eff, ind) in enumerate(zip(filtered_costs, filtered_efficiencies, filtered_indicators)):
                ax3.annotate(ind, (cost, eff), xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            cbar = plt.colorbar(scatter, ax=ax3)
            cbar.set_label('Ефективність', fontsize=11, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'Немає витрат RU', ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Ефективність витрат по показниках (GA)', fontsize=14, fontweight='bold', pad=20)
        
        if any(cost > 0 for cost in costs):
            non_zero_costs = [costs[i] for i in range(len(costs)) if costs[i] > 0]
            non_zero_indicators = [indicators[i] for i in range(len(costs)) if costs[i] > 0]
            
            wedges, texts, autotexts = ax4.pie(non_zero_costs, labels=non_zero_indicators, autopct='%1.1f%%', 
                                              startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, len(non_zero_costs))))
            ax4.set_title('Розподіл витрат RU по показниках (GA)', fontsize=14, fontweight='bold', pad=20)
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            ax4.text(0.5, 0.5, 'Немає витрат RU', ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Розподіл витрат RU по показниках (GA)', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.clf()
        
        st.subheader("📈 Динаміка покращення QS Score")
        plot_progress(ga)
        st.pyplot(plt)
        plt.clf()

        top_df, contrib_df = get_top_solutions(ga, QS_INPUT, QS_COST, QS_WEIGHTS, top_n=10)
        
        st.subheader("🏆 Топ-10 стратегій (таблиця)")
        st.dataframe(top_df, use_container_width=True)

        
        st.subheader("🔥 Heatmap стратегій")
        delta_df = top_df.set_index("#")[list(QS_INPUT.keys())] - pd.Series(QS_INPUT)
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(delta_df, cmap="RdYlGn", annot=True, fmt=".2f", ax=ax, 
                    cbar_kws={'label': 'Зміна показника'})
        ax.set_title("Зміни показників у топ-10 стратегіях", fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel("Показники", fontsize=12, fontweight='bold')
        ax.set_ylabel("Стратегія", fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("🔥 Heatmap стратегій (нормалізована)")
        delta_df = top_df.set_index("#")[list(QS_INPUT.keys())] - pd.Series(QS_INPUT)
        norm_df = delta_df.copy()
        for col in norm_df.columns:
            max_delta = norm_df[col].max()
            if max_delta > 0:
                norm_df[col] = norm_df[col] / max_delta
            else:
                norm_df[col] = 0.0

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(norm_df, cmap="RdYlGn", annot=True, fmt=".2f", ax=ax,
                    cbar_kws={'label': 'Нормалізована зміна'})
        ax.set_title("Нормалізовані зміни показників у топ-10 стратегіях", fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel("Показники", fontsize=12, fontweight='bold')
        ax.set_ylabel("Стратегія", fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)


    st.markdown("---")
    st.subheader("🔧 Альтернатива: Лінійне програмування (LP)")
    st.markdown("""
    **Що це робить:** Використовує математичний метод лінійного програмування для знаходження оптимального рішення.
    - ✅ Гарантовано знаходить глобальний оптимум
    - ✅ Швидше за генетичний алгоритм
    """)

    if st.button("🧮 Запустити LP-оптимізацію", use_container_width=True):
        print("🧮 Користувач запустив LP-оптимізацію всіх показників")
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

with tab2:
    st.subheader("🎯 Вибір показників для покращення")
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
        print(f"📊 Оновлений стан сесії SELECTED_INDICATORS: {st.session_state['SELECTED_INDICATORS']}")
        st.session_state["SELECTED_INDICATORS"] = selected_keys

    if not selected_keys:
        st.warning("⚠️ **Оберіть хоча б один показник для оптимізації!**")
    else:
        st.info(f"📊 **Обрано показників:** {len(selected_keys)} з {len(all_keys)}")
        
        st.markdown("**⚙️ Налаштування генетичного алгоритму:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            auto_find_params_selected = st.checkbox(
                "🔍 Автоматично шукати оптимальні параметри", 
                value=True,
                help="Система автоматично знайде найкращі параметри для вашої задачі",
                key="auto_find_params_selected"
            )
            
            # Логуємо зміни параметрів
            if "prev_auto_find_params_selected" not in st.session_state:
                st.session_state["prev_auto_find_params_selected"] = auto_find_params_selected
                print(f"📊 Оновлений стан сесії prev_auto_find_params_selected: {st.session_state['prev_auto_find_params_selected']}")
                
            if auto_find_params_selected != st.session_state["prev_auto_find_params_selected"]:
                print(f"🔧 Користувач змінив режим пошуку параметрів для обраних з {st.session_state['prev_auto_find_params_selected']} на {auto_find_params_selected}")
                st.session_state["prev_auto_find_params_selected"] = auto_find_params_selected
                print(f"📊 Оновлений стан сесії prev_auto_find_params_selected: {st.session_state['prev_auto_find_params_selected']}")
            
            if auto_find_params_selected:
                # st.markdown("**🔧 Параметри пошуку:**")
                n_trials_selected = 30
                timeout_minutes_selected = 8
                
                if "prev_n_trials_selected" not in st.session_state:
                    st.session_state["prev_n_trials_selected"] = n_trials_selected
                    print(f"📊 Оновлений стан сесії prev_n_trials_selected: {st.session_state['prev_n_trials_selected']}")
            else:
                st.markdown("**🔧 Ручне налаштування параметрів:**")
                num_generations_selected = st.slider("Кількість поколінь:", 100, 1000, 400, key="num_generations_selected")
                sol_per_pop_selected = st.slider("Розмір популяції:", 20, 200, 60, key="sol_per_pop_selected")
                
                if "prev_num_generations_selected" not in st.session_state:
                    st.session_state["prev_num_generations_selected"] = num_generations_selected
                    print(f"📊 Оновлений стан сесії prev_num_generations_selected: {st.session_state['prev_num_generations_selected']}")
                if num_generations_selected != st.session_state["prev_num_generations_selected"]:
                    print(f"🔧 Користувач змінив кількість поколінь для обраних з {st.session_state['prev_num_generations_selected']} на {num_generations_selected}")
                    st.session_state["prev_num_generations_selected"] = num_generations_selected
                    print(f"📊 Оновлений стан сесії prev_num_generations_selected: {st.session_state['prev_num_generations_selected']}")
                    
                if "prev_sol_per_pop_selected" not in st.session_state:
                    st.session_state["prev_sol_per_pop_selected"] = sol_per_pop_selected
                    print(f"📊 Оновлений стан сесії prev_sol_per_pop_selected: {st.session_state['prev_sol_per_pop_selected']}")
                if sol_per_pop_selected != st.session_state["prev_sol_per_pop_selected"]:
                    print(f"🔧 Користувач змінив розмір популяції для обраних з {st.session_state['prev_sol_per_pop_selected']} на {sol_per_pop_selected}")
                    st.session_state["prev_sol_per_pop_selected"] = sol_per_pop_selected
                    print(f"📊 Оновлений стан сесії prev_sol_per_pop_selected: {st.session_state['prev_sol_per_pop_selected']}")
        
        with col2:
            if auto_find_params_selected:
                st.info("""
                **🎯 Автоматичний пошук параметрів:**
                - Система сама знайде найкращі налаштування
                - Оптимізує: покоління, популяцію, батьків, мутації
                - Гарантує найкращі результати
                """)
            else:
                num_parents_mating_selected = st.slider("Кількість батьків:", 5, 50, 24, key="num_parents_mating_selected")
                mutation_percent_genes_selected = st.slider("Відсоток мутацій:", 5, 50, 20, key="mutation_percent_genes_selected")

                if "prev_num_parents_mating_selected" not in st.session_state:
                    st.session_state["prev_num_parents_mating_selected"] = num_parents_mating_selected
                    print(f"📊 Оновлений стан сесії prev_num_parents_mating_selected: {st.session_state['prev_num_parents_mating_selected']}")
                if num_parents_mating_selected != st.session_state["prev_num_parents_mating_selected"]:
                    print(f"🔧 Користувач змінив кількість батьків для обраних з {st.session_state['prev_num_parents_mating_selected']} на {num_parents_mating_selected}")
                    st.session_state["prev_num_parents_mating_selected"] = num_parents_mating_selected
                    print(f"📊 Оновлений стан сесії prev_num_parents_mating_selected: {st.session_state['prev_num_parents_mating_selected']}")
                    
                if "prev_mutation_percent_genes_selected" not in st.session_state:
                    st.session_state["prev_mutation_percent_genes_selected"] = mutation_percent_genes_selected
                    print(f"📊 Оновлений стан сесії prev_mutation_percent_genes_selected: {st.session_state['prev_mutation_percent_genes_selected']}")  
                if mutation_percent_genes_selected != st.session_state["prev_mutation_percent_genes_selected"]:
                    print(f"🔧 Користувач змінив відсоток мутацій для обраних з {st.session_state['prev_mutation_percent_genes_selected']} на {mutation_percent_genes_selected}")
                    st.session_state["prev_mutation_percent_genes_selected"] = mutation_percent_genes_selected
                    print(f"📊 Оновлений стан сесії prev_mutation_percent_genes_selected: {st.session_state['prev_mutation_percent_genes_selected']}")
                    
                st.info("""
                **⚙️ Ручне налаштування:**
                - Ви самі контролюєте параметри
                - Швидший запуск
                - Потребує досвіду в налаштуванні GA
                """)
        
        cols = st.columns(2)
        with cols[0]:
            if st.button("🚀 Запустити GA (обрані)", key="ga_selected", type="primary", use_container_width=True):
                print(f"🧬 Користувач запустив GA-оптимізацію обраних показників: {selected_keys}")
                print(f"🔍 Автоматичний пошук параметрів: {auto_find_params_selected}")
                if auto_find_params_selected:
                    print(f"📊 Параметри пошуку для обраних: експериментів={n_trials_selected}, час={timeout_minutes_selected}хв")
                else:
                    print(f"📊 Ручні параметри для обраних: поколінь={num_generations_selected}, популяція={sol_per_pop_selected}, батьки={num_parents_mating_selected}, мутації={mutation_percent_genes_selected}%")
                
                effective_delta = {k: (float(QS_DELTA[k]) if k in selected_keys else 0.0) for k in all_keys}
                
                if auto_find_params_selected:
                    ga = run_optimization(
                        QS_INPUT, QS_WEIGHTS, QS_MAX, effective_delta, QS_COST, MAX_RU,
                        auto_find_params=True,
                        n_trials=n_trials_selected,
                        verbose=True
                    )
                else:
                    ga = run_optimization(
                        QS_INPUT, QS_WEIGHTS, QS_MAX, effective_delta, QS_COST, MAX_RU,
                        auto_find_params=False,
                        num_generations=num_generations_selected,
                        sol_per_pop=sol_per_pop_selected,
                        num_parents_mating=num_parents_mating_selected,
                        mutation_percent_genes=mutation_percent_genes_selected,
                        verbose=True
                    )
                
                solution, qs_score, _ = ga.best_solution()
                print(f"✅ GA-оптимізація обраних показників завершена, QS Score: {qs_score:.2f}")

                total_ru = compute_total_ru(QS_INPUT, QS_COST, solution)

                st.success("✅ **GA-оптимізація (обрані) завершена!**")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("QS Score", f"{qs_score:.2f}", delta=f"{qs_score - current_qs:.2f}")
                with col2:
                    st.metric("Витрати RU", f"{total_ru:.0f}", delta=f"{total_ru - MAX_RU:.0f}")
                with col3:
                    efficiency = (qs_score - current_qs) / total_ru if total_ru > 0 else 0
                    st.metric("Ефективність", f"{efficiency:.3f}")
                with col4:
                    improvement = ((qs_score - current_qs) / current_qs * 100) if current_qs > 0 else 0
                    st.metric("Покращення", f"{improvement:.1f}%")

                st.subheader("📊 Детальні результати (GA, обрані)")
                result_df = pd.DataFrame({
                    "Показник": list(QS_INPUT.keys()),
                    "2025": list(QS_INPUT.values()),
                    "2026 (оптимізовано)": solution,
                    "Приріст": [solution[i] - list(QS_INPUT.values())[i] for i in range(len(QS_INPUT))]
                })
                st.dataframe(result_df, use_container_width=True)

        with cols[1]:
            if st.button("🧮 Запустити LP (обрані)", key="lp_selected", use_container_width=True):
                print(f"🧮 Користувач запустив LP-оптимізацію обраних показників: {selected_keys}")
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

with tab3:
    st.subheader("🏆 Топ стратегій: Автоматичний пошук найкращих комбінацій")
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
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**⚙️ Налаштування пошуку:**")
            num_indicators = st.selectbox(
                "Кількість показників для покращення:",
                options=list(range(2, min(len(eligible) + 1, 6))),
                index=1,
                help=f"Доступно придатних показників: {len(eligible)}",
                key="topn_num_indicators"
            )
            
            if "prev_num_indicators" not in st.session_state:
                st.session_state["prev_num_indicators"] = num_indicators
                print(f"📊 Оновлений стан сесії prev_num_indicators: {st.session_state['prev_num_indicators']}")
            if num_indicators != st.session_state["prev_num_indicators"]:
                print(f"🔢 Користувач змінив кількість показників для топ-N з {st.session_state['prev_num_indicators']} на {num_indicators}")
                st.session_state["prev_num_indicators"] = num_indicators
                print(f"📊 Оновлений стан сесії prev_num_indicators: {st.session_state['prev_num_indicators']}")
            
        with col2:
            st.markdown("**🔧 Налаштування GA параметрів:**")
            auto_find_params_topn = st.checkbox(
                "🔍 Автоматично шукати оптимальні параметри GA", 
                value=True,
                help="Система автоматично знайде найкращі параметри для кожної комбінації",
                key="auto_find_params_topn"
            )
            

            if "prev_auto_find_params_topn" not in st.session_state:
                st.session_state["prev_auto_find_params_topn"] = auto_find_params_topn
                print(f"📊 Оновлений стан сесії prev_auto_find_params_topn: {st.session_state['prev_auto_find_params_topn']}")
                
            if auto_find_params_topn != st.session_state["prev_auto_find_params_topn"]:
                print(f"🔧 Користувач змінив режим пошуку параметрів для топ-N з {st.session_state['prev_auto_find_params_topn']} на {auto_find_params_topn}")
                st.session_state["prev_auto_find_params_topn"] = auto_find_params_topn
                print(f"📊 Оновлений стан сесії prev_auto_find_params_topn: {st.session_state['prev_auto_find_params_topn']}")
            
            if auto_find_params_topn:
                n_trials_topn = 10

                if "prev_n_trials_topn" not in st.session_state:
                    st.session_state["prev_n_trials_topn"] = n_trials_topn
                    print(f"📊 Оновлений стан сесії prev_n_trials_topn: {st.session_state['prev_n_trials_topn']}")
            else:
                st.markdown("**🔧 Ручне налаштування параметрів:**")
                num_generations = st.slider("Кількість поколінь:", 100, 500, 200, key="topn_generations", help="Більше поколінь = кращі результати, але довше обчислення")
                sol_per_pop = st.slider("Розмір популяції:", 20, 100, 48, key="topn_pop_size", help="Більша популяція = більше варіантів для пошуку")
                num_parents_mating = st.slider("Кількість батьків:", 10, 50, 20, key="topn_parents", help="Скільки найкращих рішень використовувати для створення нащадків")
                mutation_percent_genes = st.slider("Відсоток мутацій:", 5, 50, 20, key="topn_mutations", help="Відсоток генів, які будуть змінені випадково")
                
                if "prev_num_generations_topn" not in st.session_state:
                    st.session_state["prev_num_generations_topn"] = num_generations
                    print(f"📊 Оновлений стан сесії prev_num_generations_topn: {st.session_state['prev_num_generations_topn']}")
                if num_generations != st.session_state["prev_num_generations_topn"]:
                    print(f"🔧 Користувач змінив кількість поколінь для топ-N з {st.session_state['prev_num_generations_topn']} на {num_generations}")
                    st.session_state["prev_num_generations_topn"] = num_generations
                    print(f"📊 Оновлений стан сесії prev_num_generations_topn: {st.session_state['prev_num_generations_topn']}")
                    
                if "prev_sol_per_pop_topn" not in st.session_state:
                    st.session_state["prev_sol_per_pop_topn"] = sol_per_pop
                    print(f"📊 Оновлений стан сесії prev_sol_per_pop_topn: {st.session_state['prev_sol_per_pop_topn']}")
                if sol_per_pop != st.session_state["prev_sol_per_pop_topn"]:
                    print(f"🔧 Користувач змінив розмір популяції для топ-N з {st.session_state['prev_sol_per_pop_topn']} на {sol_per_pop}")
                    st.session_state["prev_sol_per_pop_topn"] = sol_per_pop
                    print(f"📊 Оновлений стан сесії prev_sol_per_pop_topn: {st.session_state['prev_sol_per_pop_topn']}")
                if "prev_num_parents_mating_topn" not in st.session_state:
                    st.session_state["prev_num_parents_mating_topn"] = num_parents_mating
                    print(f"📊 Оновлений стан сесії prev_num_parents_mating_topn: {st.session_state['prev_num_parents_mating_topn']}")
                if num_parents_mating != st.session_state["prev_num_parents_mating_topn"]:
                    print(f"🔧 Користувач змінив кількість батьків для топ-N з {st.session_state['prev_num_parents_mating_topn']} на {num_parents_mating}")
                    st.session_state["prev_num_parents_mating_topn"] = num_parents_mating
                    print(f"📊 Оновлений стан сесії prev_num_parents_mating_topn: {st.session_state['prev_num_parents_mating_topn']}")
                    
                if "prev_mutation_percent_genes_topn" not in st.session_state:
                    st.session_state["prev_mutation_percent_genes_topn"] = mutation_percent_genes
                    print(f"📊 Оновлений стан сесії prev_mutation_percent_genes_topn: {st.session_state['prev_mutation_percent_genes_topn']}")
                if mutation_percent_genes != st.session_state["prev_mutation_percent_genes_topn"]:
                    print(f"🔧 Користувач змінив відсоток мутацій для топ-N з {st.session_state['prev_mutation_percent_genes_topn']} на {mutation_percent_genes}")
                    st.session_state["prev_mutation_percent_genes_topn"] = mutation_percent_genes
                    print(f"📊 Оновлений стан сесії prev_mutation_percent_genes_topn: {st.session_state['prev_mutation_percent_genes_topn']}")
                    
        total_combinations = len(list(combinations(eligible, num_indicators)))
        st.info(f"📊 **Буде перевірено {total_combinations} комбінацій показників**")
        
        st.markdown("### 🔧 Вибір алгоритму оптимізації")
        
        col1, col2 = st.columns(2)
        
        algorithm = st.radio(
            "Оберіть алгоритм для топ-N оптимізації:",
            options=["Генетичний алгоритм (GA)", "Лінійне програмування (LP)"],
            index=0,
            help="GA - більш гнучкий, LP - швидший та точніший"
        )
        

        if "prev_algorithm" not in st.session_state:
            st.session_state["prev_algorithm"] = algorithm
            print(f"📊 Оновлений стан сесії prev_algorithm: {st.session_state['prev_algorithm']}")
        if algorithm != st.session_state["prev_algorithm"]:
            print(f"🔧 Користувач змінив алгоритм топ-N оптимізації з '{st.session_state['prev_algorithm']}' на '{algorithm}'")
            st.session_state["prev_algorithm"] = algorithm
            print(f"📊 Оновлений стан сесії prev_algorithm: {st.session_state['prev_algorithm']}")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Запустити топ-N оптимізацію (GA)", type="primary", use_container_width=True, disabled=(algorithm != "Генетичний алгоритм (GA)")):
                if algorithm == "Генетичний алгоритм (GA)":
                    print(f"🏆 Користувач запустив топ-N GA-оптимізацію: {num_indicators} показників з {len(eligible)} доступних")
                    print(f"🔍 Автоматичний пошук параметрів: {auto_find_params_topn}")
                    if auto_find_params_topn:
                        print(f"📊 Параметри пошуку для топ-N: експериментів={n_trials_topn}")
                    else:
                        print(f"📊 Ручні параметри для топ-N: поколінь={num_generations}, популяція={sol_per_pop}, батьки={num_parents_mating}, мутації={mutation_percent_genes}%")
                    
                    if auto_find_params_topn:
                        run_top_n_ga_optimization(eligible, num_indicators, None, None, None, None, QS_INPUT, QS_WEIGHTS, QS_MAX, QS_DELTA, QS_COST, MAX_RU, current_qs, auto_find_params=True, n_trials=n_trials_topn)
                    else:
                        run_top_n_ga_optimization(eligible, num_indicators, num_generations, sol_per_pop, num_parents_mating, mutation_percent_genes, QS_INPUT, QS_WEIGHTS, QS_MAX, QS_DELTA, QS_COST, MAX_RU, current_qs, auto_find_params=False)
        
        with col2:
            if st.button("🧮 Запустити топ-N оптимізацію (LP)", type="primary", use_container_width=True, disabled=(algorithm != "Лінійне програмування (LP)")):
                if algorithm == "Лінійне програмування (LP)":
                    print(f"🏆 Користувач запустив топ-N LP-оптимізацію: {num_indicators} показників з {len(eligible)} доступних")
                    run_top_n_lp_optimization(eligible, num_indicators, QS_INPUT, QS_WEIGHTS, QS_MAX, QS_DELTA, QS_COST, MAX_RU, current_qs)
