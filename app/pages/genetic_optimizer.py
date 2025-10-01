# pages/2_genetic_optimizer.py
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import combinations
import time
from genetic_optimizer import run_optimization, plot_progress, get_top_solutions, compute_total_ru
from lp import optimize_qs_pulp

# Set consistent style for all plots
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

# Add info boxes
col1, col2, col3 = st.columns(3)
with col1:
    st.info("📊 **Всі показники**\n\nОптимізує всі доступні показники одночасно")
with col2:
    st.info("🎯 **Вибір показників**\n\nОберіть конкретні показники для покращення")
with col3:
    st.info("🏆 **Топ стратегій**\n\nСистема знайде найкращі комбінації автоматично")


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

# Display current data summary
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
st.subheader("📊 Оптимізація всіх показників")
st.markdown("**Що це робить:** Оптимізує всі доступні показники одночасно, щоб отримати максимальний QS Score в межах бюджету.")

if st.button("🚀 Запустити GA-оптимізацію", type="primary", use_container_width=True):
    ga = run_optimization(QS_INPUT, QS_WEIGHTS, QS_MAX, QS_DELTA, QS_COST, MAX_RU)
    solution, qs_score, _ = ga.best_solution()
    result = dict(zip(QS_INPUT.keys(), solution))

    total_ru = compute_total_ru(QS_INPUT, QS_COST, solution)

    # Results summary
    st.success("✅ **Оптимізація завершена!**")
    
    col1, col2, col3, col4 = st.columns(4)
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
    
    st.subheader("📊 Детальні результати")

    result_df = pd.DataFrame({
        "Показник": list(QS_INPUT.keys()),
        "2025": list(QS_INPUT.values()),
        "2026 (оптимізовано)": solution,
        "Приріст": [solution[i] - list(QS_INPUT.values())[i] for i in range(len(QS_INPUT))]
    })
    st.dataframe(result_df, use_container_width=True)
    
    st.subheader("📈 Динаміка покращення QS Score")
    plot_progress(ga)
    st.pyplot(plt)
    plt.clf()

    top_df, contrib_df = get_top_solutions(ga, QS_INPUT, QS_COST, QS_WEIGHTS, top_n=10)
    
    print(contrib_df)
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
- ⚠️ Обмежено дискретними кроками 0.1
""")

if st.button("🧮 Запустити LP-оптимізацію", use_container_width=True):
    selected = [k for k, d in QS_DELTA.items() if float(d) > 0]
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
    
    st.subheader("📊 Результати LP-оптимізації")
    st.dataframe(df_lp, use_container_width=True)


st.markdown("---")
st.subheader("🎯 Вибір показників для покращення")
st.markdown("""
**Що це робить:** Дозволяє вам вручну обрати конкретні показники для покращення.
- ✅ Повний контроль над тим, що оптимізувати
- ✅ Можна тестувати різні комбінації
- ✅ Підходить для стратегічного планування
""")

# Manual indicator selection
all_keys = list(QS_INPUT.keys())
default_selected = [k for k in all_keys if float(QS_DELTA.get(k, 0.0)) > 0]
if "SELECTED_INDICATORS" not in st.session_state:
    st.session_state["SELECTED_INDICATORS"] = default_selected

st.multiselect(
    "🔍 Оберіть показники для покращення:",
    options=all_keys,
    default=st.session_state["SELECTED_INDICATORS"],
    key="SELECTED_INDICATORS",
    help="Виберіть один або кілька показників для оптимізації. Інші залишаться незмінними."
)
selected_keys = list(st.session_state["SELECTED_INDICATORS"]) or []

if not selected_keys:
    st.warning("⚠️ **Оберіть хоча б один показник для оптимізації!**")
else:
    st.info(f"📊 **Обрано показників:** {len(selected_keys)} з {len(all_keys)}")
    
    cols = st.columns(2)
    with cols[0]:
        if st.button("🚀 Запустити GA (обрані)", key="ga_selected", type="primary", use_container_width=True):
            effective_delta = {k: (float(QS_DELTA[k]) if k in selected_keys else 0.0) for k in all_keys}
            ga = run_optimization(QS_INPUT, QS_WEIGHTS, QS_MAX, effective_delta, QS_COST, MAX_RU)
            solution, qs_score, _ = ga.best_solution()

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
            x_2026, qs_score_lp, df_lp = optimize_qs_pulp(
                QS_INPUT=QS_INPUT,
                QS_WEIGHTS=QS_WEIGHTS,
                QS_MAX=QS_MAX,
                QS_DELTA=QS_DELTA,
                QS_COST=QS_COST,
                MAX_RU=MAX_RU,
                selected_indicators=selected_keys,
            )

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

st.markdown("---")
st.subheader("🏆 Топ стратегій: Автоматичний пошук найкращих комбінацій")
st.markdown("""
**Що це робить:** Система автоматично перебирає всі можливі комбінації з N показників і знаходить найкращі стратегії.
- 🔍 Перебирає всі можливі комбінації
- 🏆 Показує топ-3 найкращі стратегії
- 📊 Детальний аналіз кожної стратегії
- ⚡ Швидко знаходить оптимальні рішення
""")

# Get eligible indicators
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
            index=1,  # Default to 3
            help=f"Доступно придатних показників: {len(eligible)}",
            key="topn_num_indicators"
        )
        
        # GA Parameters for top-N
        num_generations = st.slider("Кількість поколінь:", 100, 500, 200, key="topn_generations", help="Більше поколінь = кращі результати, але довше обчислення")
        sol_per_pop = st.slider("Розмір популяції:", 20, 100, 48, key="topn_pop_size", help="Більша популяція = більше варіантів для пошуку")
    
    with col2:
        st.markdown("**🔧 Параметри алгоритму:**")
        num_parents_mating = st.slider("Кількість батьків:", 10, 50, 20, key="topn_parents", help="Скільки найкращих рішень використовувати для створення нащадків")
        mutation_percent_genes = st.slider("Відсоток мутацій:", 5, 50, 20, key="topn_mutations", help="Відсоток генів, які будуть змінені випадково")
    
    # Calculate total combinations
    total_combinations = len(list(combinations(eligible, num_indicators)))
    st.info(f"📊 **Буде перевірено {total_combinations} комбінацій показників**")
    
    if st.button("🚀 Запустити топ-N оптимізацію", type="primary", use_container_width=True):
        with st.spinner("Обчислюю найкращі комбінації..."):
            start_time = time.time()
            
            # Run optimization for all combinations
            results = []
            all_keys = list(QS_INPUT.keys())
            
            # Progress bar
            total_combinations = len(list(combinations(eligible, num_indicators)))
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, combo in enumerate(combinations(eligible, num_indicators)):
                status_text.text(f"Обробляю комбінацію {i+1}/{total_combinations}: {combo}")
                
                # Freeze all except selected indicators
                target_keys = set(combo)
                frozen_delta = {k: (float(QS_DELTA[k]) if k in target_keys else 0.0) for k in all_keys}
                
                ga = run_optimization(
                    QS_INPUT,
                    QS_WEIGHTS,
                    QS_MAX,
                    frozen_delta,
                    QS_COST,
                    MAX_RU,
                    num_generations=num_generations,
                    sol_per_pop=sol_per_pop,
                    num_parents_mating=num_parents_mating,
                    mutation_percent_genes=mutation_percent_genes,
                    stop_criteria="saturate_10",
                    random_seed=42,
                )
                
                solution, qs_score, _ = ga.best_solution()
                ru = compute_total_ru(QS_INPUT, QS_COST, solution)
                values = {k: float(solution[i]) for i, k in enumerate(all_keys)}
                
                results.append({
                    "combo": combo,
                    "qs_score": float(qs_score),
                    "ru": float(ru),
                    "solution": solution,
                    "values": values,
                })
                
                progress_bar.progress((i + 1) / total_combinations)
            
            # Sort results
            results_df = pd.DataFrame(results).sort_values(
                by=["qs_score", "ru"], 
                ascending=[False, True]
            ).reset_index(drop=True)
            
            elapsed_time = time.time() - start_time
            status_text.text(f"✅ Завершено за {elapsed_time:.1f} секунд")
            progress_bar.empty()
            
            # Display results
            st.success("✅ **Топ-N оптимізація завершена!**")
            st.header("📈 Результати топ-N оптимізації")
            
            if not results_df.empty:
                best = results_df.iloc[0]
                
                # Best strategy
                st.subheader("🏆 Найкраща стратегія")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("QS Score", f"{best['qs_score']:.3f}", delta=f"{best['qs_score'] - current_qs:.3f}")
                with col2:
                    st.metric("Витрати RU", f"{best['ru']:.1f}", delta=f"{best['ru'] - MAX_RU:.1f}")
                with col3:
                    efficiency = (best['qs_score'] - current_qs) / best['ru'] if best['ru'] > 0 else 0
                    st.metric("Ефективність", f"{efficiency:.3f}")
                with col4:
                    improvement = ((best['qs_score'] - current_qs) / current_qs * 100) if current_qs > 0 else 0
                    st.metric("Покращення", f"{improvement:.1f}%")
                
                st.markdown(f"**🎯 Покращені показники:** {', '.join(best['combo'])}")
                
                # Show detailed values
                st.subheader("📊 Детальні значення найкращої стратегії")
                comparison_data = []
                for key in all_keys:
                    comparison_data.append({
                        "Показник": key,
                        "Поточне значення": QS_INPUT[key],
                        "Нове значення": best['values'][key],
                        "Зміна": best['values'][key] - QS_INPUT[key],
                        "Внесок у QS": best['values'][key] * QS_WEIGHTS[key]
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Top 3 strategies
                st.subheader("🥇 Топ-3 стратегії")
                top3_df = results_df.head(3).copy()
                top3_df['Ранг'] = range(1, 4)
                top3_df['Комбінація'] = top3_df['combo'].apply(lambda x: ', '.join(x))
                
                display_cols = ['Ранг', 'Комбінація', 'qs_score', 'ru']
                st.dataframe(top3_df[display_cols].rename(columns={
                    'qs_score': 'QS Score',
                    'ru': 'Витрати RU'
                }), use_container_width=True)
                
                # Visualization
                st.subheader("📊 Візуалізація результатів")
                
                # Bar chart of QS scores
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                
                # Top 10 QS scores
                top10 = results_df.head(10)
                bars = ax1.bar(range(len(top10)), top10['qs_score'], color='#2E86AB', alpha=0.8)
                ax1.set_xlabel('Ранг стратегії', fontsize=12, fontweight='bold')
                ax1.set_ylabel('QS Score', fontsize=12, fontweight='bold')
                ax1.set_title('Топ-10 стратегій за QS Score', fontsize=14, fontweight='bold', pad=20)
                ax1.grid(True, alpha=0.3)
                
                # Highlight best strategy
                bars[0].set_color('#E63946')
                bars[0].set_alpha(1.0)
                
                # Resource usage vs QS Score
                scatter = ax2.scatter(results_df['ru'], results_df['qs_score'], 
                                    alpha=0.6, c=results_df['qs_score'], 
                                    cmap='viridis', s=50)
                ax2.set_xlabel('Витрати RU', fontsize=12, fontweight='bold')
                ax2.set_ylabel('QS Score', fontsize=12, fontweight='bold')
                ax2.set_title('Залежність QS Score від витрат ресурсів', fontsize=14, fontweight='bold', pad=20)
                ax2.grid(True, alpha=0.3)
                
                # Highlight best strategy
                ax2.scatter(best['ru'], best['qs_score'], color='#E63946', s=150, 
                           label='Найкраща стратегія', edgecolors='black', linewidth=2)
                ax2.legend(fontsize=11)
                
                # Add colorbar for scatter plot
                cbar = plt.colorbar(scatter, ax=ax2)
                cbar.set_label('QS Score', fontsize=11, fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Summary statistics
                st.subheader("📋 Підсумкова статистика")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Всього комбінацій", len(results_df))
                with col2:
                    st.metric("Найкращий QS Score", f"{results_df['qs_score'].max():.3f}")
                with col3:
                    st.metric("Середній QS Score", f"{results_df['qs_score'].mean():.3f}")
                with col4:
                    st.metric("Час обчислення", f"{elapsed_time:.1f}с")
            else:
                st.warning("Не знайдено жодної валідної стратегії з заданими обмеженнями.")