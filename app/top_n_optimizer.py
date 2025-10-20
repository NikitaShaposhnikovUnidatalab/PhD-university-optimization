import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
from itertools import combinations
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from genetic_optimizer import run_optimization, compute_total_ru, save_experiment_to_session
from lp import optimize_qs_pulp

def run_top_n_ga_optimization(eligible, num_indicators, num_generations, sol_per_pop, num_parents_mating, mutation_percent_genes, QS_INPUT, QS_WEIGHTS, QS_MAX, QS_DELTA, QS_COST, MAX_RU, current_qs, auto_find_params=False, n_trials=10):
    """Запускає GA оптимізацію для всіх комбінацій показників"""
    with st.spinner("Обчислюю найкращі комбінації з GA..."):
        start_time = time.time()
        
        results = []
        all_keys = list(QS_INPUT.keys())
        
        total_combinations = len(list(combinations(eligible, num_indicators)))
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, combo in enumerate(combinations(eligible, num_indicators)):
            status_text.text(f"Обробляю комбінацію {i+1}/{total_combinations}: {combo}")
            
            target_keys = set(combo)
            frozen_delta = {k: (float(QS_DELTA[k]) if k in target_keys else 0.0) for k in all_keys}
            
            if auto_find_params:
                ga = run_optimization(
                    QS_INPUT,
                    QS_WEIGHTS,
                    QS_MAX,
                    frozen_delta,
                    QS_COST,
                    MAX_RU,
                    auto_find_params=True,
                    n_trials=n_trials,
                    stop_criteria="saturate_10",
                    verbose=False
                )
            else:
                ga = run_optimization(
                    QS_INPUT,
                    QS_WEIGHTS,
                    QS_MAX,
                    frozen_delta,
                    QS_COST,
                    MAX_RU,
                    auto_find_params=False,
                    num_generations=num_generations,
                    sol_per_pop=sol_per_pop,
                    num_parents_mating=num_parents_mating,
                    mutation_percent_genes=mutation_percent_genes,
                    stop_criteria="saturate_10",
                    random_seed=42,
                    verbose=False
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
                "algorithm": "GA"
            })
            
            progress_bar.progress((i + 1) / total_combinations)
        
        results_df = pd.DataFrame(results).sort_values(
            by=["qs_score", "ru"], 
            ascending=[False, True]
        ).reset_index(drop=True)
        
        elapsed_time = time.time() - start_time
        status_text.text(f"✅ GA завершено за {elapsed_time:.1f} секунд")
        progress_bar.empty()
        
        # Зберігаємо дані про топ-N експеримент в сесії
        if not results_df.empty:
            best_result = results_df.iloc[0]
            experiment = save_experiment_to_session(
                algorithm="GA_TopN",
                current_qs=current_qs,
                qs_score=best_result['qs_score'],
                ru_used=best_result['ru'],
                execution_time=elapsed_time,
                solution_details={
                    "best_combo": list(best_result['combo']),
                    "total_combinations_tested": len(results_df),
                    "algorithm": "GA"
                },
                comparison_metrics={
                    "improvement": best_result['qs_score'] - current_qs,
                    "improvement_percent": ((best_result['qs_score'] - current_qs) / current_qs * 100) if current_qs > 0 else 0,
                    "efficiency": (best_result['qs_score'] - current_qs) / best_result['ru'] if best_result['ru'] > 0 else 0,
                    "budget_utilization": best_result['ru'] / MAX_RU if MAX_RU > 0 else 0,
                    "current_qs": current_qs
                },
                improved_indicators=list(best_result['combo']),
                QS_INPUT=QS_INPUT,
                solution=best_result['solution']
            )
            
            # Зберігаємо експеримент для AI аналізу
            st.session_state["last_ga_topn_experiment"] = experiment
        
        display_top_n_results(results_df, current_qs, MAX_RU, elapsed_time, "GA", QS_INPUT, QS_WEIGHTS)

def run_top_n_lp_optimization(eligible, num_indicators, QS_INPUT, QS_WEIGHTS, QS_MAX, QS_DELTA, QS_COST, MAX_RU, current_qs):
    """Запускає LP оптимізацію для всіх комбінацій показників"""
    with st.spinner("Обчислюю найкращі комбінації з LP..."):
        start_time = time.time()
        
        results = []
        all_keys = list(QS_INPUT.keys())
        
        total_combinations = len(list(combinations(eligible, num_indicators)))
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, combo in enumerate(combinations(eligible, num_indicators)):
            status_text.text(f"Обробляю комбінацію {i+1}/{total_combinations}: {combo}")
            
            try:
                x_2026, qs_score_lp, df_lp = optimize_qs_pulp(
                    QS_INPUT=QS_INPUT,
                    QS_WEIGHTS=QS_WEIGHTS,
                    QS_MAX=QS_MAX,
                    QS_DELTA=QS_DELTA,
                    QS_COST=QS_COST,
                    MAX_RU=MAX_RU,
                    selected_indicators=list(combo),
                )
                
                deltas = {k: float(x_2026[k]) - float(QS_INPUT[k]) for k in QS_INPUT.keys()}
                ru_used = sum(
                    (deltas[k] * float(QS_COST[k])) if QS_COST[k] < float("inf") else 0.0
                    for k in QS_INPUT.keys()
                )
                
                values = {k: float(x_2026[k]) for k in all_keys}
                
                results.append({
                    "combo": combo,
                    "qs_score": float(qs_score_lp),
                    "ru": float(ru_used),
                    "solution": [float(x_2026[k]) for k in all_keys],
                    "values": values,
                    "algorithm": "LP"
                })
                
            except Exception as e:
                st.warning(f"⚠️ Помилка LP для комбінації {combo}: {str(e)}")
                results.append({
                    "combo": combo,
                    "qs_score": 0.0,
                    "ru": 0.0,
                    "solution": [float(QS_INPUT[k]) for k in all_keys],
                    "values": {k: float(QS_INPUT[k]) for k in all_keys},
                    "algorithm": "LP (помилка)"
                })
            
            progress_bar.progress((i + 1) / total_combinations)
        
        results_df = pd.DataFrame(results)
        results_df = results_df[results_df['qs_score'] > 0].sort_values(
            by=["qs_score", "ru"], 
            ascending=[False, True]
        ).reset_index(drop=True)
        
        elapsed_time = time.time() - start_time
        status_text.text(f"✅ LP завершено за {elapsed_time:.1f} секунд")
        progress_bar.empty()
        
        if not results_df.empty:
            best_result = results_df.iloc[0]
            experiment = save_experiment_to_session(
                algorithm="LP_TopN",
                current_qs=current_qs,
                qs_score=best_result['qs_score'],
                ru_used=best_result['ru'],
                execution_time=elapsed_time,
                solution_details={
                    "best_combo": list(best_result['combo']),
                    "total_combinations_tested": len(results_df),
                    "algorithm": "LP"
                },
                comparison_metrics={
                    "improvement": best_result['qs_score'] - current_qs,
                    "improvement_percent": ((best_result['qs_score'] - current_qs) / current_qs * 100) if current_qs > 0 else 0,
                    "efficiency": (best_result['qs_score'] - current_qs) / best_result['ru'] if best_result['ru'] > 0 else 0,
                    "budget_utilization": best_result['ru'] / MAX_RU if MAX_RU > 0 else 0,
                    "current_qs": current_qs
                },
                improved_indicators=list(best_result['combo']),
                QS_INPUT=QS_INPUT,
                solution=best_result['solution']
            )
            
            # Зберігаємо експеримент для AI аналізу
            st.session_state["last_lp_topn_experiment"] = experiment
        
        display_top_n_results(results_df, current_qs, MAX_RU, elapsed_time, "LP", QS_INPUT, QS_WEIGHTS)

def display_top_n_results(results_df, current_qs, MAX_RU, elapsed_time, algorithm, QS_INPUT, QS_WEIGHTS):
    """Відображає результати для топ-N оптимізації"""
    st.success(f"✅ **Топ-N оптимізація ({algorithm}) завершена!**")
    st.header(f"📈 Результати топ-N оптимізації ({algorithm})")
    
    if not results_df.empty:
        best = results_df.iloc[0]
        
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
        st.markdown(f"**🔧 Алгоритм:** {best['algorithm']}")
        
        st.subheader("📊 Детальні значення найкращої стратегії")
        all_keys = list(QS_INPUT.keys())
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
        
        st.subheader("🥇 Топ-3 стратегії")
        top3_df = results_df.head(3).copy()
        top3_df['Ранг'] = range(1, 4)
        top3_df['Комбінація'] = top3_df['combo'].apply(lambda x: ', '.join(x))
        
        display_cols = ['Ранг', 'Комбінація', 'qs_score', 'ru', 'algorithm']
        st.dataframe(top3_df[display_cols].rename(columns={
            'qs_score': 'QS Score',
            'ru': 'Витрати RU',
            'algorithm': 'Алгоритм'
        }), use_container_width=True)
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
