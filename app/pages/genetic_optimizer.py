# pages/2_genetic_optimizer.py
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from genetic_optimizer import run_optimization, plot_progress, get_top_solutions, plot_stacked_bar

st.set_page_config(page_title="QS Genetic Optimizer", layout="centered")
st.title("QS Ranking — Genetic Algorithm Optimization")

required_keys = ["QS_INPUT", "QS_WEIGHTS", "QS_MAX", "QS_DELTA", "QS_COST", "MAX_RU"]

if not all(k in st.session_state for k in required_keys):
    st.error("❌ Дані ще не введені. Спочатку налаштуйте параметри на першій сторінці.")
    st.stop()
    
QS_INPUT = st.session_state["QS_INPUT"]
QS_WEIGHTS = st.session_state["QS_WEIGHTS"]
QS_MAX = st.session_state["QS_MAX"]
QS_DELTA = st.session_state["QS_DELTA"]
QS_COST = st.session_state["QS_COST"]
MAX_RU = st.session_state["MAX_RU"]

if st.button("🚀 Запустити оптимізацію"):
    ga = run_optimization(QS_INPUT, QS_WEIGHTS, QS_MAX, QS_DELTA, QS_COST, MAX_RU)
    solution, qs_score, _ = ga.best_solution()
    result = dict(zip(QS_INPUT.keys(), solution))

    from genetic_optimizer import compute_total_ru
    total_ru = compute_total_ru(QS_INPUT, QS_COST, solution)

    st.subheader("📊 Найкраще рішення на 2026 рік")
    st.write(f"**QS Overall Score (2026): {qs_score:.2f}**")
    st.write(f"**Використано ресурсів (RU): {total_ru:.2f} / {MAX_RU}**")

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

    st.subheader("📊 Внесок показників (Stacked Bar)")
    plot_stacked_bar(contrib_df)
    st.pyplot(plt)
    plt.clf()
    
    # st.subheader("📊 Внесок показників (Normalized Stacked Bar)")
    # plot_stacked_bar_normalized(contrib_df, QS_INPUT)
    # st.pyplot(plt)
    # plt.clf()
    
    st.subheader("🔥 Heatmap стратегій")
    delta_df = top_df.set_index("#")[list(QS_INPUT.keys())] - pd.Series(QS_INPUT)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(delta_df, cmap="RdYlGn", annot=True, fmt=".2f", ax=ax)
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

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(norm_df, cmap="RdYlGn", annot=True, fmt=".2f", ax=ax)
    st.pyplot(fig)