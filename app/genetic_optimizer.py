import pygad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_total_ru(QS_INPUT, QS_COST, solution):
    total_ru = 0
    for i, k in enumerate(QS_INPUT.keys()):
        prev = QS_INPUT[k]
        curr = solution[i]
        delta = curr - prev
        if QS_COST[k] != float("inf") and delta > 0:
            total_ru += QS_COST[k] * delta
    return total_ru

# === QS Score === #
def compute_qs_score(solution, QS_WEIGHTS):
    return sum(solution[i] * w for i, w in enumerate(QS_WEIGHTS.values()))

# === Фітнес-функція === #
def make_fitness(QS_INPUT, QS_COST, QS_WEIGHTS, MAX_RU):
    keys = list(QS_INPUT.keys())

    def fitness_func(ga_instance, solution, solution_idx):
        x_new = dict(zip(keys, solution))
        total_ru = 0

        for k in keys:
            prev, curr = QS_INPUT[k], x_new[k]
            delta = curr - prev

            if QS_COST[k] == float("inf"):
                if delta != 0:
                    return -10000
            else:
                if delta > 0:
                    cost = QS_COST[k] * delta
                    total_ru += cost

        if total_ru > MAX_RU:
            return -1000 * (total_ru - MAX_RU)

        return compute_qs_score(solution, QS_WEIGHTS)

    return fitness_func

# === Простір генів === #
def generate_gene_space(QS_INPUT, QS_DELTA, QS_MAX):
    gene_space = []
    for k in QS_INPUT.keys():
        low = QS_INPUT[k]
        if QS_DELTA[k] == 0:
            gene_space.append([low])
        else:
            high = min(low + QS_DELTA[k], QS_MAX[k])
            gene_space.append({"low": low, "high": high, "step": 0.1})
    print("Gene space:", gene_space)
    return gene_space

# === Запуск оптимізації === #
def run_optimization(QS_INPUT, QS_WEIGHTS, QS_MAX, QS_DELTA, QS_COST, MAX_RU):
    gene_space = generate_gene_space(QS_INPUT, QS_DELTA, QS_MAX)
    fitness_func = make_fitness(QS_INPUT, QS_COST, QS_WEIGHTS, MAX_RU)

    ga_instance = pygad.GA(
        num_generations=500,
        num_parents_mating=20,
        fitness_func=fitness_func,
        sol_per_pop=45,
        num_genes=len(QS_INPUT),
        gene_space=gene_space,
        mutation_percent_genes=20,
        mutation_type="random",
        random_mutation_min_val=0,
        random_mutation_max_val=1,
        stop_criteria="saturate_10"
    )

    ga_instance.run()
    print("=== GA finished ===")
    print("Best solution:", ga_instance.best_solution())
    print("Best fitness:", ga_instance.best_solution()[1])
    return ga_instance

def plot_progress(ga_instance):
    plt.plot(ga_instance.best_solutions_fitness)
    plt.xlabel("Покоління")
    plt.ylabel("QS Overall Score")
    plt.title("Динаміка покращення QS Score")
    plt.grid(True)
    plt.show()
    
# utils
def get_top_solutions(ga_instance, QS_INPUT, QS_COST, QS_WEIGHTS, top_n=10):
    import numpy as np
    scores = []
    pop = np.asarray(ga_instance.population)
    top_n = min(top_n, len(pop))
    for sol in pop:
        fitness = ga_instance.fitness_func(ga_instance, sol, 0)
        ru = compute_total_ru(QS_INPUT, QS_COST, sol)
        scores.append((fitness, ru, sol))

    scores = sorted(scores, key=lambda x: x[0], reverse=True)[:top_n]

    rows, contrib_rows = [], []
    keys = list(QS_INPUT.keys())
    for rank, (fitness, ru, sol) in enumerate(scores, 1):
        row = {"#": rank, "QS Score": round(float(fitness), 4), "RU": round(float(ru), 2)}
        row.update({k: round(float(v), 2) for k, v in zip(keys, sol)})
        rows.append(row)

        contrib = {k: float(sol[i]) * float(QS_WEIGHTS[k]) for i, k in enumerate(keys)}
        contrib["#"] = rank
        contrib_rows.append(contrib)

    df = pd.DataFrame(rows)
    contrib_df = pd.DataFrame(contrib_rows).set_index("#").astype(float)
    contrib_df = contrib_df[keys]
    return df, contrib_df


def plot_stacked_bar(contrib_df, title="Stacked Bar внеску показників у QS Score"):
    contrib_df.plot(
        kind="bar",
        stacked=True,
        figsize=(10, 6),
        cmap="tab20"
    )
    plt.xlabel("Стратегія (топ #)")
    plt.ylabel("Внесок у QS Score")
    plt.title(title)
    plt.legend(title="Показник", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    
# def plot_stacked_bar_normalized(delta_df, title="Нормалізовані прирости показників"):
#     norm_df = delta_df.copy()
#     for col in norm_df.columns:
#         max_delta = norm_df[col].max()
#         if max_delta > 0:
#             norm_df[col] = norm_df[col] / max_delta
#         else:
#             norm_df[col] = 0.0

#     norm_df.plot(
#         kind="bar",
#         stacked=True,
#         figsize=(10, 6),
#         cmap="tab20"
#     )
#     print(norm_df)
#     plt.xlabel("Стратегія (топ #)")
#     plt.ylabel("Нормалізований приріст (0..1)")
#     plt.title(title)
#     plt.legend(title="Показник", bbox_to_anchor=(1.02, 1), loc="upper left")
#     plt.tight_layout()
#     plt.show()

if __name__ == "__main__":
    QS_INPUT = {"AR": 6.5, "ER": 10.6, "FSR": 54.3, "CPF": 1.3,
                "IFR": 1.7, "ISR": 20.1, "IRN": 11.4, "EO": 4.0, "SUS": 1.6}
    QS_WEIGHTS = {"AR": 0.30, "ER": 0.15, "FSR": 0.10, "CPF": 0.20,
                  "IFR": 0.05, "ISR": 0.05, "IRN": 0.05, "EO": 0.05, "SUS": 0.05}
    QS_MAX = {"AR": 15, "ER": 20, "FSR": 70, "CPF": 3,
              "IFR": 12, "ISR": 20, "IRN": 30, "EO": 15, "SUS": 10}
    QS_DELTA = {"AR": 1.0, "ER": 1.0, "FSR": 1.0, "CPF": 0.3,
                "IFR": 2.0, "ISR": 0.0, "IRN": 5.0, "EO": 2.0, "SUS": 1.0}
    QS_COST = {"AR": 100, "ER": 90, "FSR": 40, "CPF": 30,
               "IFR": 60, "ISR": float("inf"), "IRN": 20, "EO": 20, "SUS": 10}
    MAX_RU = 200

    ga = run_optimization(QS_INPUT, QS_WEIGHTS, QS_MAX, QS_DELTA, QS_COST, MAX_RU)
    solution, qs_score, _ = ga.best_solution()
    result = dict(zip(QS_INPUT.keys(), solution))

    print("\nНайкраще рішення на 2026 рік:")
    for k in result:
        print(f"{k}: {result[k]:.2f} (було: {QS_INPUT[k]:.2f})")

    print(f"\nQS Overall Score (2026): {qs_score:.2f}")

    # Побудова графіка
    plot_progress(ga)