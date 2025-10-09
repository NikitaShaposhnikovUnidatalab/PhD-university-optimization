import pygad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from typing import Dict, Any, Optional

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
def compute_qs_score(solution, QS_WEIGHTS, keys):
    """Обчислює оцінку використовуючи той самий порядок ключів як у масиві рішення."""
    return sum(float(solution[i]) * float(QS_WEIGHTS[k]) for i, k in enumerate(keys))

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

        return compute_qs_score(solution, QS_WEIGHTS, keys)

    return fitness_func

# === Простір генів === #
def generate_gene_space(QS_INPUT, QS_DELTA, QS_MAX, QS_COST):
    gene_space = []
    for k in QS_INPUT.keys():
        low = float(QS_INPUT[k])
        # Заморожуємо якщо delta == 0 або вартість нескінченна
        if float(QS_DELTA.get(k, 0.0)) == 0.0 or QS_COST.get(k, 0.0) == float("inf"):
            gene_space.append([low])
        else:
            high = float(min(low + float(QS_DELTA[k]), float(QS_MAX[k])))
            gene_space.append({"low": low, "high": high, "step": 0.1})
    return gene_space

# === Автоматичний пошук параметрів === #
def find_optimal_parameters(
    QS_INPUT,
    QS_WEIGHTS,
    QS_MAX,
    QS_DELTA,
    QS_COST,
    MAX_RU,
    *,
    n_trials: int = 20,
    n_trials_per_eval: int = 2,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Автоматично знаходить оптимальні параметри для генетичного алгоритму
    """
    if verbose:
        print(f"🔍 Початок пошуку оптимальних параметрів: {n_trials} експериментів")
    
    def objective(trial):
        num_generations = trial.suggest_int("num_generations", 100, 500)
        sol_per_pop = trial.suggest_int("sol_per_pop", 20, 100)
        num_parents_mating = trial.suggest_int("num_parents_mating", 5, sol_per_pop // 2)
        mutation_percent_genes = trial.suggest_int("mutation_percent_genes", 5, 40)
        random_seed = trial.suggest_int("random_seed", 1, 1000)
        
        # Запускаємо кілька оцінок для стабільності
        scores = []
        for _ in range(n_trials_per_eval):
            try:
                ga = run_optimization_internal(
                    QS_INPUT, QS_WEIGHTS, QS_MAX, QS_DELTA, QS_COST, MAX_RU,
                    num_generations=num_generations,
                    sol_per_pop=sol_per_pop,
                    num_parents_mating=num_parents_mating,
                    mutation_percent_genes=mutation_percent_genes,
                    stop_criteria="saturate_10",
                    random_seed=random_seed
                )
                
                solution, qs_score, _ = ga.best_solution()
                scores.append(float(qs_score))
            except Exception as e:
                if verbose:
                    print(f"⚠️ Помилка в trial {trial.number}: {str(e)}")
                scores.append(0.0)
        
        return np.mean(scores)
    
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner()
    )
    
    study.optimize(
        objective,
        n_trials=n_trials
    )
    
    if verbose:
        print(f"✅ Пошук завершено! Найкращий QS Score: {study.best_value:.3f}")
        print(f"🎯 Найкращі параметри: {study.best_params}")
    
    return study.best_params

# === Внутрішня функція оптимізації (без пошуку параметрів) === #
def run_optimization_internal(
    QS_INPUT,
    QS_WEIGHTS,
    QS_MAX,
    QS_DELTA,
    QS_COST,
    MAX_RU,
    *,
    num_generations: int = 400,
    sol_per_pop: int = 60,
    num_parents_mating: int = 24,
    mutation_percent_genes: int = 20,
    stop_criteria: str | None = "saturate_15",
    random_seed: int | None = 42,
):
    gene_space = generate_gene_space(QS_INPUT, QS_DELTA, QS_MAX, QS_COST)
    fitness_func = make_fitness(QS_INPUT, QS_COST, QS_WEIGHTS, MAX_RU)

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=len(QS_INPUT),
        gene_space=gene_space,
        mutation_percent_genes=mutation_percent_genes,
        mutation_type="random",
        random_mutation_min_val=0,
        random_mutation_max_val=1,
        stop_criteria=stop_criteria,
        random_seed=random_seed,
    )

    ga_instance.run()
    return ga_instance

# === Основна функція оптимізації з автоматичним пошуком параметрів === #
def run_optimization(
    QS_INPUT,
    QS_WEIGHTS,
    QS_MAX,
    QS_DELTA,
    QS_COST,
    MAX_RU,
    *,
    auto_find_params: bool = True,
    n_trials: int = 15,
    n_trials_per_eval: int = 2,
    num_generations: Optional[int] = None,
    sol_per_pop: Optional[int] = None,
    num_parents_mating: Optional[int] = None,
    mutation_percent_genes: Optional[int] = None,
    stop_criteria: str | None = "saturate_15",
    random_seed: int | None = 42,
    verbose: bool = True
):
    """
    Запускає оптимізацію з автоматичним пошуком параметрів або з заданими параметрами
    
    Args:
        auto_find_params: Чи автоматично шукати оптимальні параметри
        n_trials: Кількість експериментів для пошуку параметрів
        n_trials_per_eval: Кількість оцінок на експеримент
        timeout_minutes: Максимальний час пошуку параметрів
        verbose: Чи виводити інформацію про пошук
        ... інші параметри GA
    """
    
    # Якщо потрібно автоматично знайти параметри
    if auto_find_params and all(param is None for param in [num_generations, sol_per_pop, num_parents_mating, mutation_percent_genes]):
        if verbose:
            print("🔍 Автоматичний пошук оптимальних параметрів...")
        
        optimal_params = find_optimal_parameters(
            QS_INPUT, QS_WEIGHTS, QS_MAX, QS_DELTA, QS_COST, MAX_RU,
            n_trials=n_trials,
            n_trials_per_eval=n_trials_per_eval,
            verbose=verbose
        )
        
        # Використовуємо знайдені параметри
        num_generations = optimal_params["num_generations"]
        sol_per_pop = optimal_params["sol_per_pop"]
        num_parents_mating = optimal_params["num_parents_mating"]
        mutation_percent_genes = optimal_params["mutation_percent_genes"]
        random_seed = optimal_params["random_seed"]
        
        if verbose:
            print(f"🎯 Використовую знайдені параметри: поколінь={num_generations}, популяція={sol_per_pop}, батьки={num_parents_mating}, мутації={mutation_percent_genes}%")
    
    # Використовуємо стандартні параметри, якщо не вказано інші
    if num_generations is None:
        num_generations = 400
    if sol_per_pop is None:
        sol_per_pop = 60
    if num_parents_mating is None:
        num_parents_mating = 24
    if mutation_percent_genes is None:
        mutation_percent_genes = 20
    
    # Запускаємо оптимізацію з фінальними параметрами
    return run_optimization_internal(
        QS_INPUT, QS_WEIGHTS, QS_MAX, QS_DELTA, QS_COST, MAX_RU,
        num_generations=num_generations,
        sol_per_pop=sol_per_pop,
        num_parents_mating=num_parents_mating,
        mutation_percent_genes=mutation_percent_genes,
        stop_criteria=stop_criteria,
        random_seed=random_seed
    )

def plot_progress(ga_instance):
    plt.figure(figsize=(10, 6))
    plt.plot(ga_instance.best_solutions_fitness, linewidth=2, color='#2E86AB')
    plt.xlabel("Покоління", fontsize=12, fontweight='bold')
    plt.ylabel("QS Overall Score", fontsize=12, fontweight='bold')
    plt.title("Динаміка покращення QS Score", fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
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


if __name__ == "__main__":
    QS_INPUT = {"AR": 6.5, "ER": 10.6, "FSR": 54.3, "CPF": 1.3,
                "IFR": 1.7, "ISR": 20.1, "IRN": 11.4, "EO": 4.0, "SUS": 1.6}
    QS_WEIGHTS = {"AR": 0.30, "ER": 0.15, "FSR": 0.10, "CPF": 0.20,
                  "IFR": 0.05, "ISR": 0.05, "IRN": 0.05, "EO": 0.05, "SUS": 0.05}
    QS_MAX = {"AR": 15, "ER": 20, "FSR": 70, "CPF": 3,
              "IFR": 12, "ISR": 20, "IRN": 30, "EO": 15, "SUS": 10}
    QS_DELTA = {"AR": 1.0, "ER": 1.0, "FSR": 1.0, "CPF": 0.3,
                "IFR": 2.0, "ISR": 0.0, "IRN": 5.0, "EO": 2.0, "SUS": 1.0}
    QS_COST = {"AR": 50, "ER": 45, "FSR": 20, "CPF": 15,
               "IFR": 30, "ISR": float("inf"), "IRN": 10, "EO": 10, "SUS": 5}
    MAX_RU = 100

    ga = run_optimization(QS_INPUT, QS_WEIGHTS, QS_MAX, QS_DELTA, QS_COST, MAX_RU)
    solution, qs_score, _ = ga.best_solution()
    result = dict(zip(QS_INPUT.keys(), solution))

    print("\nНайкраще рішення на 2026 рік:")
    for k in result:
        print(f"{k}: {result[k]:.2f} (було: {QS_INPUT[k]:.2f})")

    print(f"\nQS Overall Score (2026): {qs_score:.2f}")

    plot_progress(ga)