import itertools
import time
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from genetic_optimizer import (
    run_optimization,
    compute_total_ru,
    get_top_solutions,
)


def _eligible_keys_for_improvement(
    qs_input: dict[str, float],
    qs_delta: dict[str, float],
    qs_cost: dict[str, float],
) -> list[str]:
    eligible = []
    for key in qs_input.keys():
        if float(qs_delta.get(key, 0.0)) > 0.0 and float(qs_cost.get(key, 0.0)) != float("inf"):
            eligible.append(key)
    return eligible


def _freeze_all_except(target_keys: set[str], qs_delta: dict[str, float]) -> dict[str, float]:
    frozen_delta: dict[str, float] = {}
    for key, delta in qs_delta.items():
        if key in target_keys:
            frozen_delta[key] = float(delta)
        else:
            frozen_delta[key] = 0.0
    return frozen_delta


def _summarize_solution(
    solution: Sequence[float],
    qs_input: dict[str, float],
    qs_weights: dict[str, float],
) -> dict[str, float]:
    keys = list(qs_input.keys())
    contrib = {k: float(solution[i]) * float(qs_weights[k]) for i, k in enumerate(keys)}
    return contrib


def evaluate_top3_combinations(
    qs_input: dict[str, float],
    qs_weights: dict[str, float],
    qs_max: dict[str, float],
    qs_delta: dict[str, float],
    qs_cost: dict[str, float],
    max_ru: float,
    *,
    num_generations: int = 250,
    sol_per_pop: int = 48,
    num_parents_mating: int = 20,
    mutation_percent_genes: int = 20,
    stop_criteria: str | None = "saturate_10",
    random_seed: int | None = 42,
    top_k: int = 15,
) -> pd.DataFrame:
    """
    Try all 3-indicator combinations that are improvable and rank by best QS score.

    Returns a DataFrame with columns:
      - combo: tuple[str, str, str]
      - qs_score: float (best for that combo)
      - ru: float (resource units used by best solution)
      - solution: np.ndarray (best gene vector)
      - values: dict of indicator -> value (for best)
    """
    i = 0
    eligible = _eligible_keys_for_improvement(qs_input, qs_delta, qs_cost)
    print(eligible)
    if len(eligible) < 3:
        raise ValueError("Not enough eligible indicators (need at least 3 with delta>0 and finite cost).")

    results: list[dict] = []
    all_keys: list[str] = list(qs_input.keys())

    start_ts = time.time()
    for combo in itertools.combinations(eligible, 3):
        i += 1
        print(combo)
        target_keys = set(combo)
        frozen_delta = _freeze_all_except(target_keys, qs_delta)

        ga = run_optimization(
            qs_input,
            qs_weights,
            qs_max,
            frozen_delta,
            qs_cost,
            max_ru,
            num_generations=num_generations,
            sol_per_pop=sol_per_pop,
            num_parents_mating=num_parents_mating,
            mutation_percent_genes=mutation_percent_genes,
            stop_criteria=stop_criteria,
            random_seed=random_seed,
        )

        solution, qs_score, _ = ga.best_solution()
        ru = compute_total_ru(qs_input, qs_cost, solution)
        values = {k: float(solution[i]) for i, k in enumerate(all_keys)}

        results.append(
            {
                "combo": tuple(combo),
                "qs_score": float(qs_score),
                "ru": float(ru),
                "solution": np.asarray(solution, dtype=float),
                "values": values,
            }
        )
    print(i)
    elapsed_s = time.time() - start_ts

    df = pd.DataFrame(results).sort_values(by=["qs_score", "ru"], ascending=[False, True]).reset_index(drop=True)
    df.attrs["elapsed_seconds"] = elapsed_s
    return df.head(top_k)


def best_strategies_detailed(
    ga_instance,
    qs_input: dict[str, float],
    qs_cost: dict[str, float],
    qs_weights: dict[str, float],
    *,
    top_n: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper around get_top_solutions for a given GA instance.
    Returns (summary_df, contribution_df).
    """
    return get_top_solutions(ga_instance, qs_input, qs_cost, qs_weights, top_n=top_n)


def run_cli_example():
    # Example data (same as in app/genetic_optimizer.py __main__)
    qs_input = {
        "AR": 6.5,
        "ER": 10.6,
        "FSR": 54.3,
        "CPF": 1.3,
        "IFR": 1.7,
        "ISR": 20.1,
        "IRN": 11.4,
        "EO": 4.0,
        "SUS": 1.6,
    }

    qs_weights = {
        "AR": 0.30,
        "ER": 0.15,
        "FSR": 0.10,
        "CPF": 0.20,
        "IFR": 0.05,
        "ISR": 0.05,
        "IRN": 0.05,
        "EO": 0.05,
        "SUS": 0.05,
    }

    qs_max = {
        "AR": 15,
        "ER": 20,
        "FSR": 70,
        "CPF": 3,
        "IFR": 12,
        "ISR": 20,
        "IRN": 30,
        "EO": 15,
        "SUS": 10,
    }

    qs_delta = {
        "AR": 1.0,
        "ER": 1.0,
        "FSR": 1.0,
        "CPF": 0.3,
        "IFR": 2.0,
        "ISR": 0.0,
        "IRN": 5.0,
        "EO": 2.0,
        "SUS": 1.0,
    }

    qs_cost = {
        "AR": 100,
        "ER": 90,
        "FSR": 40,
        "CPF": 30,
        "IFR": 60,
        "ISR": float("inf"),
        "IRN": 20,
        "EO": 20,
        "SUS": 10,
    }

    max_ru = 200

    top_df = evaluate_top3_combinations(
        qs_input,
        qs_weights,
        qs_max,
        qs_delta,
        qs_cost,
        max_ru,
        num_generations=200,
        sol_per_pop=60,
        num_parents_mating=24,
        mutation_percent_genes=20,
        stop_criteria="saturate_12",
        random_seed=42,
        top_k=10,
    )

    print("Top-10 3-indicator strategies (by best QS score):")
    for idx, row in top_df.iterrows():
        combo = row["combo"]
        qs_score = row["qs_score"]
        ru = row["ru"]
        print(f"{idx+1:>2}. {combo} -> QS={qs_score:.3f}, RU={ru:.1f}")

    # Show the indicator values for the best combo
    if not top_df.empty:
        best = top_df.iloc[0]
        print("\nBest combo values (indicator -> value):")
        for k, v in best["values"].items():
            print(f"  {k}: {v:.2f} (was {qs_input[k]:.2f})")


if __name__ == "__main__":
    run_cli_example()


