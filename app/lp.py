import pulp
import pandas as pd
from typing import Dict, List, Tuple


def optimize_qs_pulp(
    QS_INPUT: Dict[str, float],
    QS_WEIGHTS: Dict[str, float],
    QS_MAX: Dict[str, float],
    QS_DELTA: Dict[str, float],
    QS_COST: Dict[str, float],
    MAX_RU: float,
    selected_indicators: List[str] | None = None,
) -> Tuple[Dict[str, float], float, pd.DataFrame]:
    """
    Solve a discrete linear optimization of QS score using Pulp.

    - Decision vars: integer k_k (how many 0.1 steps we take for indicator k)
    - Objective: maximize sum_k weight_k * (x0_k + 0.1 * k_k)
    - Constraints:
        * 0 <= 0.1 * k_k <= min(QS_DELTA[k], QS_MAX[k] - x0_k)
        * sum_k QS_COST[k] * (0.1 * k_k) <= MAX_RU (only for finite costs)
        * if k not selected -> k_k = 0
        * if QS_COST[k] == inf -> k_k = 0 (frozen)
    """

    keys = list(QS_INPUT.keys())
    if selected_indicators is None:
        selected_indicators = keys

    model = pulp.LpProblem("QS_Optimization", pulp.LpMaximize)

    k_vars: Dict[str, pulp.LpVariable] = {}
    for k in keys:
        # effective max increase considering QS_DELTA and QS_MAX
        max_inc = max(0.0, min(float(QS_DELTA.get(k, 0.0)), float(QS_MAX[k]) - float(QS_INPUT[k])))
        max_steps = int(round(max_inc / 0.1))

        if (k in selected_indicators) and (max_steps > 0) and (QS_COST[k] < float("inf")):
            k_vars[k] = pulp.LpVariable(f"k_{k}", lowBound=0, upBound=max_steps, cat="Integer")
        else:
            k_vars[k] = pulp.LpVariable(f"k_{k}", lowBound=0, upBound=0, cat="Integer")

    # Objective
    model += pulp.lpSum([
        float(QS_WEIGHTS[k]) * (float(QS_INPUT[k]) + 0.1 * k_vars[k]) for k in keys
    ])

    # RU budget constraint
    model += pulp.lpSum([
        float(QS_COST[k]) * 0.1 * k_vars[k] for k in keys if QS_COST[k] < float("inf")
    ]) <= float(MAX_RU)

    model.solve(pulp.PULP_CBC_CMD(msg=0))

    deltas = {k: 0.1 * pulp.value(k_vars[k]) for k in keys}
    x_2026 = {k: float(QS_INPUT[k]) + float(deltas[k]) for k in keys}
    qs_score = sum(x_2026[k] * float(QS_WEIGHTS[k]) for k in keys)

    df = pd.DataFrame({
        "Показник": keys,
        "2025": [QS_INPUT[k] for k in keys],
        "2026 (оптимізовано)": [x_2026[k] for k in keys],
        "Приріст": [deltas[k] for k in keys],
        "Витрати RU": [deltas[k] * QS_COST[k] if QS_COST[k] < float("inf") else 0 for k in keys]
    })

    return x_2026, float(qs_score), df

if __name__ == "__main__":
    QS_INPUT = {
        "AR": 6.5,
        "ER": 10.6,
        "FSR": 54.3,
        "CPF": 1.3,
        "IFR": 1.7,
        "ISR": 20.1,
        "IRN": 11.4,
        "EO": 4.0,
        "SUS": 1.6
    }
    QS_WEIGHTS = {
        "AR": 0.30,
        "ER": 0.15,
        "FSR": 0.10,
        "CPF": 0.20,
        "IFR": 0.05,
        "ISR": 0.05,
        "IRN": 0.05,
        "EO": 0.05,
        "SUS": 0.05
    }
    QS_MAX = {
        "AR": 15,
        "ER": 20,
        "FSR": 70,
        "CPF": 3,
        "IFR": 12,
        "ISR": 20,
        "IRN": 30,
        "EO": 15,
        "SUS": 10
    }
    QS_DELTA = {
        "AR": 1.0,
        "ER": 1.0,
        "FSR": 1.0,
        "CPF": 0.3,
        "IFR": 2.0,
        "ISR": 0.0,
        "IRN": 5.0,
        "EO": 2.0,
        "SUS": 1.0
    }
    QS_COST = {
        "AR": 100,
        "ER": 90,
        "FSR": 40,
        "CPF": 30,
        "IFR": 60,
        "ISR": float("inf"),
        "IRN": 20,
        "EO": 20,
        "SUS": 10
    }
    MAX_RU = 200
    x_2026, qs_score, df = optimize_qs_pulp(
        QS_INPUT=QS_INPUT,
        QS_WEIGHTS=QS_WEIGHTS,
        QS_MAX=QS_MAX,
        QS_DELTA=QS_DELTA,
        QS_COST=QS_COST,
        MAX_RU=MAX_RU
    )
    print(df)
    print(qs_score)