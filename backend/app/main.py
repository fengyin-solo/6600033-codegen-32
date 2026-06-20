from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np

app = FastAPI(title="Monte Carlo API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def root():
    return {"service": "Monte Carlo API", "status": "running"}


class OptimizationConfig(BaseModel):
    assets: int = 5
    population_size: int = 100
    generations: int = 50
    expected_returns: Optional[List[float]] = None
    cov_matrix: Optional[List[List[float]]] = None
    cost_per_sample: float = 0.1
    sample_size_base: int = 100
    risk_weight: float = 0.4
    return_weight: float = 0.4
    cost_weight: float = 0.2
    seed: Optional[int] = None


class Solution(BaseModel):
    weights: List[float]
    expected_return: float
    risk: float
    sample_cost: float
    rank: int
    crowding_distance: float
    score: Optional[float] = None


class OptimizationResult(BaseModel):
    pareto_front: List[Solution]
    all_solutions: List[Solution]
    recommended: Solution
    generations: int
    population_size: int


def _generate_random_cov(n: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.standard_normal((n, n))
    cov = A @ A.T / n
    cov = cov * 0.04 / np.mean(np.diag(cov))
    np.fill_diagonal(cov, np.maximum(np.diag(cov), 0.01))
    return cov


def _evaluate(weights: np.ndarray, er: np.ndarray, cov: np.ndarray,
              cost_per_sample: float, sample_size_base: int) -> tuple:
    port_return = float(np.sum(weights * er))
    port_risk = float(np.sqrt(weights @ cov @ weights))
    diversity = 1.0 / (np.sum(weights ** 2) + 1e-8)
    sample_cost = float(cost_per_sample * sample_size_base * diversity)
    return port_return, port_risk, sample_cost


def _dominates(a: tuple, b: tuple) -> bool:
    ar, ak, ac = a
    br, bk, bc = b
    return (ar >= br and ak <= bk and ac <= bc) and (ar > br or ak < bk or ac < bc)


def _fast_non_dominated_sort(objectives: List[tuple]) -> List[List[int]]:
    n = len(objectives)
    domination_count = [0] * n
    dominates_set = [[] for _ in range(n)]
    ranks = [0] * n
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p != q:
                if _dominates(objectives[p], objectives[q]):
                    dominates_set[p].append(q)
                elif _dominates(objectives[q], objectives[p]):
                    domination_count[p] += 1
        if domination_count[p] == 0:
            ranks[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominates_set[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    ranks[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return fronts[:-1]


def _crowding_distance(objectives: List[tuple], front: List[int]) -> dict:
    dist = {i: 0.0 for i in front}
    m = len(objectives[0])

    for obj_idx in range(m):
        sorted_front = sorted(front, key=lambda i: objectives[i][obj_idx])
        if len(sorted_front) <= 2:
            for i in sorted_front:
                dist[i] = float('inf')
            continue
        dist[sorted_front[0]] = float('inf')
        dist[sorted_front[-1]] = float('inf')
        obj_min = objectives[sorted_front[0]][obj_idx]
        obj_max = objectives[sorted_front[-1]][obj_idx]
        rng = obj_max - obj_min if obj_max != obj_min else 1.0
        for k in range(1, len(sorted_front) - 1):
            dist[sorted_front[k]] += (
                objectives[sorted_front[k + 1]][obj_idx] - objectives[sorted_front[k - 1]][obj_idx]
            ) / rng
    return dist


def _tournament_selection(population: List[np.ndarray], ranks: List[int],
                          crowding: dict, k: int = 2) -> np.ndarray:
    candidates = np.random.choice(len(population), k, replace=False)
    best = candidates[0]
    for c in candidates[1:]:
        if ranks[c] < ranks[best] or (
            ranks[c] == ranks[best] and crowding.get(c, 0) > crowding.get(best, 0)
        ):
            best = c
    return population[best].copy()


def _crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    n = len(p1)
    child = np.zeros(n)
    for i in range(n):
        child[i] = p1[i] if np.random.random() < 0.5 else p2[i]
    s = np.sum(child)
    if s > 0:
        child = child / s
    else:
        child = np.ones(n) / n
    return child


def _mutate(ind: np.ndarray, rate: float = 0.1) -> np.ndarray:
    n = len(ind)
    mutant = ind.copy()
    for i in range(n):
        if np.random.random() < rate:
            mutant[i] += np.random.normal(0, 0.1)
    mutant = np.maximum(mutant, 0)
    s = np.sum(mutant)
    if s > 0:
        mutant = mutant / s
    else:
        mutant = np.ones(n) / n
    return mutant


@app.post("/api/multi-objective-optimize", response_model=OptimizationResult)
def multi_objective_optimize(config: OptimizationConfig):
    rng = np.random.default_rng(config.seed)
    n = config.assets

    if config.expected_returns and len(config.expected_returns) == n:
        er = np.array(config.expected_returns)
    else:
        er = rng.uniform(0.03, 0.25, n)

    if config.cov_matrix and len(config.cov_matrix) == n and len(config.cov_matrix[0]) == n:
        cov = np.array(config.cov_matrix)
    else:
        cov = _generate_random_cov(n, rng)

    population = []
    for _ in range(config.population_size):
        w = rng.dirichlet(np.ones(n))
        population.append(w)

    best_history = []

    for gen in range(config.generations):
        objectives = []
        for w in population:
            r, k, c = _evaluate(w, er, cov, config.cost_per_sample, config.sample_size_base)
            objectives.append((r, k, c))

        fronts = _fast_non_dominated_sort(objectives)
        ranks = [0] * len(population)
        for rank_idx, front in enumerate(fronts):
            for i in front:
                ranks[i] = rank_idx

        crowding = {}
        for front in fronts:
            cd = _crowding_distance(objectives, front)
            crowding.update(cd)

        new_population = []
        front_idx = 0
        while front_idx < len(fronts) and len(new_population) + len(fronts[front_idx]) <= config.population_size:
            for i in fronts[front_idx]:
                new_population.append(population[i])
            front_idx += 1

        if len(new_population) < config.population_size and front_idx < len(fronts):
            remaining = config.population_size - len(new_population)
            front_crowding = [(i, crowding.get(i, 0)) for i in fronts[front_idx]]
            front_crowding.sort(key=lambda x: -x[1])
            for i in range(remaining):
                new_population.append(population[front_crowding[i][0]])

        offspring = []
        for _ in range(config.population_size):
            p1 = _tournament_selection(new_population, ranks, crowding)
            p2 = _tournament_selection(new_population, ranks, crowding)
            child = _crossover(p1, p2)
            child = _mutate(child)
            offspring.append(child)

        combined = new_population + offspring
        combined_objectives = []
        for w in combined:
            r, k, c = _evaluate(w, er, cov, config.cost_per_sample, config.sample_size_base)
            combined_objectives.append((r, k, c))

        combined_fronts = _fast_non_dominated_sort(combined_objectives)
        combined_ranks = [0] * len(combined)
        combined_crowding = {}
        for rank_idx, front in enumerate(combined_fronts):
            for i in front:
                combined_ranks[i] = rank_idx
            cd = _crowding_distance(combined_objectives, front)
            combined_crowding.update(cd)

        next_population = []
        fi = 0
        while fi < len(combined_fronts) and len(next_population) + len(combined_fronts[fi]) <= config.population_size:
            for i in combined_fronts[fi]:
                next_population.append(combined[i])
            fi += 1

        if len(next_population) < config.population_size and fi < len(combined_fronts):
            remaining = config.population_size - len(next_population)
            fc = [(i, combined_crowding.get(i, 0)) for i in combined_fronts[fi]]
            fc.sort(key=lambda x: -x[1])
            for i in range(remaining):
                next_population.append(combined[fc[i][0]])

        population = next_population

    final_objectives = []
    for w in population:
        r, k, c = _evaluate(w, er, cov, config.cost_per_sample, config.sample_size_base)
        final_objectives.append((r, k, c))

    final_fronts = _fast_non_dominated_sort(final_objectives)
    final_ranks = [0] * len(population)
    for rank_idx, front in enumerate(final_fronts):
        for i in front:
            final_ranks[i] = rank_idx
    final_crowding = {}
    for front in final_fronts:
        cd = _crowding_distance(final_objectives, front)
        final_crowding.update(cd)

    r_list = [o[0] for o in final_objectives]
    k_list = [o[1] for o in final_objectives]
    c_list = [o[2] for o in final_objectives]

    def normalize(v, lst):
        mn, mx = min(lst), max(lst)
        if mx == mn:
            return 0.5
        return (v - mn) / (mx - mn)

    scores = []
    for i in range(len(population)):
        sr = normalize(r_list[i], r_list)
        sk = 1 - normalize(k_list[i], k_list)
        sc = 1 - normalize(c_list[i], c_list)
        score = config.return_weight * sr + config.risk_weight * sk + config.cost_weight * sc
        scores.append(score)

    all_solutions = []
    for i in range(len(population)):
        all_solutions.append(Solution(
            weights=population[i].tolist(),
            expected_return=round(r_list[i], 6),
            risk=round(k_list[i], 6),
            sample_cost=round(c_list[i], 6),
            rank=final_ranks[i],
            crowding_distance=round(final_crowding.get(i, 0), 6),
            score=round(scores[i], 6)
        ))

    pareto_indices = final_fronts[0] if final_fronts else []
    pareto_front = [all_solutions[i] for i in pareto_indices]

    best_idx = int(np.argmax(scores))
    recommended = all_solutions[best_idx]

    return OptimizationResult(
        pareto_front=pareto_front,
        all_solutions=all_solutions,
        recommended=recommended,
        generations=config.generations,
        population_size=config.population_size
    )

