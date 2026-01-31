import numpy as np

# --- CEC BENCHMARK CONFIGURATIONS ---
# N_RUNS: 30 (Standard for statistical significance)
# w: 0.7298, c1/c2: 1.49618 (Clerc-Kennedy Constriction Factor for stability)

CEC_CONFIGS = {
    2:  {'name': 'Rosenbrock',   'N_RUNS': 30, 'D': 5, 'N': 30,  'w': 0.7298, 'c1': 1.49618, 'c2': 1.49618, 'max_iterations': 1000, 'range': (-30.0, 30.0)},
    1:  {'name': 'Sphere',       'N_RUNS': 30, 'D': 5, 'N': 30,  'w': 0.7298, 'c1': 1.49618, 'c2': 1.49618, 'max_iterations': 1000, 'range': (-100.0, 100.0)},
    3:  {'name': 'Ackley',       'N_RUNS': 30, 'D': 5, 'N': 30, 'w': 0.7298, 'c1': 1.49618, 'c2': 1.49618, 'max_iterations': 1000, 'range': (-32.768, 32.768)},
    4:  {'name': 'Rastrigin',    'N_RUNS': 30, 'D': 5, 'N': 30, 'w': 0.7298, 'c1': 1.49618, 'c2': 1.49618, 'max_iterations': 1000, 'range': (-5.12, 5.12)},
    6:  {'name': 'Griewank',     'N_RUNS': 30, 'D': 5, 'N': 30,  'w': 0.7298, 'c1': 1.49618, 'c2': 1.49618, 'max_iterations': 1000, 'range': (-600.0, 600.0)},
    7:  {'name': 'Levy',         'N_RUNS': 30, 'D': 5, 'N': 30,  'w': 0.7298, 'c1': 1.49618, 'c2': 1.49618, 'max_iterations': 1000, 'range': (-10.0, 10.0)},
    8:  {'name': 'Shaffer F6',   'N_RUNS': 30, 'D': 5, 'N': 30,  'w': 0.7298, 'c1': 1.49618, 'c2': 1.49618, 'max_iterations': 1000, 'range': (-100.0, 100.0)},
    9:  {'name': 'Schwefel',     'N_RUNS': 30, 'D': 5, 'N': 30, 'w': 0.7298, 'c1': 1.49618, 'c2': 1.49618, 'max_iterations': 1000, 'range': (-500.0, 500.0)},
    10: {'name': 'Perm',         'N_RUNS': 30, 'D': 5, 'N': 30, 'w': 0.7298, 'c1': 1.49618, 'c2': 1.49618, 'max_iterations': 1000, 'range': (-10.0, 10.0)},
    11: {'name': 'Zakharov',     'N_RUNS': 30, 'D': 5, 'N': 30,  'w': 0.7298, 'c1': 1.49618, 'c2': 1.49618, 'max_iterations': 1000, 'range': (-10.0, 10.0)},
    12: {'name': 'Dixon-Price',  'N_RUNS': 30, 'D': 5, 'N': 30,  'w': 0.7298, 'c1': 1.49618, 'c2': 1.49618, 'max_iterations': 1000, 'range': (-10.0, 10.0)},
    13: {'name': 'Sum Squares',  'N_RUNS': 30, 'D': 5, 'N': 30,  'w': 0.7298, 'c1': 1.49618, 'c2': 1.49618, 'max_iterations': 1000, 'range': (-10.0, 10.0)},
    14: {'name': 'Michalewicz',  'N_RUNS': 30, 'D': 5, 'N': 30, 'w': 0.7298, 'c1': 1.49618, 'c2': 1.49618, 'max_iterations': 1000, 'range': (0.0, np.pi)},
    15: {'name': 'Powell',       'N_RUNS': 30, 'D': 5, 'N': 30,  'w': 0.7298, 'c1': 1.49618, 'c2': 1.49618, 'max_iterations': 1000, 'range': (-4.0, 5.0)},
}
E = np.e
PI = np.pi