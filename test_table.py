import numpy as np
import sys
import random
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

# --- IMPORTING FROM CONFIG FILE ---
# Assuming your first code block is saved as pso_configs.py
from pso_configs import CEC_CONFIGS, E, PI

# --- 1. Objective Function Implementations ---
def sphere_function(x, func_params=None): return np.sum(x**2)
def rosenbrock_function(x, func_params=None): return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
def ackley_function(x, func_params=None):
    a, b, c = 20, 0.2, 2 * PI
    D = len(x)
    term1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / D))
    term2 = -np.exp(np.sum(np.cos(c * x)) / D)
    return term1 + term2 + a + E
def rastrigin_function(x, func_params=None):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * PI * x))
def griewank_function(x, func_params=None):
    D = len(x)
    return np.sum(x**2 / 4000) - np.prod(np.cos(x / np.sqrt(np.arange(1, D + 1)))) + 1
def levy_function(x, func_params=None):
    w = 1 + (x - 1) / 4
    return np.sin(PI * w[0])**2 + np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(PI * w[:-1] + 1)**2)) + (w[-1] - 1)**2 * (1 + np.sin(2 * PI * w[-1])**2)
def shaffer_f6_function(x, func_params=None):
    sum_sq = np.sum(x**2)
    return 0.5 + (np.sin(np.sqrt(sum_sq))**2 - 0.5) / (1 + 0.001 * sum_sq)**2
def schwefel_function(x, func_params=None):
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))
def perm_function(x, func_params=None):
    beta = 10.0
    D, outer_sum = len(x), 0
    for k in range(1, D + 1):
        inner_sum = np.sum([(i**k + beta) * ((x[i-1] / i)**k - 1) for i in range(1, D + 1)])
        outer_sum += inner_sum**2
    return outer_sum
def zakharov_function(x, func_params=None):
    d = len(x)
    s1, s2 = np.sum(x**2), np.sum(0.5 * np.arange(1, d + 1) * x)
    return s1 + s2**2 + s2**4
def dixon_price_function(x, func_params=None):
    n = len(x)
    return (x[0] - 1)**2 + np.sum(np.arange(2, n + 1) * (2 * x[1:]**2 - x[:-1])**2)
def sum_squares_function(x, func_params=None):
    return np.sum(np.arange(1, len(x) + 1) * x**2)
def michalewicz_function(x, func_params=None):
    m = 10
    indices = np.arange(1, len(x) + 1)
    return -np.sum(np.sin(x) * (np.sin(indices * x**2 / np.pi))**(2 * m))
def powell_function(x, func_params=None):
    n = len(x)
    res = 0
    for i in range(0, n - 3, 4):
        res += (x[i] + 10*x[i+1])**2 + 5*(x[i+2] - x[i+3])**2 + (x[i+1] - 2*x[i+2])**4 + 10*(x[i] - x[i+3])**4
    return res

# --- 2. Internal Function Mapping ---
# This maps the logic to the config IDs
FUNC_MAP = {
    1: sphere_function, 2: rosenbrock_function, 3: ackley_function, 4: rastrigin_function,
    6: griewank_function, 7: levy_function, 8: shaffer_f6_function, 9: schwefel_function,
    10: perm_function, 11: zakharov_function, 12: dixon_price_function, 13: sum_squares_function,
    14: michalewicz_function, 15: powell_function
}

# --- 3. Core RMPSO Run Function ---
def run_pso_once(objective_function, x_range, seed, pso_params):
    D, N = pso_params['D'], pso_params['N']
    c1, c2 = pso_params['c1'], pso_params['c2']
    max_iters = pso_params['max_iterations']
    x_min, x_max = x_range

    np.random.seed(seed)
    x = np.random.uniform(x_min, x_max, size=(N, D))
    v = np.zeros((N, D))
    
    pVal = np.array([objective_function(x[i]) for i in range(N)])
    pBest_rep = [[x[i].copy()] for i in range(N)]
    
    Gval = np.min(pVal)
    gBest_rep = [x[np.argmin(pVal)].copy()]
    history = []

    for t in range(max_iters):
        w_dynamic = 0.9 - ((0.9 - 0.4) * t / max_iters)
        r1, r2 = np.random.rand(N, D), np.random.rand(N, D)
        
        for i in range(N):
            current_pBest = random.choice(pBest_rep[i])
            current_gBest = random.choice(gBest_rep)

            v[i] = w_dynamic * v[i] + c1 * r1[i] * (current_pBest - x[i]) + c2 * r2[i] * (current_gBest - x[i])
            x[i] = np.clip(x[i] + v[i], x_min, x_max)
            current_val = objective_function(x[i])
        
            if current_val < pVal[i]:
                pVal[i] = current_val
                pBest_rep[i] = [x[i].copy()]
            elif np.isclose(current_val, pVal[i]):
                if not any(np.array_equal(x[i], pos) for pos in pBest_rep[i]):
                    pBest_rep[i].append(x[i].copy())
        
            if current_val < Gval:
                Gval = current_val
                gBest_rep = [x[i].copy()]
            elif np.isclose(current_val, Gval):
                if not any(np.array_equal(x[i], pos) for pos in gBest_rep):
                    gBest_rep.append(x[i].copy())
    
        history.append(Gval)
    return Gval, history

# --- 4. Main Execution ---
# --- 4. Main Execution (Integrated with Statistical Table) ---
def main():
    print("\n--- Available Functions in Config ---")
    available_ids = sorted(CEC_CONFIGS.keys())
    headers = ["ID", "Name", "Dim", "Range"]
    table_data = [[fid, CEC_CONFIGS[fid]['name'], CEC_CONFIGS[fid]['D'], CEC_CONFIGS[fid]['range']] for fid in available_ids]
    print(tabulate(table_data, headers=headers, tablefmt="simple"))

    selection = input("\nEnter IDs to run (e.g., '1, 3, 4') or 'all': ")
    if selection.lower() == 'all':
        selected_ids = available_ids
    else:
        selected_ids = [int(x.strip()) for x in selection.split(',')]

    # --- TABLE INITIALIZATION ---
    results_table = [] 
    plt.figure(figsize=(12, 7))

    for fid in selected_ids:
        if fid not in CEC_CONFIGS: continue
        
        conf = CEC_CONFIGS[fid]
        func = FUNC_MAP[fid]
        
        final_fitnesses, histories = [], []
        print(f"Running {conf['name']} (D={conf['D']}) for {conf['N_RUNS']} runs...", end="\r")

        for run in range(conf['N_RUNS']):
            res, hist = run_pso_once(func, conf['range'], 12340 + run, conf)
            final_fitnesses.append(res)
            histories.append(hist)

        # --- DATA PROCESSING FOR TABLE ---
        fit_arr = np.array(final_fitnesses)
        results_table.append([
            conf['name'], 
            f"{np.min(fit_arr):.6e}",    # Minimum
            f"{np.mean(fit_arr):.6e}",   # Mean
            f"{np.median(fit_arr):.6e}", # Median (Added)
            f"{np.std(fit_arr):.6e}",    # Std Dev
            f"N={conf['N']}, D={conf['D']}"
        ])

        # Plot Average Convergence
        avg_hist = np.mean(histories, axis=0)
        plt.semilogy(avg_hist + 1e-20, label=f"{conf['name']}")

    # --- FINAL TABLE OUTPUT ---
    print("\n" + "="*95)
    print("PSO PERFORMANCE STATISTICAL SUMMARY")
    # Using fancy_grid for the polished look from the first script
    print(tabulate(results_table, 
                   headers=["Function Name", "Minimum", "Mean", "Median", "Std Dev", "Params"], 
                   tablefmt="fancy_grid"))
    print("="*95)

    plt.title("Convergence Curves (Log Scale)")
    plt.xlabel("Iteration"); plt.ylabel("Best Cost"); plt.legend(); plt.grid(True, which="both", ls="-", alpha=0.2); plt.show()
if __name__ == "__main__":
    main()