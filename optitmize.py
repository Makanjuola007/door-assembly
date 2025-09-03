import pandas as pd
import numpy as np
import json
from typing import Dict, Any
from collections import defaultdict

def load_flexsim_data(
    task_data_path="task_data.csv",
    station_data_path="station_data.csv", 
    historical_path="historical_performance.csv",
    precedence_path="precedence_matrix.csv"  # You'll need to create this
) -> Dict[str, Any]:
    """
    Load data exported from FlexSim and convert to DALB optimization format
    """
    
    # Load CSV files
    task_df = pd.read_csv(task_data_path)
    station_df = pd.read_csv(station_data_path)
    historical_df = pd.read_csv(historical_path)
    
    # Load precedence relationships (create this manually or from FlexSim)
    try:
        prec_df = pd.read_csv(precedence_path)
        P = set(zip(prec_df['predecessor'], prec_df['successor']))
    except FileNotFoundError:
        print("Warning: Using default precedence relationships")
        P = {(1, 6), (1, 7), (6, 7), (4, 5), (2, 8), (3, 8), (4, 8), 
             (5, 8), (7, 8), (8, 9), (9, 10), (10, 11), (2, 12), 
             (3, 12), (10, 12), (11, 12), (12, 13)}
    
    # Extract basic sets
    I = sorted(task_df['task'].unique())
    S = sorted(station_df['station'].unique())
    T = [1, 2]  # Two shifts
    V = ["L-Base", "L-Premium", "R-Base", "R-Premium"]
    
    # Build task times p[(i,v)] from FlexSim data
    p = {}
    rbar = {}
    gamma = {}
    
    for _, row in task_df.iterrows():
        i = row['task']
        p[(i, "L-Base")] = row['L_Base_time']
        p[(i, "L-Premium")] = row['L_Premium_time'] 
        p[(i, "R-Base")] = row['R_Base_time']
        p[(i, "R-Premium")] = row['R_Premium_time']
        rbar[i] = row['rework_rate']
        gamma[i] = row['rework_overhead']
    
    # Station constraints from FlexSim
    M_s = dict(zip(station_df['station'], station_df['max_workers']))
    N_s_max = dict(zip(station_df['station'], station_df['max_tasks']))
    E_s_max = dict(zip(station_df['station'], station_df['ergonomic_limit']))
    G_s = dict(zip(station_df['station'], station_df['footprint_limit']))
    
    # Extract demand patterns and variant mix from historical data
    n_scenarios = min(len(historical_df), 50)  # Limit scenarios for tractability
    Omega = list(range(1, n_scenarios + 1))
    
    # Demand scenarios
    d = {}
    q = {}
    
    for idx, (w, row) in enumerate(zip(Omega, historical_df.iterrows())):
        _, data = row
        total_dem = data['total_demand']
        
        for t in T:
            d[(t, w)] = total_dem / 2  # Split across shifts
            
            # Variant shares
            q[(t, w)] = {
                "L-Base": data['L_Base_demand'] / total_dem,
                "L-Premium": data['L_Premium_demand'] / total_dem,
                "R-Base": data['R_Base_demand'] / total_dem,
                "R-Premium": data['R_Premium_demand'] / total_dem
            }
    
    # Station availability from historical data
    a = {}
    delta = {}
    
    for idx, (w, row) in enumerate(zip(Omega, historical_df.iterrows())):
        _, data = row
        for s in S:
            avail_col = f'station_{s}_availability'
            if avail_col in data:
                for t in T:
                    a[(s, t, w)] = data[avail_col]
                    # Setup loss from station data
                    setup_base = station_df[station_df['station'] == s]['setup_loss_base'].iloc[0]
                    delta[(s, t, w)] = setup_base * (1 + np.random.normal(0, 0.2))
    
    # Generate rework scenarios around historical averages
    r = {}
    for i in I:
        base_rework = rbar[i]
        for t in T:
            for w in Omega:
                # Add some variability around historical rework rates
                r[(i, t, w)] = max(0.0, min(1.0, 
                    base_rework + np.random.normal(0, base_rework * 0.1)))
    
    # Time parameters
    A = {1: 8 * 60, 2: 8 * 60}  # 8-hour shifts in minutes
    tau = {(t, w): A[t] / d[(t, w)] for t in T for w in Omega}
    
    # Baseline variant shares (weighted average from historical)
    tilde_q = {}
    for v in V:
        v_col = v.replace("-", "_") + "_demand"
        if v_col in historical_df.columns:
            total_v = historical_df[v_col].sum()
            total_all = sum(historical_df[col].sum() 
                          for col in historical_df.columns 
                          if 'demand' in col and col != 'total_demand')
            tilde_q[v] = total_v / total_all if total_all > 0 else 0.25
        else:
            tilde_q[v] = 0.25  # default equal split
    
    # Task ergonomic and footprint factors (you may need to add these to FlexSim)
    E_i = {i: np.random.randint(1, 5) for i in I}  # placeholder
    g_i = {i: np.random.randint(1, 4) for i in I}  # placeholder
    
    # Cost parameters (adjust based on your economics)
    c_st = 200.0    # Station opening cost per period
    c_wk = 60.0     # Worker cost per person per period  
    c_ot = 1.2      # Overtime cost multiplier
    c_sh = 4.0      # Shortage penalty per minute/door
    c_wb = 0.5      # Workload balance penalty
    H_max = 2.0     # Max overtime per station-period
    
    # Service level and risk parameters
    alpha = 0.05              # 5% scenarios allowed to violate
    lam_cvar = 0.0           # CVaR penalty weight
    Gamma_t = {t: 2 for t in T}  # Robust budget
    
    # Big-M calculation
    max_time_i = {
        i: max(p[(i, v)] * (1.0 + gamma[i] * max(r[(i, t, w)] for t in T for w in Omega)) 
               for v in V)
        for i in I
    }
    M = sum(max_time_i.values())
    
    return {
        'I': I, 'S': S, 'T': T, 'Omega': Omega, 'V': V, 'P': P,
        'p': p, 'tilde_q': tilde_q, 'rbar': rbar, 'gamma': gamma,
        'A': A, 'd': d, 'tau': tau, 'q': q, 'r': r, 'a': a, 'delta': delta,
        'M_s': M_s, 'N_s_max': N_s_max, 'E_i': E_i, 'g_i': g_i, 
        'E_s_max': E_s_max, 'G_s': G_s,
        'c_st': c_st, 'c_wk': c_wk, 'c_ot': c_ot, 'c_sh': c_sh, 'c_wb': c_wb, 'H_max': H_max,
        'alpha': alpha, 'lam_cvar': lam_cvar, 'Gamma_t': Gamma_t, 'M': M
    }


def export_solution_to_flexsim(model, VAR, params, filename="dalb_solution.json"):
    """
    Export optimization solution in format that FlexSim can import
    """
    I, S = params['I'], params['S']
    x = VAR['x']
    y = VAR['y'] 
    mheads = VAR['m']
    
    if model.status not in [gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL]:
        print("No optimal solution to export")
        return
    
    solution = {
        "stations": {},
        "task_assignments": {},
        "summary": {
            "total_cost": model.ObjVal,
            "open_stations": [s for s in S if y[s].X > 0.5],
            "total_workers": sum(int(round(mheads[s].X)) for s in S if y[s].X > 0.5)
        }
    }
    
    # Station configuration
    for s in S:
        if y[s].X > 0.5:
            solution["stations"][s] = {
                "open": True,
                "workers": int(round(mheads[s].X)),
                "assigned_tasks": [i for i in I if x[i, s].X > 0.5],
                "workload": VAR['w'][s].X
            }
        else:
            solution["stations"][s] = {"open": False}
    
    # Task assignments (reverse lookup)
    for i in I:
        for s in S:
            if x[i, s].X > 0.5:
                solution["task_assignments"][i] = s
                break
    
    # Save to JSON file that FlexSim can read
    with open(filename, 'w') as f:
        json.dump(solution, f, indent=2)
    
    print(f"Solution exported to {filename}")
    return solution


# Main integration function
def run_integrated_optimization(flexsim_data_dir="./flexsim_exports/"):
    """
    Complete integration workflow
    """
    import os
    
    # Load FlexSim data
    print("Loading FlexSim data...")
    data = load_flexsim_data(
        task_data_path=os.path.join(flexsim_data_dir, "task_data.csv"),
        station_data_path=os.path.join(flexsim_data_dir, "station_data.csv"),
        historical_path=os.path.join(flexsim_data_dir, "historical_performance.csv")
    )
    
    # Import the DALB model builder (assuming the original code is in dalb_model.py)
    from dalb_model import build_dalb_model, compute_kpis, report_solution
    
    # Build and solve optimization model
    print("Building optimization model...")
    model, VAR = build_dalb_model(
        data["I"], data["S"], data["T"], data["Omega"], data["V"], data["P"],
        data["p"], data["tilde_q"], data["rbar"], data["gamma"],
        data["A"], data["d"], data["tau"], data["q"], data["r"], data["a"], data["delta"],
        data["M_s"], data["N_s_max"], data["E_i"], data["g_i"], data["E_s_max"], data["G_s"],
        data["c_st"], data["c_wk"], data["c_ot"], data["c_sh"], data["c_wb"],
        data["H_max"], data["alpha"], data["M"],
        use_chance=True, use_cvar=False, lam_cvar=0.0
    )
    
    print("Solving optimization model...")
    model.optimize()
    
    # Compute KPIs and report
    kpis = compute_kpis(model, VAR, data, True, data["alpha"], False, 0.0)
    report_solution(model, VAR, data, True, data["alpha"], False, False, 0.0, kpis)
    
    # Export solution back to FlexSim
    solution = export_solution_to_flexsim(model, VAR, data, 
                                        filename=os.path.join(flexsim_data_dir, "optimal_solution.json"))
    
    return model, VAR, data, kpis, solution


if __name__ == "__main__":
    # Run the integrated optimization
    model, VAR, data, kpis, solution = run_integrated_optimization()