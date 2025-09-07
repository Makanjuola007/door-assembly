import math
import random
import argparse
from collections import defaultdict
from typing import Dict, Tuple, List, Any

import numpy as np
import matplotlib.pyplot as plt

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as e:
    raise RuntimeError(
        "gurobipy not found. Install Gurobi and ensure your license is set.\n"
        "Conda: `conda install -c gurobi gurobi` (with a valid license)."
    ) from e


def avg(x):
    x = list(x)
    return sum(x) / len(x) if x else 0.0


# ---------------- Real Door Assembly Data ----------------
def create_door_assembly_data(n_stations: int = 13, n_scenarios: int = 20, seed: int = 7):
    random.seed(seed)

    # Sets - Door assembly tasks from the document
    I = list(range(1, 14))                      # tasks 1..13 (door assembly tasks)
    S = list(range(1, n_stations + 1))          # stations
    T = [1, 2]                                  # periods (shifts)
    V = ["L-Base", "L-Premium", "R-Base", "R-Premium"]  # door variants
    Omega = list(range(1, n_scenarios + 1))     # scenarios

    # Real precedence arcs from document (i ≺ j)
    P = {
        (1, 6), (1, 7), (6, 7), (4, 5),
        (2, 8), (3, 8), (4, 8), (5, 8), (7, 8),
        (8, 9), (9, 10), (10, 11),
        (2, 12), (3, 12), (10, 12), (11, 12), (12, 13)
    }

    # Real task times from document (min/door)
    task_data = {
        # i: (task_name, base_time, premium_time, rework_prob, rework_overhead)
        1: ("Window regulator install", 1.8, 2.0, 0.30, 0.30), #
        2: ("Latch/lock module", 1.5, 1.7, 0.25, 0.25), #
        3: ("Handles & linkages", 1.2, 1.5, 0.20, 0.20),#
        4: ("Wiring harness", 2.5, 3.2, 0.40, 0.35),#
        5: ("Speaker module", 0.9, 1.1, 0.18, 0.20),#
        6: ("Window glass insert", 1.7, 1.9, 0.35, 0.35),#
        7: ("Regulator–glass tie", 0.8, 0.9, 0.25, 0.25),#
        8: ("Vapor barrier", 1.3, 1.6, 0.22, 0.25),#
        9: ("Weatherstrip / seals", 1.0, 1.1, 0.15, 0.15),#
        10: ("Trim panel & clips", 2.0, 2.4, 0.30, 0.30),#
        11: ("Switch & connections", 0.8, 0.9, 0.15, 0.20),#
        12: ("Torque audits", 0.7, 0.7, 0.10, 0.10),#
        13: ("End-of-line functional test", 1.5, 1.8, 0.50, 0.50)#
    }

    # Extract times and rework data
    base_times = {i: task_data[i][1] for i in I}
    prem_times = {i: task_data[i][2] for i in I}
    rbar = {i: task_data[i][3] for i in I}        # baseline rework probability
    gamma = {i: task_data[i][4] for i in I}       # rework overhead factor

    # p[(i,v)] - task times by variant
    p = {}
    for i in I:
        for v in V:
            if "Base" in v:
                p[(i, v)] = base_times[i]
            else:  # Premium
                p[(i, v)] = prem_times[i]

    # Real baseline variant share from document
    tilde_q = {"L-Base": 0.30, "L-Premium": 0.20, "R-Base": 0.25, "R-Premium": 0.25}

    # Period time A_t (min); stochastic demand d_{t,ω}; τ_{t,ω} = A_t / d_{t,ω}
    A = {1: 8 * 60, 2: 8 * 60}  # two 8h shifts
    
    # More realistic door production demand (doors per shift)
    d = {(t, w): random.randint(400, 600) for t in T for w in Omega}
    tau = {(t, w): A[t] / d[(t, w)] for t in T for w in Omega}

    # Scenario variant shares q_{t,ω,v} ~ Dirichlet around tilde_q
    def dirichlet_from_probs(probs, kappa=200):
        alpha = [kappa * probs[v] for v in V]
        samples = [random.gammavariate(a, 1.0) for a in alpha]
        s = sum(samples)
        return {v: samples[i] / s for i, v in enumerate(V)}

    q = {(t, w): dirichlet_from_probs(tilde_q, kappa=200) for t in T for w in Omega}

    # Scenario rework r_{i,t,ω} ~ around rbar_i with realistic variation
    r = {}
    for i in I:
        for t in T:
            for w in Omega:
                # Add scenario variation around baseline rework probability
                variation = random.uniform(-0.005, 0.005)
                r[(i, t, w)] = max(0.0, min(1.0, rbar[i] + variation))

    # Station availability a_{s,t,ω} - realistic manufacturing availability
    a = {}
    for s in S:
        for t in T:
            for w in Omega:
                # Higher availability for door assembly (well-established process)
                base_avail = 0.92 + 0.03 * random.random()  # 92-95% base
                noise = random.normalvariate(0, 0.02)
                a[(s, t, w)] = max(0.80, min(0.98, base_avail + noise))

    # Setup/variant loss δ_{s,t,ω}
    delta = {}
    for s in S:
        for t in T:
            for w in Omega:
                # Lower setup losses for door assembly (less complex than engine)
                base_loss = 0.03 + 0.01 * (1 if s in (1, n_stations) else 0)  # slightly higher at ends
                delta[(s, t, w)] = max(0.0, random.normalvariate(base_loss, 0.008))

    # Station constraints - door assembly specific
    M_s = {s: 4 for s in S}               # max 4 workers per station (door assembly)
    N_s_max = {s: 5 for s in S}           # max 5 tasks colocated (reasonable for door tasks)
    
    # Ergonomic scores based on door assembly task complexity
    E_i = {}
    for i in I:
        task_name = task_data[i][0]
        if "glass" in task_name.lower() or "trim" in task_name.lower():
            E_i[i] = 4  # High ergonomic load for glass/trim handling
        elif "harness" in task_name.lower() or "vapor" in task_name.lower():
            E_i[i] = 3  # Medium-high for complex assembly
        elif "test" in task_name.lower() or "torque" in task_name.lower():
            E_i[i] = 2  # Medium for testing/auditing
        else:
            E_i[i] = 2  # Standard for other tasks
    
    # Footprint scores
    g_i = {}
    for i in I:
        task_name = task_data[i][0]
        if "glass" in task_name.lower() or "trim" in task_name.lower():
            g_i[i] = 3  # Large footprint for glass/trim
        elif "test" in task_name.lower():
            g_i[i] = 3  # Test equipment needs space
        else:
            g_i[i] = 2  # Standard footprint

    E_s_max = {s: 15 for s in S}          # Station ergonomic limit
    G_s = {s: 12 for s in S}              # Station footprint limit

    # Cost parameters - adjusted for door assembly
    c_st, c_wk = 250.0, 65.0              # station open, per worker (higher skill)
    c_ot, c_sh = 1.5, 5.0                 # overtime, shortage per (min/door)
    c_wb = 0.8                             # smoothness penalty
    H_max = 2.5                            # overtime cap per (open station, period)

    # Service level / risk parameters
    alpha = 0.05                           # <= 5% scenarios may violate per period
    lam_cvar = 0.0                         # set >0.0 to enable CVaR penalty
    Gamma_t = {t: 2 for t in T}            # robust budget (#stations worst-case)

    # Big-M: safe upper bound on total shortage per period (min/door)
    max_time_i = {}
    for i in I:
        max_rework = max(r[(i, t, w)] for t in T for w in Omega)
        max_time_i[i] = max(p[(i, v)] * (1.0 + gamma[i] * max_rework) for v in V)
    M = sum(max_time_i.values())

    return dict(
        I=I, S=S, T=T, Omega=Omega, V=V, P=P,
        p=p, tilde_q=tilde_q, rbar=rbar, gamma=gamma,
        A=A, d=d, tau=tau, q=q, r=r, a=a, delta=delta,
        M_s=M_s, N_s_max=N_s_max, E_i=E_i, g_i=g_i, E_s_max=E_s_max, G_s=G_s,
        c_st=c_st, c_wk=c_wk, c_ot=c_ot, c_sh=c_sh, c_wb=c_wb, H_max=H_max,
        alpha=alpha, lam_cvar=lam_cvar, Gamma_t=Gamma_t, M=M,
        task_data=task_data  # Include task descriptions for reporting
    )


# ---------------- Model Builder (Equations from document) ----------------
def build_dalb_model(
    I, S, T, Omega, V, P,
    p, tilde_q, rbar, gamma, A, d, tau, q, r, a, delta,
    M_s, N_s_max, E_i, g_i, E_s_max, G_s,
    c_st, c_wk, c_ot, c_sh, c_wb, H_max, alpha, M,
    use_chance=True, use_cvar=False, lam_cvar=0.0,
    use_robust=False, Gamma_t=None, log_to_console=True
):
    m = gp.Model("DALB_Door_Assembly")
    if not log_to_console:
        m.Params.OutputFlag = 0

    # Derived times from document: (D2) tilde_p_i and (D1) phat_{i,t,ω}
    tilde_p = {
        i: sum(tilde_q[v] * p[(i, v)] * (1.0 + gamma[i] * rbar[i]) for v in V)
        for i in I
    }
    phat = {
        (i, t, w): sum(q[(t, w)][v] * p[(i, v)] * (1.0 + gamma[i] * r[(i, t, w)]) for v in V)
        for i in I for t in T for w in Omega
    }

    # Decision Variables (from document notation)
    x = m.addVars(I, S, vtype=GRB.BINARY, name="x")             # assignment (3.2)
    y = m.addVars(S, vtype=GRB.BINARY, name="y")                # station open
    mheads = m.addVars(S, vtype=GRB.INTEGER, lb=0, name="m")    # workers (3.4)

    w_s = m.addVars(S, lb=0.0, name="w")                        # workload (3.6)
    wbar = m.addVar(lb=0.0, name="wbar")                        # mean workload (3.7)
    z = m.addVars(S, lb=0.0, name="z")                          # abs deviation (3.8)

    h = m.addVars(S, T, Omega, lb=0.0, name="h")                # overtime (3.10)
    xi = m.addVars(S, T, Omega, lb=0.0, name="xi")              # shortage slack (ξ in doc)
    s_loss = m.addVars(S, T, Omega, lb=0.0, name="s_loss")      # setup/variant loss (3.9a)

    vflag = None
    if use_chance:
        vflag = m.addVars(T, Omega, vtype=GRB.BINARY, name="vflag")   # (3.13–3.14)

    eta = u = None
    if use_cvar and lam_cvar > 0.0:
        eta = m.addVars(T, lb=-GRB.INFINITY, name="eta")              # (3.16)
        u = m.addVars(T, Omega, lb=0.0, name="u")                     # (3.17)

    hR = xiR = None
    bar_c = Delta = {}
    if use_robust:
        if Gamma_t is None:
            Gamma_t = {t: 0 for t in T}
        c_stw = {(s, t, w): a[(s, t, w)] * tau[(t, w)] for s in S for t in T for w in Omega}
        bar_c = {(s, t): avg(c_stw[(s, t, w)] for w in Omega) for s in S for t in T}
        # deviation = mean - P10
        for s in S:
            for t in T:
                vals = sorted(c_stw[(s, t, w)] for w in Omega)
                p10 = vals[max(0, int(0.1 * len(vals)) - 1)]
                Delta[(s, t)] = max(0.0, bar_c[(s, t)] - p10)
        hR = m.addVars(S, T, lb=0.0, name="hR")
        xiR = m.addVars(S, T, lb=0.0, name="xiR")

    # Objective (3.1) + optional CVaR (3.16)
    recourse = gp.quicksum(c_ot * h[s, t, w] + c_sh * xi[s, t, w]
                           for s in S for t in T for w in Omega) / len(Omega)
    obj = (c_st * gp.quicksum(y[s] for s in S) +
           c_wk * gp.quicksum(mheads[s] for s in S) +
           recourse +
           c_wb * gp.quicksum(z[s] for s in S))
    if use_cvar and lam_cvar > 0.0:
        obj += lam_cvar * gp.quicksum(eta[t] + (1.0 / ((1.0 - alpha) * len(Omega))) *
                                      gp.quicksum(u[t, w] for w in Omega) for t in T)
    if use_robust:
        obj += 0.10 * gp.quicksum(hR[s, t] for s in S for t in T) + \
               0.25 * gp.quicksum(xiR[s, t] for s in S for t in T)
    m.setObjective(obj, GRB.MINIMIZE)

    # Constraints from document
    m.addConstrs((gp.quicksum(x[i, s] for s in S) == 1 for i in I), name="assign_3_2")               # (3.2)
    m.addConstrs((gp.quicksum(x[i, s] for i in I) <= N_s_max[s] * y[s] for s in S), name="pack_3_3") # (3.3)
    m.addConstrs((mheads[s] <= M_s[s] * y[s] for s in S), name="bind_3_4")                           # (3.4)
    m.addConstrs((mheads[s] >= y[s] for s in S), name="bind_3_4a")                                    # (3.4a)
    
    # Precedence constraints (3.5) - using station indices
    for (i, j) in P:
        m.addConstr((gp.quicksum(s * x[i, s] for s in S) <= gp.quicksum(s * x[j, s] for s in S)),
                    name=f"prec_3_5_{i}_{j}")
    
    m.addConstrs((w_s[s] == gp.quicksum(tilde_p[i] * x[i, s] for i in I) for s in S), name="work_3_6") # (3.6)
    m.addConstr(len(S) * wbar == gp.quicksum(w_s[s] for s in S), name="wbar_3_7")                     # (3.7)
    m.addConstrs((z[s] >= w_s[s] - wbar for s in S), name="zpos_3_8")                                 # (3.8)
    m.addConstrs((z[s] >= wbar - w_s[s] for s in S), name="zneg_3_8")                                 # (3.8)
    
    # Capacity constraint (3.9) - main feasibility constraint
    m.addConstrs((
        gp.quicksum(phat[(i, t, w)] * x[i, s] for i in I)
        <= a[(s, t, w)] * tau[(t, w)] * mheads[s] - s_loss[(s, t, w)] + h[(s, t, w)] + xi[(s, t, w)]
        for s in S for t in T for w in Omega
    ), name="cap_3_9")
    
    m.addConstrs((s_loss[(s, t, w)] <= delta[(s, t, w)] * mheads[s] for s in S for t in T for w in Omega),
                 name="loss_3_9a")                                                                     # (3.9a)
    m.addConstrs((h[(s, t, w)] <= H_max * y[s] for s in S for t in T for w in Omega), name="otcap_3_10")# (3.10)
    m.addConstrs((gp.quicksum(E_i[i] * x[i, s] for i in I) <= E_s_max[s] for s in S), name="ergo_3_11") # (3.11)
    m.addConstrs((gp.quicksum(g_i[i] * x[i, s] for i in I) <= G_s[s] for s in S), name="foot_3_12")     # (3.12)
    
    if use_chance:                                                                                     # (3.13)-(3.14)
        m.addConstrs((gp.quicksum(xi[(s, t, w)] for s in S) <= M * vflag[(t, w)]
                      for t in T for w in Omega), name="chance_3_13")
        m.addConstrs((gp.quicksum(vflag[(t, w)] for w in Omega) <= alpha * len(Omega)
                      for t in T), name="chance_3_14")
    if use_cvar and lam_cvar > 0.0:                                                                    # (3.17)
        m.addConstrs((u[(t, w)] >= gp.quicksum(xi[(s, t, w)] for s in S) - eta[t]
                      for t in T for w in Omega), name="cvar_3_17")
    if use_robust:                                                                                     # (3.18)
        for s in S:
            for t in T:
                m.addConstr(
                    gp.quicksum(tilde_p[i] * x[i, s] for i in I)
                    <= (bar_c[(s, t)] - Gamma_t[t] * Delta[(s, t)]) * mheads[s] + hR[(s, t)] + xiR[(s, t)],
                    name=f"robust_3_18_{s}_{t}"
                )

    # Solver parameters
    m.Params.MIPGap = 0.005
    m.Params.TimeLimit = 600
    m.Params.Seed = 777

    return m, {
        "x": x, "y": y, "m": mheads, "w": w_s, "wbar": wbar, "z": z,
        "h": h, "xi": xi, "s_loss": s_loss,
        "vflag": vflag, "eta": eta, "u": u,
        "phat": phat, "tilde_p": tilde_p,
        "use_robust": use_robust, "hR": (hR if use_robust else None),
        "xiR": (xiR if use_robust else None),
        "bar_c": (bar_c if use_robust else None), "Delta": (Delta if use_robust else None)
    }


# ---------------- KPI Computation ----------------
def compute_kpis(model, VAR, params, use_chance, alpha, use_cvar, lam_cvar):
    I, S, T, Omega = params["I"], params["S"], params["T"], params["Omega"]
    a, tau, p, q, gamma, r = params["a"], params["tau"], params["p"], params["q"], params["gamma"], params["r"]

    x = VAR["x"]; y = VAR["y"]; mheads = VAR["m"]
    w = VAR["w"]; wbar = VAR["wbar"]; z = VAR["z"]
    h = VAR["h"]; xi = VAR["xi"]; s_loss = VAR["s_loss"]
    phat = VAR["phat"]; vflag = VAR["vflag"]
    tilde_p = VAR["tilde_p"]

    open_stations = [s for s in S if y[s].X > 0.5]
    w_vals = [w[s].X for s in open_stations] if open_stations else []
    z_vals = [z[s].X for s in open_stations] if open_stations else []

    # Smoothness metrics
    wbar_val = wbar.X
    smooth_abs_dev = sum(z_vals)
    smooth_cv = (np.std(w_vals) / (wbar_val + 1e-9)) if w_vals else 0.0

    # Expected overtime/shortage per station and totals (min/door)
    Eh_by_s = {s: avg(h[s, t, w_].X for t in T for w_ in Omega) for s in S}
    Exi_by_s = {s: avg(xi[s, t, w_].X for t in T for w_ in Omega) for s in S}
    Eh_total = sum(Eh_by_s.values())
    Exi_total = sum(Exi_by_s.values())

    # Service level per period (fraction of scenarios with zero total shortage)
    service_by_t = {}
    for t in T:
        zero_count = 0
        for w_ in Omega:
            tot_short = sum(xi[s, t, w_].X for s in S)
            if tot_short <= 1e-6:
                zero_count += 1
        service_by_t[t] = zero_count / len(Omega)

    # Empirical CVaR_α of shortage (from solution slacks), per period
    cvar_by_t = {}
    if use_cvar and lam_cvar > 0.0:
        for t in T:
            tail = sorted(sum(xi[s, t, w_].X for s in S) for w_ in Omega)
            q_index = int(math.ceil((1.0 - alpha) * len(tail))) - 1
            q_index = max(0, min(q_index, len(tail) - 1))
            thresh = tail[q_index]
            tail_excess = [v for v in tail if v >= thresh]
            cvar_by_t[t] = avg(tail_excess)
    else:
        for t in T:
            cvar_by_t[t] = np.nan

    # Pre-slack utilization p95 per station (LHS / core capacity, clipped)
    p95_util_by_s = {}
    for s in open_stations:
        util_samples = []
        for t in T:
            for w_ in Omega:
                lhs = sum(phat[(i, t, w_)] * x[i, s].X for i in I)
                core_cap = a[(s, t, w_)] * tau[(t, w_)] * mheads[s].X - s_loss[(s, t, w_)].X
                core_cap = max(core_cap, 1e-9)
                util = max(0.0, min(2.0, lhs / core_cap))
                util_samples.append(util)
        p95_util_by_s[s] = float(np.percentile(util_samples, 95)) if util_samples else np.nan

    # Chance-constraint flags used
    flagged_by_t = {}
    if use_chance and vflag is not None:
        for t in T:
            flagged_by_t[t] = sum(1 for w_ in Omega if vflag[(t, w_)].X > 0.5)
    else:
        for t in T:
            flagged_by_t[t] = 0

    # ---- Line Efficiency & Balance Loss ----
    sum_ws = sum(w_vals)
    max_ws = max(w_vals) if w_vals else 1.0
    n_open = len(open_stations) if open_stations else 1
    line_efficiency = (sum_ws / (n_open * max_ws)) if n_open > 0 else np.nan
    balance_loss = 1.0 - line_efficiency if not np.isnan(line_efficiency) else np.nan

    # ---- Bottleneck Frequency (pre-slack) ----
    bottleneck_count = {s: 0 for s in S}
    total_scen = len(T) * len(Omega)
    for t in T:
        for w_ in Omega:
            util = {}
            max_util = -1.0
            for s in open_stations:
                lhs = sum(phat[(i, t, w_)] * x[i, s].X for i in I)
                core_cap = a[(s, t, w_)] * tau[(t, w_)] * mheads[s].X - s_loss[(s, t, w_)].X
                core_cap = max(core_cap, 1e-9)
                util[s] = lhs / core_cap
                if util[s] > max_util:
                    max_util = util[s]
            for s in open_stations:
                if abs(util[s] - max_util) <= 1e-6:
                    bottleneck_count[s] += 1
    bottleneck_freq = {s: (bottleneck_count[s] / total_scen) if total_scen > 0 else np.nan for s in S}

    # ---- Rework Load Share (%) per station ----
    rework_minutes = {s: 0.0 for s in S}
    total_minutes = {s: 0.0 for s in S}
    for s in S:
        if y[s].X <= 0.5:
            continue
        for t in T:
            for w_ in Omega:
                lhs = sum(phat[(i, t, w_)] * x[i, s].X for i in I)
                total_minutes[s] += lhs
                rw = 0.0
                for i in I:
                    if x[i, s].X > 0.5:
                        rw += sum(q[(t, w_)][v] * p[(i, v)] * gamma[i] * r[(i, t, w_)] for v in q[(t, w_)])
                rework_minutes[s] += rw
        total_minutes[s] /= max(1, len(T) * len(Omega))
        rework_minutes[s] /= max(1, len(T) * len(Omega))
    rework_share_pct = {s: (100.0 * rework_minutes[s] / total_minutes[s]) if total_minutes[s] > 1e-12 else 0.0 for s in S}

    # ---- Load Variability (CV) per station across scenarios ----
    load_cv_by_s = {}
    for s in S:
        if y[s].X <= 0.5:
            load_cv_by_s[s] = np.nan
            continue
        samples = [sum(phat[(i, t, w_)] * x[i, s].X for i in I) for t in T for w_ in Omega]
        mean_s = avg(samples)
        sd_s = np.std(samples)
        load_cv_by_s[s] = (sd_s / (mean_s + 1e-12)) if mean_s > 1e-12 else 0.0

    return {
        "open_stations": open_stations,
        "w_s": {s: w[s].X for s in S}, "wbar": wbar_val, "z_s": {s: z[s].X for s in S},
        "smooth_abs_dev": smooth_abs_dev, "smooth_cv": smooth_cv,
        "Eh_by_s": Eh_by_s, "Exi_by_s": Exi_by_s, "Eh_total": Eh_total, "Exi_total": Exi_total,
        "service_by_t": service_by_t, "cvar_by_t": cvar_by_t,
        "p95_util_by_s": p95_util_by_s, "flagged_by_t": flagged_by_t,
        # New KPIs:
        "line_efficiency": line_efficiency, "balance_loss": balance_loss,
        "bottleneck_freq": bottleneck_freq,
        "rework_share_pct": rework_share_pct,
        "load_cv_by_s": load_cv_by_s
    }


# ---------------- Fixed Plotting Functions ----------------
def _save_or_show(fig, save_dir, fname, plot_cfg):
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        path = f"{save_dir.rstrip('/')}/{fname}.{plot_cfg['format']}"
        fig.savefig(path, dpi=plot_cfg['dpi'], bbox_inches='tight', format=plot_cfg['format'], transparent=True)
        print(f"Saved: {path}")
        plt.close(fig)  # Close after saving
    else:
        fig.set_dpi(plot_cfg['dpi'])
        plt.show()
        # Don't close when showing interactively


def plot_station_workloads(kpis, save_dir=None, plot_cfg=None):
    """Plot station workloads with error handling"""
    if plot_cfg is None:
        plot_cfg = {"dpi": 100, "format": "png"}
    
    s_open = kpis["open_stations"]
    if not s_open:
        print("Warning: No open stations found for workload plot")
        return
    
    try:
        vals = [kpis["w_s"][s] for s in s_open]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar([f"Station {s}" for s in s_open], vals)
        ax.axhline(kpis["wbar"], linestyle="--", alpha=0.7, label=f"Mean: {kpis['wbar']:.2f}")
        ax.set_title("Door Assembly Station Workloads")
        ax.set_xlabel("Station"); ax.set_ylabel("Minutes per door")
        ax.legend()
        plt.xticks(rotation=45)
        _save_or_show(fig, save_dir, "door_station_workloads", plot_cfg)
        print("✓ Station workloads plot completed")
    except Exception as e:
        print(f"Error in workload plot: {e}")


def plot_imbalance(kpis, save_dir=None, plot_cfg=None):
    """Plot workload imbalance with error handling"""
    if plot_cfg is None:
        plot_cfg = {"dpi": 100, "format": "png"}
    
    s_open = kpis["open_stations"]
    if not s_open:
        print("Warning: No open stations found for imbalance plot")
        return
    
    try:
        vals = [kpis["z_s"][s] for s in s_open]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar([f"Station {s}" for s in s_open], vals)
        ax.set_title("Door Assembly Workload Imbalance")
        ax.set_xlabel("Station"); ax.set_ylabel("Absolute deviation (min/door)")
        plt.xticks(rotation=45)
        _save_or_show(fig, save_dir, "door_imbalance", plot_cfg)
        print("✓ Imbalance plot completed")
    except Exception as e:
        print(f"Error in imbalance plot: {e}")


def plot_overtime_per_station(kpis, save_dir=None, plot_cfg=None):
    """Plot overtime per station with error handling"""
    if plot_cfg is None:
        plot_cfg = {"dpi": 100, "format": "png"}
    
    s_open = kpis["open_stations"]
    if not s_open:
        print("Warning: No open stations found for overtime plot")
        return
    
    try:
        vals = [kpis["Eh_by_s"][s] for s in s_open]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar([f"Station {s}" for s in s_open], vals)
        ax.set_title("Expected Overtime per Door Assembly Station")
        ax.set_xlabel("Station"); ax.set_ylabel("Minutes per door")
        plt.xticks(rotation=45)
        _save_or_show(fig, save_dir, "door_overtime_by_station", plot_cfg)
        print("✓ Overtime plot completed")
    except Exception as e:
        print(f"Error in overtime plot: {e}")


def plot_shortage_per_station(kpis, save_dir=None, plot_cfg=None):
    """Plot shortage per station with error handling"""
    if plot_cfg is None:
        plot_cfg = {"dpi": 100, "format": "png"}
    
    s_open = kpis["open_stations"]
    if not s_open:
        print("Warning: No open stations found for shortage plot")
        return
    
    try:
        vals = [kpis["Exi_by_s"][s] for s in s_open]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar([f"Station {s}" for s in s_open], vals)
        ax.set_title("Expected Shortage per Door Assembly Station")
        ax.set_xlabel("Station"); ax.set_ylabel("Minutes per door")
        plt.xticks(rotation=45)
        _save_or_show(fig, save_dir, "door_shortage_by_station", plot_cfg)
        print("✓ Shortage plot completed")
    except Exception as e:
        print(f"Error in shortage plot: {e}")


def plot_service_levels(kpis, alpha, save_dir=None, plot_cfg=None):
    """Plot service levels with error handling"""
    if plot_cfg is None:
        plot_cfg = {"dpi": 100, "format": "png"}
    
    try:
        tlist = sorted(kpis["service_by_t"].keys())
        vals = [kpis["service_by_t"][t] for t in tlist]
        target = 1.0 - alpha
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar([f"Shift {t}" for t in tlist], vals)
        ax.axhline(target, linestyle="--", alpha=0.7, label=f"Target: {target:.1%}")
        ax.set_ylim(0, 1.05)
        ax.set_title("Door Assembly Service Level by Shift")
        ax.set_xlabel("Shift"); ax.set_ylabel("Service level (fraction)")
        ax.legend()
        _save_or_show(fig, save_dir, "door_service_levels", plot_cfg)
        print("✓ Service levels plot completed")
    except Exception as e:
        print(f"Error in service levels plot: {e}")


def plot_utilization_p95(kpis, save_dir=None, plot_cfg=None):
    """Plot 95th percentile utilization with error handling"""
    if plot_cfg is None:
        plot_cfg = {"dpi": 100, "format": "png"}
    
    s_open = kpis["open_stations"]
    if not s_open:
        print("Warning: No open stations found for utilization plot")
        return
    
    try:
        vals = [kpis["p95_util_by_s"][s] for s in s_open]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar([f"Station {s}" for s in s_open], vals)
        ax.axhline(1.0, linestyle="--", alpha=0.7, label="100% Capacity")
        ax.set_title("Door Assembly Station Utilization (95th percentile)")
        ax.set_xlabel("Station"); ax.set_ylabel("Utilization ratio")
        ax.legend()
        plt.xticks(rotation=45)
        _save_or_show(fig, save_dir, "door_utilization_p95", plot_cfg)
        print("✓ Utilization plot completed")
    except Exception as e:
        print(f"Error in utilization plot: {e}")


def plot_line_efficiency(kpis, save_dir=None, plot_cfg=None):
    """Plot line efficiency metrics with error handling"""
    if plot_cfg is None:
        plot_cfg = {"dpi": 100, "format": "png"}
    
    try:
        eff = kpis["line_efficiency"]; loss = kpis["balance_loss"]
        
        # Handle NaN values
        if np.isnan(eff) or np.isnan(loss):
            print("Warning: Line efficiency metrics contain NaN values")
            return
            
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(["Line Efficiency", "Balance Loss"], [eff, loss])
        ax.set_ylim(0, 1.05)
        ax.set_title("Door Assembly Line Efficiency Metrics")
        ax.set_ylabel("Fraction")
        # Add percentage labels
        ax.text(0, eff + 0.02, f"{eff:.1%}", ha='center', va='bottom')
        ax.text(1, loss + 0.02, f"{loss:.1%}", ha='center', va='bottom')
        _save_or_show(fig, save_dir, "door_line_efficiency", plot_cfg)
        print("✓ Line efficiency plot completed")
    except Exception as e:
        print(f"Error in line efficiency plot: {e}")


def plot_bottleneck_frequency(kpis, save_dir=None, plot_cfg=None):
    """Plot bottleneck frequency with error handling"""
    if plot_cfg is None:
        plot_cfg = {"dpi": 100, "format": "png"}
    
    s_open = kpis["open_stations"]
    if not s_open:
        print("Warning: No open stations found for bottleneck plot")
        return
    
    try:
        vals = [kpis["bottleneck_freq"][s] for s in s_open]
        
        # Handle NaN values
        vals = [v if not np.isnan(v) else 0.0 for v in vals]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar([f"Station {s}" for s in s_open], vals)
        ax.set_ylim(0, max(vals) * 1.1 if vals and max(vals) > 0 else 1.05)
        ax.set_title("Door Assembly Bottleneck Frequency by Station")
        ax.set_xlabel("Station"); ax.set_ylabel("Bottleneck frequency")
        plt.xticks(rotation=45)
        _save_or_show(fig, save_dir, "door_bottleneck_freq", plot_cfg)
        print("✓ Bottleneck frequency plot completed")
    except Exception as e:
        print(f"Error in bottleneck frequency plot: {e}")


def plot_rework_share(kpis, save_dir=None, plot_cfg=None):
    """Plot rework share with error handling"""
    if plot_cfg is None:
        plot_cfg = {"dpi": 100, "format": "png"}
    
    s_open = kpis["open_stations"]
    if not s_open:
        print("Warning: No open stations found for rework plot")
        return
    
    try:
        vals = [kpis["rework_share_pct"][s] for s in s_open]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar([f"Station {s}" for s in s_open], vals)
        ax.set_title("Door Assembly Rework Load Share by Station")
        ax.set_xlabel("Station"); ax.set_ylabel("Rework share (%)")
        plt.xticks(rotation=45)
        _save_or_show(fig, save_dir, "door_rework_share", plot_cfg)
        print("✓ Rework share plot completed")
    except Exception as e:
        print(f"Error in rework share plot: {e}")


def plot_load_variability(kpis, save_dir=None, plot_cfg=None):
    """Plot load variability with error handling"""
    if plot_cfg is None:
        plot_cfg = {"dpi": 100, "format": "png"}
    
    s_open = kpis["open_stations"]
    if not s_open:
        print("Warning: No open stations found for load variability plot")
        return
    
    try:
        vals = [kpis["load_cv_by_s"][s] for s in s_open]
        
        # Handle NaN values
        vals = [v if not np.isnan(v) else 0.0 for v in vals]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar([f"Station {s}" for s in s_open], vals)
        ax.set_title("Door Assembly Load Variability by Station")
        ax.set_xlabel("Station"); ax.set_ylabel("Coefficient of Variation")
        plt.xticks(rotation=45)
        _save_or_show(fig, save_dir, "door_load_variability", plot_cfg)
        print("✓ Load variability plot completed")
    except Exception as e:
        print(f"Error in load variability plot: {e}")


# ---------------- Enhanced Reporting with Real Task Names ----------------
def report_solution(model, VAR, params, use_chance, alpha, use_robust, use_cvar, lam_cvar, kpis):
    I, S, T, Omega = params["I"], params["S"], params["T"], params["Omega"]
    task_data = params.get("task_data", {})
    x = VAR["x"]; y = VAR["y"]; mheads = VAR["m"]; w = VAR["w"]; wbar = VAR["wbar"]; z = VAR["z"]

    if model.status not in (GRB.OPTIMAL, GRB.INTERRUPTED, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        print(f"Model status: {model.status}")
        return

    print("\n" + "="*70)
    print("DOOR ASSEMBLY LINE BALANCING SOLUTION")
    print("="*70)
    
    print(f"\nObjective value: ${model.ObjVal:.2f}")

    print("\n=== STATION CONFIGURATION ===")
    total_workers = 0
    for s in S:
        if y[s].X > 0.5:
            workers = int(round(mheads[s].X))
            total_workers += workers
            print(f"Station {s}: {workers} workers, workload={w[s].X:.2f} min/door, imbalance={z[s].X:.2f}")
    print(f"Total workers: {total_workers}")
    print(f"Mean workload: {wbar.X:.2f} min/door")

    print("\n=== TASK ASSIGNMENTS ===")
    station_tasks = {s: [] for s in S}
    for i in I:
        for s in S:
            if x[i, s].X > 0.5:
                task_name = task_data.get(i, [f"Task {i}"])[0] if task_data else f"Task {i}"
                station_tasks[s].append((i, task_name))
    
    for s in S:
        if y[s].X > 0.5:
            print(f"\nStation {s}:")
            for i, task_name in sorted(station_tasks[s]):
                print(f"  • Task {i}: {task_name}")

    print("\n=== PERFORMANCE METRICS ===")
    print(f"Line efficiency: {kpis['line_efficiency']:.1%}")
    print(f"Balance loss: {kpis['balance_loss']:.1%}")
    print(f"Total smoothness penalty: {kpis['smooth_abs_dev']:.2f} min/door")
    print(f"Workload CV: {kpis['smooth_cv']:.3f}")
    
    print("\nExpected recourse per door:")
    print(f"  • Total overtime: {kpis['Eh_total']:.3f} min/door")
    print(f"  • Total shortage: {kpis['Exi_total']:.3f} min/door")

    print("\nService levels (target ≥ {:.1%}):".format(1.0 - alpha))
    for t in T:
        achieved = kpis['service_by_t'][t]
        status = "✓" if achieved >= (1.0 - alpha) else "✗"
        print(f"  • Shift {t}: {achieved:.1%} {status}")

    print("\n=== BOTTLENECK ANALYSIS ===")
    bottleneck_data = [(s, kpis["bottleneck_freq"][s]) for s in kpis["open_stations"]]
    bottleneck_data.sort(key=lambda x: x[1], reverse=True)
    
    print("Most frequently bottlenecked stations:")
    for s, freq in bottleneck_data[:3]:
        if freq > 0.1:  # Show stations that are bottlenecks >10% of time
            print(f"  • Station {s}: {freq:.1%} of scenarios")

    print("\n=== REWORK IMPACT ===")
    rework_data = [(s, kpis["rework_share_pct"][s]) for s in kpis["open_stations"]]
    rework_data.sort(key=lambda x: x[1], reverse=True)
    
    print("Highest rework load stations:")
    for s, pct in rework_data[:3]:
        if pct > 5.0:  # Show stations with >5% rework
            print(f"  • Station {s}: {pct:.1f}% rework load")


# ---------------- Fixed Main Function ----------------
def main():
    parser = argparse.ArgumentParser(description="Door Assembly Line Balancing with Real Manufacturing Data")
    parser.add_argument("--scenarios", type=int, default=20, help="Number of scenarios (|Ω|)")
    parser.add_argument("--stations", type=int, default=6, help="Number of stations (|S|)")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--chance", dest="chance", action="store_true", help="Enable chance constraints (default ON)")
    parser.add_argument("--no-chance", dest="chance", action="store_false", help="Disable chance constraints")
    parser.set_defaults(chance=True)
    parser.add_argument("--cvar", type=float, default=0.0, help="CVaR penalty lambda (>0 enables CVaR)")
    parser.add_argument("--robust", action="store_true", help="Enable distribution-free robust guard (3.18)")
    parser.add_argument("--silent", action="store_true", help="Suppress Gurobi solver output")
    parser.add_argument("--savefig", type=str, default="", help="Directory to save figures (omit to display)")
    parser.add_argument("--figdpi", type=int, default=600, help="Figure DPI for publication-quality images")
    parser.add_argument("--figformat", type=str, default="png", choices=["png","pdf","svg","tif"],
                        help="Figure file format")
    
    # Jupyter-safe parsing: ignore unknown args like "-f <kernel.json>"
    args, _ = parser.parse_known_args()

    # Publication-grade plot config
    plot_cfg = {"dpi": int(args.figdpi), "format": args.figformat}

    # Build real door assembly data
    print("Building door assembly line data...")
    data = create_door_assembly_data(n_stations=args.stations, n_scenarios=args.scenarios, seed=args.seed)

    # Configuration
    use_chance = bool(args.chance)
    use_cvar = bool(args.cvar > 0.0)
    lam_cvar = float(args.cvar)
    use_robust = bool(args.robust)

    print(f"Configuration: {args.stations} stations, {args.scenarios} scenarios")
    print(f"Chance constraints: {'ON' if use_chance else 'OFF'}")
    print(f"CVaR optimization: {'ON (λ={})'.format(lam_cvar) if use_cvar else 'OFF'}")
    print(f"Robust optimization: {'ON' if use_robust else 'OFF'}")

    # Build & solve model
    print("\nBuilding optimization model...")
    model, VAR = build_dalb_model(
        data["I"], data["S"], data["T"], data["Omega"], data["V"], data["P"],
        data["p"], data["tilde_q"], data["rbar"], data["gamma"],
        data["A"], data["d"], data["tau"], data["q"], data["r"], data["a"], data["delta"],
        data["M_s"], data["N_s_max"], data["E_i"], data["g_i"], data["E_s_max"], data["G_s"],
        data["c_st"], data["c_wk"], data["c_ot"], data["c_sh"], data["c_wb"],
        data["H_max"], data["alpha"], data["M"],
        use_chance=use_chance, use_cvar=use_cvar, lam_cvar=lam_cvar,
        use_robust=use_robust, Gamma_t=data["Gamma_t"], log_to_console=not args.silent
    )
    
    print("Solving optimization model...")
    model.optimize()

    # Check if optimization was successful
    if model.status not in (GRB.OPTIMAL, GRB.INTERRUPTED, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        print(f"Optimization failed with status: {model.status}")
        return

    # Compute KPIs and generate report
    print("Computing KPIs...")
    kpis = compute_kpis(model, VAR, data, use_chance, data["alpha"], use_cvar, lam_cvar)
    report_solution(model, VAR, data, use_chance, data["alpha"], use_robust, use_cvar, lam_cvar, kpis)

    # Generate visualizations with better error handling
    print(f"\nGenerating visualizations...")
    save_dir = args.savefig if args.savefig else None
    
    # Plot each visualization with individual error handling
    plots_to_generate = [
        ("Station Workloads", plot_station_workloads),
        ("Imbalance", plot_imbalance),
        ("Overtime per Station", plot_overtime_per_station),
        ("Shortage per Station", plot_shortage_per_station),
        ("Service Levels", lambda k, s, p: plot_service_levels(k, data["alpha"], s, p)),
        ("Utilization P95", plot_utilization_p95),
        ("Line Efficiency", plot_line_efficiency),
        ("Bottleneck Frequency", plot_bottleneck_frequency),
        ("Rework Share", plot_rework_share),
        ("Load Variability", plot_load_variability)
    ]
    
    successful_plots = 0
    for plot_name, plot_func in plots_to_generate:
        try:
            print(f"Generating {plot_name}...")
            plot_func(kpis, save_dir=save_dir, plot_cfg=plot_cfg)
            successful_plots += 1
        except Exception as e:
            print(f"Failed to generate {plot_name}: {e}")
    
    print(f"\nSuccessfully generated {successful_plots}/{len(plots_to_generate)} plots")
    if save_dir:
        print(f"All plots saved to: {save_dir}")
    
    print("\nDoor Assembly Line Balancing optimization complete!")


if __name__ == "__main__":
    main()