import pandas as pd
import numpy as np
import os
import argparse
from datetime import datetime
import json

# For standalone use when the main module isn't available
def create_door_assembly_data_standalone(n_stations=6, n_scenarios=20, seed=7):
    """
    Standalone version of door assembly data generation
    This is a simplified version for when the main module isn't available
    """
    np.random.seed(seed)
    
    # Basic parameters
    n_tasks = 18  # Typical number of tasks for door assembly
    n_variants = 3  # Standard, Premium, Electric
    n_periods = 2
    
    # Task data
    task_names = [
        "Mount Door Frame", "Install Hinges", "Mount Door Panel", "Install Lock Mechanism",
        "Install Window Glass", "Mount Window Regulator", "Install Door Handle", 
        "Connect Electrical Harness", "Install Weather Strips", "Mount Interior Trim",
        "Install Speaker", "Mount Armrest", "Install Door Light", "Mount Mirror",
        "Install Window Switch", "Quality Check", "Final Assembly", "Test Operations"
    ]
    
    I = list(range(1, n_tasks + 1))
    S = list(range(1, n_stations + 1))
    T = list(range(1, n_periods + 1))
    V = list(range(1, n_variants + 1))
    Omega = list(range(1, n_scenarios + 1))
    
    # Generate task data
    task_data = {}
    p = {}  # Processing times by task and variant
    
    for i in I:
        # Base processing time (2-8 minutes)
        base_time = np.random.uniform(2.0, 8.0)
        # Premium variant takes 10-30% longer
        prem_factor = np.random.uniform(1.1, 1.3)
        prem_time = base_time * prem_factor
        
        # Rework probability (1-8%)
        rework_prob = np.random.uniform(0.01, 0.08)
        # Rework overhead (20-50% additional time)
        rework_overhead = np.random.uniform(0.2, 0.5)
        
        task_name = task_names[i-1] if i <= len(task_names) else f"Task_{i}"
        task_data[i] = (task_name, base_time, prem_time, rework_prob, rework_overhead)
        
        # Processing times by variant
        for v in V:
            if v == 1:  # Standard
                p[(i, v)] = base_time
            elif v == 2:  # Premium
                p[(i, v)] = prem_time
            else:  # Electric - some tasks take longer
                electric_factor = 1.2 if i in [4, 8, 11, 13, 15] else 1.0
                p[(i, v)] = base_time * electric_factor
    
    # Precedence constraints (simplified)
    P = [(1, 2), (1, 3), (2, 4), (3, 5), (4, 7), (5, 6), (6, 9), (7, 10), 
         (8, 11), (9, 12), (10, 13), (11, 14), (12, 15), (13, 16), (14, 16), 
         (15, 17), (16, 17), (17, 18)]
    
    # Station parameters
    M_s = {s: 3 for s in S}  # Max 3 workers per station
    N_s_max = {s: 5 for s in S}  # Max 5 tasks per station
    E_s_max = {s: 15.0 for s in S}  # Ergonomic limits
    G_s = {s: 20.0 for s in S}  # Footprint limits
    
    # Task characteristics
    E_i = {i: np.random.uniform(1.0, 5.0) for i in I}  # Ergonomic scores
    g_i = {i: np.random.uniform(2.0, 8.0) for i in I}  # Footprint scores
    
    # Variant baseline shares
    tilde_q = {1: 0.6, 2: 0.3, 3: 0.1}  # Standard, Premium, Electric
    
    # Rework parameters
    rbar = {i: task_data[i][3] for i in I}  # Baseline rework probabilities
    gamma = {i: task_data[i][4] for i in I}  # Rework overhead factors
    
    # Available time per period (8 hours = 480 minutes)
    A = {t: 480.0 for t in T}
    
    # Generate scenario-dependent data
    d = {}  # Demand
    tau = {}  # Takt times
    q = {}  # Variant shares by scenario
    r = {}  # Rework probabilities by scenario
    a = {}  # Station availability
    delta = {}  # Setup losses
    
    for t in T:
        for w in Omega:
            # Demand varies ±30% around base demand
            base_demand = 100 if t == 1 else 120
            demand_factor = np.random.uniform(0.7, 1.3)
            d[(t, w)] = int(base_demand * demand_factor)
            
            # Takt time
            tau[(t, w)] = A[t] / d[(t, w)]
            
            # Variant mix varies around baseline
            q_vals = {}
            for v in V:
                noise = np.random.uniform(-0.1, 0.1)
                q_vals[v] = max(0.05, min(0.8, tilde_q[v] + noise))
            
            # Normalize to sum to 1
            total = sum(q_vals.values())
            q[(t, w)] = {v: q_vals[v] / total for v in V}
            
            # Rework probabilities vary ±50% around baseline
            for i in I:
                rework_factor = np.random.uniform(0.5, 1.5)
                r[(i, t, w)] = min(0.15, rbar[i] * rework_factor)
            
            # Station availability (90-99%)
            for s in S:
                a[(s, t, w)] = np.random.uniform(0.90, 0.99)
                # Setup losses (0.1-0.5 min per door)
                delta[(s, t, w)] = np.random.uniform(0.1, 0.5)
    
    # Cost parameters
    c_st = 1000  # Station opening cost
    c_wk = 500   # Worker cost
    c_ot = 10    # Overtime cost per minute per door
    c_sh = 50    # Shortage cost per minute per door
    c_wb = 5     # Workload balance cost
    
    # Other parameters
    H_max = 60   # Max overtime per station
    alpha = 0.9  # Service level
    M = 1000     # Big M
    
    return {
        "I": I, "S": S, "T": T, "V": V, "Omega": Omega, "P": P,
        "task_data": task_data, "p": p, "tilde_q": tilde_q,
        "rbar": rbar, "gamma": gamma, "A": A, "d": d, "tau": tau,
        "q": q, "r": r, "a": a, "delta": delta,
        "M_s": M_s, "N_s_max": N_s_max, "E_i": E_i, "g_i": g_i,
        "E_s_max": E_s_max, "G_s": G_s, "c_st": c_st, "c_wk": c_wk,
        "c_ot": c_ot, "c_sh": c_sh, "c_wb": c_wb, "H_max": H_max,
        "alpha": alpha, "M": M
    }


# Try to import from main module, fall back to standalone
try:
    from door_assembly_dalb import create_door_assembly_data, build_dalb_model, compute_kpis
    from gurobipy import GRB
    USE_MAIN_MODULE = True
except ImportError:
    print("Warning: Could not import main DALB module. Using standalone data generation.")
    print("Some features (model solving) will not be available.")
    create_door_assembly_data = create_door_assembly_data_standalone
    USE_MAIN_MODULE = False


class DoorAssemblyDataExporter:
    """
    Export all door assembly line balancing data to CSV files for analysis
    """
    
    def __init__(self, output_dir="door_assembly_data", n_stations=6, n_scenarios=20, seed=7):
        self.output_dir = output_dir
        self.n_stations = n_stations
        self.n_scenarios = n_scenarios
        self.seed = seed
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate data
        print(f"Generating door assembly data: {n_stations} stations, {n_scenarios} scenarios...")
        self.data = create_door_assembly_data(n_stations, n_scenarios, seed)
        
        if self.data is None:
            raise ValueError("Data generation failed - check your data generation function")
        
        # Metadata
        self.metadata = {
            "export_timestamp": datetime.now().isoformat(),
            "n_stations": n_stations,
            "n_scenarios": n_scenarios,
            "seed": seed,
            "description": "Door Assembly Line Balancing Data Export",
            "data_source": "main_module" if USE_MAIN_MODULE else "standalone"
        }
    
    def export_tasks(self):
        """Export task data with descriptions, times, and rework parameters"""
        print("Exporting task data...")
        
        task_data = self.data["task_data"]
        
        rows = []
        for i in self.data["I"]:
            task_name, base_time, prem_time, rework_prob, rework_overhead = task_data[i]
            
            # Calculate derived baseline time (D2 from document)
            tilde_p_i = sum(self.data["tilde_q"][v] * self.data["p"][(i, v)] * 
                           (1.0 + self.data["gamma"][i] * self.data["rbar"][i]) 
                           for v in self.data["V"])
            
            rows.append({
                "task_id": i,
                "task_name": task_name,
                "base_time_min_per_door": base_time,
                "premium_time_min_per_door": prem_time,
                "baseline_rework_prob": rework_prob,
                "rework_overhead_factor": rework_overhead,
                "ergonomic_score": self.data["E_i"][i],
                "footprint_score": self.data["g_i"][i],
                "baseline_effective_time": tilde_p_i
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(f"{self.output_dir}/tasks.csv", index=False)
        return df
    
    def export_variants(self):
        """Export variant data and processing times"""
        print("Exporting variant data...")
        
        rows = []
        for v in self.data["V"]:
            for i in self.data["I"]:
                task_name = self.data["task_data"][i][0]
                rows.append({
                    "variant": v,
                    "task_id": i,
                    "task_name": task_name,
                    "processing_time_min_per_door": self.data["p"][(i, v)],
                    "baseline_share": self.data["tilde_q"][v]
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(f"{self.output_dir}/variant_times.csv", index=False)
        
        # Also export variant baseline shares separately
        variant_shares = pd.DataFrame([
            {"variant": v, "baseline_share": self.data["tilde_q"][v]} 
            for v in self.data["V"]
        ])
        variant_shares.to_csv(f"{self.output_dir}/variant_baseline_shares.csv", index=False)
        
        return df
    
    def export_precedence(self):
        """Export precedence constraints"""
        print("Exporting precedence constraints...")
        
        rows = []
        for (i, j) in self.data["P"]:
            task_i_name = self.data["task_data"][i][0]
            task_j_name = self.data["task_data"][j][0]
            
            rows.append({
                "predecessor_id": i,
                "predecessor_name": task_i_name,
                "successor_id": j,
                "successor_name": task_j_name,
                "precedence_type": "must_precede"
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(f"{self.output_dir}/precedence_constraints.csv", index=False)
        return df
    
    def export_stations(self):
        """Export station configuration parameters"""
        print("Exporting station parameters...")
        
        rows = []
        for s in self.data["S"]:
            rows.append({
                "station_id": s,
                "max_workers": self.data["M_s"][s],
                "max_tasks": self.data["N_s_max"][s],
                "ergonomic_limit": self.data["E_s_max"][s],
                "footprint_limit": self.data["G_s"][s]
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(f"{self.output_dir}/stations.csv", index=False)
        return df
    
    def export_scenarios(self):
        """Export all scenario-dependent data"""
        print("Exporting scenario data...")
        
        # Demand and takt times
        demand_rows = []
        for t in self.data["T"]:
            for w in self.data["Omega"]:
                demand_rows.append({
                    "period": t,
                    "scenario": w,
                    "demand_doors": self.data["d"][(t, w)],
                    "available_time_min": self.data["A"][t],
                    "takt_time_min_per_door": self.data["tau"][(t, w)]
                })
        
        demand_df = pd.DataFrame(demand_rows)
        demand_df.to_csv(f"{self.output_dir}/scenario_demand.csv", index=False)
        
        # Variant mix by scenario
        variant_mix_rows = []
        for t in self.data["T"]:
            for w in self.data["Omega"]:
                for v in self.data["V"]:
                    variant_mix_rows.append({
                        "period": t,
                        "scenario": w,
                        "variant": v,
                        "realized_share": self.data["q"][(t, w)][v]
                    })
        
        variant_mix_df = pd.DataFrame(variant_mix_rows)
        variant_mix_df.to_csv(f"{self.output_dir}/scenario_variant_mix.csv", index=False)
        
        # Rework probabilities by scenario
        rework_rows = []
        for i in self.data["I"]:
            task_name = self.data["task_data"][i][0]
            for t in self.data["T"]:
                for w in self.data["Omega"]:
                    rework_rows.append({
                        "task_id": i,
                        "task_name": task_name,
                        "period": t,
                        "scenario": w,
                        "rework_probability": self.data["r"][(i, t, w)],
                        "baseline_rework_prob": self.data["rbar"][i],
                        "rework_overhead_factor": self.data["gamma"][i]
                    })
        
        rework_df = pd.DataFrame(rework_rows)
        rework_df.to_csv(f"{self.output_dir}/scenario_rework.csv", index=False)
        
        return demand_df, variant_mix_df, rework_df
    
    def export_station_scenarios(self):
        """Export station-specific scenario data (availability, setup losses)"""
        print("Exporting station scenario data...")
        
        rows = []
        for s in self.data["S"]:
            for t in self.data["T"]:
                for w in self.data["Omega"]:
                    rows.append({
                        "station_id": s,
                        "period": t,
                        "scenario": w,
                        "availability_fraction": self.data["a"][(s, t, w)],
                        "setup_loss_min_per_door": self.data["delta"][(s, t, w)]
                    })
        
        df = pd.DataFrame(rows)
        df.to_csv(f"{self.output_dir}/station_scenarios.csv", index=False)
        return df
    
    def export_scenario_times(self):
        """Export effective processing times by scenario (D1 from document)"""
        print("Exporting scenario-effective processing times...")
        
        rows = []
        for i in self.data["I"]:
            task_name = self.data["task_data"][i][0]
            for t in self.data["T"]:
                for w in self.data["Omega"]:
                    # Calculate phat (D1) - scenario effective time
                    phat_val = sum(self.data["q"][(t, w)][v] * self.data["p"][(i, v)] * 
                                  (1.0 + self.data["gamma"][i] * self.data["r"][(i, t, w)]) 
                                  for v in self.data["V"])
                    
                    rows.append({
                        "task_id": i,
                        "task_name": task_name,
                        "period": t,
                        "scenario": w,
                        "effective_time_min_per_door": phat_val
                    })
        
        df = pd.DataFrame(rows)
        df.to_csv(f"{self.output_dir}/scenario_effective_times.csv", index=False)
        return df
    
    def export_costs(self):
        """Export cost parameters"""
        print("Exporting cost parameters...")
        
        costs = [
            {"cost_type": "station_opening", "cost_value": self.data["c_st"], "units": "currency_per_station"},
            {"cost_type": "worker", "cost_value": self.data["c_wk"], "units": "currency_per_worker"},
            {"cost_type": "overtime", "cost_value": self.data["c_ot"], "units": "currency_per_min_per_door"},
            {"cost_type": "shortage", "cost_value": self.data["c_sh"], "units": "currency_per_min_per_door"},
            {"cost_type": "workload_balance", "cost_value": self.data["c_wb"], "units": "currency_per_min_per_door"},
        ]
        
        operational = [
            {"parameter": "overtime_cap_per_station", "value": self.data["H_max"], "units": "min_per_door"},
            {"parameter": "service_level_alpha", "value": self.data["alpha"], "units": "fraction"},
            {"parameter": "big_M", "value": self.data["M"], "units": "min_per_door"},
        ]
        
        cost_df = pd.DataFrame(costs)
        cost_df.to_csv(f"{self.output_dir}/cost_parameters.csv", index=False)
        
        ops_df = pd.DataFrame(operational)
        ops_df.to_csv(f"{self.output_dir}/operational_parameters.csv", index=False)
        
        return cost_df, ops_df
    
    def export_summary_statistics(self):
        """Export summary statistics of the generated data"""
        print("Exporting summary statistics...")
        
        stats = []
        
        # Task statistics
        base_times = [self.data["task_data"][i][1] for i in self.data["I"]]
        prem_times = [self.data["task_data"][i][2] for i in self.data["I"]]
        rework_probs = [self.data["task_data"][i][3] for i in self.data["I"]]
        
        stats.extend([
            {"category": "tasks", "statistic": "count", "value": len(self.data["I"])},
            {"category": "tasks", "statistic": "base_time_mean", "value": np.mean(base_times)},
            {"category": "tasks", "statistic": "base_time_std", "value": np.std(base_times)},
            {"category": "tasks", "statistic": "premium_time_mean", "value": np.mean(prem_times)},
            {"category": "tasks", "statistic": "rework_prob_mean", "value": np.mean(rework_probs)},
        ])
        
        # Demand statistics
        demands = [self.data["d"][(t, w)] for t in self.data["T"] for w in self.data["Omega"]]
        takts = [self.data["tau"][(t, w)] for t in self.data["T"] for w in self.data["Omega"]]
        
        stats.extend([
            {"category": "demand", "statistic": "scenarios", "value": len(self.data["Omega"])},
            {"category": "demand", "statistic": "periods", "value": len(self.data["T"])},
            {"category": "demand", "statistic": "demand_mean", "value": np.mean(demands)},
            {"category": "demand", "statistic": "demand_std", "value": np.std(demands)},
            {"category": "demand", "statistic": "takt_mean", "value": np.mean(takts)},
            {"category": "demand", "statistic": "takt_std", "value": np.std(takts)},
        ])
        
        # Availability statistics
        avails = [self.data["a"][(s, t, w)] for s in self.data["S"] 
                 for t in self.data["T"] for w in self.data["Omega"]]
        
        stats.extend([
            {"category": "availability", "statistic": "mean", "value": np.mean(avails)},
            {"category": "availability", "statistic": "std", "value": np.std(avails)},
            {"category": "availability", "statistic": "min", "value": np.min(avails)},
            {"category": "availability", "statistic": "max", "value": np.max(avails)},
        ])
        
        stats_df = pd.DataFrame(stats)
        stats_df.to_csv(f"{self.output_dir}/summary_statistics.csv", index=False)
        return stats_df
    
    def export_metadata(self):
        """Export metadata about the data generation"""
        print("Exporting metadata...")
        
        # Save metadata as JSON
        with open(f"{self.output_dir}/metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)
        
        # Also save as CSV for easier reading
        meta_rows = []
        for key, value in self.metadata.items():
            meta_rows.append({"parameter": key, "value": str(value)})
        
        meta_df = pd.DataFrame(meta_rows)
        meta_df.to_csv(f"{self.output_dir}/metadata.csv", index=False)
        return meta_df
    
    def export_all(self):
        """Export all data to CSV files"""
        print(f"\nExporting door assembly data to: {self.output_dir}")
        print("="*60)
        
        # Export all data categories
        self.export_tasks()
        self.export_variants()
        self.export_precedence()
        self.export_stations()
        self.export_scenarios()
        self.export_station_scenarios()
        self.export_scenario_times()
        self.export_costs()
        self.export_summary_statistics()
        self.export_metadata()
        
        print("\n" + "="*60)
        print("DATA EXPORT COMPLETE")
        print("="*60)
        
        # List all created files
        files = [f for f in os.listdir(self.output_dir) if f.endswith(('.csv', '.json'))]
        files.sort()
        
        print(f"\nCreated {len(files)} files:")
        for f in files:
            file_path = os.path.join(self.output_dir, f)
            size_kb = os.path.getsize(file_path) / 1024
            print(f"  • {f:<35} ({size_kb:.1f} KB)")
        
        return files


def export_solution_data(model, VAR, params, kpis, output_dir="door_assembly_solution"):
    """
    Export optimization solution data to CSV files
    """
    if not USE_MAIN_MODULE:
        print("Solution export not available without main DALB module")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nExporting solution data to: {output_dir}")
    
    I, S, T, Omega = params["I"], params["S"], params["T"], params["Omega"]
    x, y, mheads = VAR["x"], VAR["y"], VAR["m"]
    h, xi = VAR["h"], VAR["xi"]
    
    if model.status not in (GRB.OPTIMAL, GRB.INTERRUPTED, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        print("Warning: Model did not solve to optimality")
        return
    
    # Task assignments
    assignment_rows = []
    for i in I:
        for s in S:
            if x[i, s].X > 0.5:
                task_name = params["task_data"][i][0]
                assignment_rows.append({
                    "task_id": i,
                    "task_name": task_name,
                    "station_id": s,
                    "assignment_value": x[i, s].X
                })
    
    assignment_df = pd.DataFrame(assignment_rows)
    assignment_df.to_csv(f"{output_dir}/task_assignments.csv", index=False)
    
    # Station configuration
    station_rows = []
    for s in S:
        station_rows.append({
            "station_id": s,
            "is_open": int(y[s].X > 0.5),
            "workers": int(round(mheads[s].X)),
            "workload_min_per_door": kpis["w_s"][s],
            "workload_deviation": kpis["z_s"][s],
            "expected_overtime": kpis["Eh_by_s"][s],
            "expected_shortage": kpis["Exi_by_s"][s],
            "p95_utilization": kpis["p95_util_by_s"].get(s, 0),
            "bottleneck_frequency": kpis["bottleneck_freq"][s],
            "rework_share_pct": kpis["rework_share_pct"][s],
            "load_cv": kpis["load_cv_by_s"].get(s, 0)
        })
    
    station_df = pd.DataFrame(station_rows)
    station_df.to_csv(f"{output_dir}/station_solution.csv", index=False)
    
    # Scenario recourse (overtime & shortage)
    recourse_rows = []
    for s in S:
        if y[s].X > 0.5:  # Only open stations
            for t in T:
                for w in Omega:
                    recourse_rows.append({
                        "station_id": s,
                        "period": t,
                        "scenario": w,
                        "overtime": h[s, t, w].X,
                        "shortage": xi[s, t, w].X
                    })
    
    recourse_df = pd.DataFrame(recourse_rows)
    recourse_df.to_csv(f"{output_dir}/scenario_recourse.csv", index=False)
    
    # Solution summary
    summary = [
        {"metric": "objective_value", "value": model.ObjVal},
        {"metric": "stations_opened", "value": len(kpis["open_stations"])},
        {"metric": "total_workers", "value": sum(int(round(mheads[s].X)) for s in S)},
        {"metric": "line_efficiency", "value": kpis["line_efficiency"]},
        {"metric": "balance_loss", "value": kpis["balance_loss"]},
        {"metric": "total_overtime", "value": kpis["Eh_total"]},
        {"metric": "total_shortage", "value": kpis["Exi_total"]},
        {"metric": "service_level_period_1", "value": kpis["service_by_t"][1]},
        {"metric": "service_level_period_2", "value": kpis["service_by_t"][2]},
        {"metric": "solve_time_seconds", "value": model.Runtime}
    ]
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f"{output_dir}/solution_summary.csv", index=False)
    
    print("Solution data export complete!")


def main():
    parser = argparse.ArgumentParser(description="Export Door Assembly Line Balancing Data to CSV")
    parser.add_argument("--stations", type=int, default=6, help="Number of stations")
    parser.add_argument("--scenarios", type=int, default=20, help="Number of scenarios") 
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--output", type=str, default="door_assembly_data", 
                       help="Output directory for CSV files")
    parser.add_argument("--solve", action="store_true", 
                       help="Also solve the model and export solution data")
    
    args = parser.parse_args()
    
    try:
        # Export input data
        exporter = DoorAssemblyDataExporter(
            output_dir=args.output,
            n_stations=args.stations,
            n_scenarios=args.scenarios,
            seed=args.seed
        )
        
        files = exporter.export_all()
        
        # Optionally solve and export solution
        if args.solve and USE_MAIN_MODULE:
            try:
                print("\nSolving optimization model...")
                model, VAR = build_dalb_model(
                    exporter.data["I"], exporter.data["S"], exporter.data["T"], 
                    exporter.data["Omega"], exporter.data["V"], exporter.data["P"],
                    exporter.data["p"], exporter.data["tilde_q"], exporter.data["rbar"], 
                    exporter.data["gamma"], exporter.data["A"], exporter.data["d"], 
                    exporter.data["tau"], exporter.data["q"], exporter.data["r"], 
                    exporter.data["a"], exporter.data["delta"], exporter.data["M_s"], 
                    exporter.data["N_s_max"], exporter.data["E_i"], exporter.data["g_i"], 
                    exporter.data["E_s_max"], exporter.data["G_s"], exporter.data["c_st"], 
                    exporter.data["c_wk"], exporter.data["c_ot"], exporter.data["c_sh"], 
                    exporter.data["c_wb"], exporter.data["H_max"], exporter.data["alpha"], 
                    exporter.data["M"], log_to_console=False
                )
                
                model.optimize()
                kpis = compute_kpis(model, VAR, exporter.data, True, exporter.data["alpha"], False, 0.0)
                
                export_solution_data(model, VAR, exporter.data, kpis, 
                                   output_dir=f"{args.output}_solution")
                                   
            except Exception as e:
                print(f"Error solving model: {e}")
        elif args.solve and not USE_MAIN_MODULE:
            print("Cannot solve model: main DALB module not available")
        
        print(f"\nAll data exported successfully to: {args.output}")
        
    except Exception as e:
        print(f"Error during data export: {e}")
        return 1
    
    return 0
    

if __name__ == "__main__":
    exit(main())