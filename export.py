import json

def export_dalb_solution():
    """
    Create a simple solution file based on your optimization results
    """
    
    solution = {
        "objective_value": 1799.96,
        "optimization_time": 1.85,
        "status": "OPTIMAL",
        
        "stations": {
            "1": {
                "open": True,
                "workers": 3,
                "assigned_tasks": [2, 4],
                "workload": 4.454,
                "workload_deviation": 1.242
            },
            "2": {
                "open": True, 
                "workers": 3,
                "assigned_tasks": [1, 3, 5],
                "workload": 4.241,
                "workload_deviation": 1.029
            },
            "3": {
                "open": True,
                "workers": 3, 
                "assigned_tasks": [6, 7, 8],
                "workload": 4.105,
                "workload_deviation": 0.893
            },
            "4": {
                "open": True,
                "workers": 2,
                "assigned_tasks": [9, 10], 
                "workload": 3.247,
                "workload_deviation": 0.035
            },
            "5": {
                "open": False,
                "workers": 0,
                "assigned_tasks": [],
                "workload": 0.0,
                "workload_deviation": 0.0
            },
            "6": {
                "open": True,
                "workers": 2,
                "assigned_tasks": [11, 12, 13],
                "workload": 3.224, 
                "workload_deviation": 0.012
            }
        },
        
        "task_assignments": {
            "1": 2, "2": 1, "3": 2, "4": 1, "5": 2,
            "6": 3, "7": 3, "8": 3, "9": 4, "10": 4,
            "11": 6, "12": 6, "13": 6
        },
        
        "performance_metrics": {
            "total_workers": 13,
            "open_stations": [1, 2, 3, 4, 6],
            "closed_stations": [5],
            "mean_workload": 3.212,
            "balance_loss": 0.1347,
            "line_efficiency": 0.8653,
            "expected_overtime": 6.9761,
            "expected_shortage": 0.0006,
            "service_level_period1": 0.950,
            "service_level_period2": 1.000,
            "primary_bottleneck": 4,
            "secondary_bottleneck": 6
        },
        
        "recommendations": {
            "balance_improvement": "Consider redistributing tasks from Stations 1-2 to Stations 4,6",
            "bottleneck_mitigation": "Station 4 (painting) is primary bottleneck - consider additional capacity",
            "efficiency_gain": "13.47% efficiency loss due to imbalance could be reduced",
            "rework_focus": "Station 6 has highest rework rate - focus quality improvement there"
        }
    }
    
    # Save to file
    with open('optimal_solution.json', 'w') as f:
        json.dump(solution, f, indent=2)
    
    print("✓ Solution exported to optimal_solution.json")
    print("\nQuick Summary:")
    print(f"- Total Cost: ${solution['objective_value']:.2f}")
    print(f"- Stations Open: {len(solution['performance_metrics']['open_stations'])}/6")
    print(f"- Total Workers: {solution['performance_metrics']['total_workers']}")
    print(f"- Line Efficiency: {solution['performance_metrics']['line_efficiency']:.1%}")
    print(f"- Service Level: {solution['performance_metrics']['service_level_period1']:.1%}")
    
    return solution

if __name__ == "__main__":
    solution = export_dalb_solution()