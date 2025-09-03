#!/usr/bin/env python3
"""
Python script to export DALB optimization results
Based on your console output from dalb_model.py
"""

import json
import os
from datetime import datetime

def create_solution_export():
    """
    Create solution file based on your DALB optimization results
    """
    
    # Results from your optimization run
    solution = {
        "metadata": {
            "export_time": datetime.now().isoformat(),
            "optimization_status": "OPTIMAL",
            "objective_value": 1799.9591,
            "optimization_time_seconds": 1.85,
            "gap_percent": 0.0000
        },
        
        "station_configuration": {
            "1": {
                "open": True,
                "workers": 3,
                "assigned_tasks": [2, 4],
                "workload_min_per_door": 4.454,
                "workload_deviation": 1.242,
                "status": "HEAVY_LOADED"
            },
            "2": {
                "open": True,
                "workers": 3,
                "assigned_tasks": [1, 3, 5],
                "workload_min_per_door": 4.241,
                "workload_deviation": 1.029,
                "status": "HEAVY_LOADED"
            },
            "3": {
                "open": True,
                "workers": 3,
                "assigned_tasks": [6, 7, 8],
                "workload_min_per_door": 4.105,
                "workload_deviation": 0.893,
                "status": "MEDIUM_LOADED"
            },
            "4": {
                "open": True,
                "workers": 2,
                "assigned_tasks": [9, 10],
                "workload_min_per_door": 3.247,
                "workload_deviation": 0.035,
                "status": "LIGHT_LOADED"
            },
            "5": {
                "open": False,
                "workers": 0,
                "assigned_tasks": [],
                "workload_min_per_door": 0.0,
                "workload_deviation": 0.0,
                "status": "CLOSED"
            },
            "6": {
                "open": True,
                "workers": 2,
                "assigned_tasks": [11, 12, 13],
                "workload_min_per_door": 3.224,
                "workload_deviation": 0.012,
                "status": "LIGHT_LOADED"
            }
        },
        
        "task_to_station_mapping": {
            1: 2,   # Task 1 -> Station 2
            2: 1,   # Task 2 -> Station 1
            3: 2,   # Task 3 -> Station 2
            4: 1,   # Task 4 -> Station 1
            5: 2,   # Task 5 -> Station 2
            6: 3,   # Task 6 -> Station 3
            7: 3,   # Task 7 -> Station 3
            8: 3,   # Task 8 -> Station 3
            9: 4,   # Task 9 -> Station 4
            10: 4,  # Task 10 -> Station 4
            11: 6,  # Task 11 -> Station 6
            12: 6,  # Task 12 -> Station 6
            13: 6   # Task 13 -> Station 6
        },
        
        "performance_metrics": {
            "total_workers": 13,
            "stations_open": 5,
            "stations_closed": 1,
            "mean_workload": 3.212,
            "workload_cv": 0.1611,
            "balance_loss": 0.1347,
            "line_efficiency": 0.8653,
            "expected_overtime_min_per_door": 6.9761,
            "expected_shortage_min_per_door": 0.0006,
            "service_level": {
                "period_1": 0.950,
                "period_2": 1.000,
                "target": 0.950
            },
            "bottleneck_analysis": {
                "primary_bottleneck": {
                    "station": 4,
                    "frequency": 0.575,
                    "tasks": [9, 10]
                },
                "secondary_bottleneck": {
                    "station": 6, 
                    "frequency": 0.425,
                    "tasks": [11, 12, 13]
                }
            },
            "rework_analysis": {
                "highest_rework_station": 6,
                "rework_percentage": 1.35,
                "stations_by_rework": {
                    "station_6": 1.35,
                    "station_1": 1.11,
                    "station_3": 0.85
                }
            }
        },
        
        "implementation_recommendations": {
            "immediate": [
                "Close Station 5 - not needed in optimal configuration",
                "Assign 3 workers each to Stations 1, 2, 3",
                "Assign 2 workers each to Stations 4, 6"
            ],
            "balance_improvements": [
                "Consider redistributing tasks from Stations 1-2 to Stations 4,6",
                "Station 1 and 2 are 38% above average workload"
            ],
            "bottleneck_mitigation": [
                "Station 4 (painting) is primary bottleneck - consider process improvements",
                "Station 6 (final assembly) is secondary bottleneck - monitor capacity"
            ],
            "quality_focus": [
                "Station 6 has highest rework rate (1.35%) - focus quality improvements",
                "Station 1 has second highest rework rate (1.11%)"
            ]
        }
    }
    
    return solution

def save_solution_files():
    """
    Save solution in multiple formats for different uses
    """
    
    solution = create_solution_export()
    
    # Create output directory
    os.makedirs("solution_export", exist_ok=True)
    
    # 1. Complete JSON for FlexSim integration
    with open("solution_export/dalb_solution_complete.json", 'w') as f:
        json.dump(solution, f, indent=2)
    
    # 2. Simple configuration for quick reference
    simple_config = {
        "stations": {
            1: {"open": True, "workers": 3, "tasks": [2, 4]},
            2: {"open": True, "workers": 3, "tasks": [1, 3, 5]},
            3: {"open": True, "workers": 3, "tasks": [6, 7, 8]},
            4: {"open": True, "workers": 2, "tasks": [9, 10]},
            5: {"open": False, "workers": 0, "tasks": []},
            6: {"open": True, "workers": 2, "tasks": [11, 12, 13]}
        },
        "summary": {
            "total_workers": 13,
            "objective_value": 1799.96,
            "line_efficiency": 0.8653
        }
    }
    
    with open("solution_export/simple_configuration.json", 'w') as f:
        json.dump(simple_config, f, indent=2)
    
    # 3. CSV for spreadsheet analysis
    import csv
    
    with open("solution_export/station_assignments.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Station', 'Open', 'Workers', 'Assigned_Tasks', 'Workload', 'Status'])
        
        for station_id, config in solution['station_configuration'].items():
            writer.writerow([
                station_id,
                config['open'],
                config['workers'],
                ','.join(map(str, config['assigned_tasks'])),
                config['workload_min_per_door'],
                config['status']
            ])
    
    # 4. Task assignment table
    with open("solution_export/task_assignments.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Task', 'Assigned_Station'])
        
        for task, station in solution['task_to_station_mapping'].items():
            writer.writerow([task, station])
    
    print("✅ Solution files created:")
    print("  📁 solution_export/")
    print("    📄 dalb_solution_complete.json    - Full solution for FlexSim")
    print("    📄 simple_configuration.json      - Quick reference")
    print("    📄 station_assignments.csv        - Station configuration")
    print("    📄 task_assignments.csv          - Task assignments")
    
    return solution

def print_summary():
    """
    Print a human-readable summary
    """
    solution = create_solution_export()
    
    print("\n" + "="*60)
    print("🏭 DALB OPTIMIZATION RESULTS SUMMARY")
    print("="*60)
    
    print(f"💰 Total Cost: ${solution['metadata']['objective_value']:.2f}")
    print(f"⏱️  Solve Time: {solution['metadata']['optimization_time_seconds']:.1f} seconds")
    print(f"✅ Status: {solution['metadata']['optimization_status']}")
    
    print(f"\n📊 PERFORMANCE METRICS:")
    metrics = solution['performance_metrics']
    print(f"  👥 Total Workers: {metrics['total_workers']}")
    print(f"  🏢 Stations Open: {metrics['stations_open']}/6")
    print(f"  ⚖️  Line Efficiency: {metrics['line_efficiency']:.1%}")
    print(f"  🎯 Service Level: {metrics['service_level']['period_1']:.1%}")
    print(f"  ⏰ Expected Overtime: {metrics['expected_overtime_min_per_door']:.2f} min/door")
    
    print(f"\n🏭 STATION CONFIGURATION:")
    for station_id, config in solution['station_configuration'].items():
        if config['open']:
            print(f"  Station {station_id}: {config['workers']} workers, "
                  f"Tasks {config['assigned_tasks']}, "
                  f"{config['workload_min_per_door']:.2f} min/door ({config['status']})")
        else:
            print(f"  Station {station_id}: CLOSED")
    
    print(f"\n⚠️  BOTTLENECK ANALYSIS:")
    bottleneck = solution['performance_metrics']['bottleneck_analysis']
    print(f"  🔴 Primary: Station {bottleneck['primary_bottleneck']['station']} "
          f"({bottleneck['primary_bottleneck']['frequency']:.1%} of time)")
    print(f"  🟡 Secondary: Station {bottleneck['secondary_bottleneck']['station']} "
          f"({bottleneck['secondary_bottleneck']['frequency']:.1%} of time)")
    
    print(f"\n🔧 KEY RECOMMENDATIONS:")
    for rec in solution['implementation_recommendations']['immediate']:
        print(f"  • {rec}")

def main():
    """
    Main execution function
    """
    print("🚀 Creating DALB solution export...")
    
    # Create all solution files
    solution = save_solution_files()
    
    # Print summary
    print_summary()
    
    print(f"\n📋 NEXT STEPS:")
    print("  1. Review the solution files in 'solution_export/' folder")
    print("  2. Use 'simple_configuration.json' for FlexSim implementation")
    print("  3. Import task assignments to your FlexSim model")
    print("  4. Test the configuration and compare performance")
    
    return solution

if __name__ == "__main__":
    solution = main()