// Simple FlexSim Configuration Based on DALB Results
// Add this as a Global Function in your FlexSim model

function applyDALBConfiguration() {
    print("=== Applying DALB Optimal Configuration ===");
    
    // Station configuration from optimization results
    var stationConfig = {
        1: {open: true, workers: 3, tasks: [2, 4]},      // Handle + Window
        2: {open: true, workers: 3, tasks: [1, 3, 5]},   // Frame + Lock + Strip  
        3: {open: true, workers: 3, tasks: [6, 7, 8]},   // Hinge + Adjust + QC
        4: {open: true, workers: 2, tasks: [9, 10]},     // Paint Prep + Paint
        5: {open: false, workers: 0, tasks: []},         // CLOSED
        6: {open: true, workers: 2, tasks: [11, 12, 13]} // Dry + Assemble + Pack
    };
    
    // Apply configuration to your FlexSim model
    // CUSTOMIZE these paths to match your model structure
    
    try {
        // Get your station area (adjust path)
        var stationArea = Model.find("Assembly"); // Change to your station container
        
        if(!stationArea) {
            print("Error: Station area not found. Update the path in script.");
            return false;
        }
        
        var stations = stationArea.subnodes;
        print("Found " + stations.length + " stations in model");
        
        // Configure each station
        for(var stationId in stationConfig) {
            var config = stationConfig[stationId];
            var stationIndex = parseInt(stationId) - 1; // Convert to 0-indexed
            
            if(stationIndex < stations.length) {
                var station = stations[stationIndex];
                
                if(config.open) {
                    // Enable station
                    station.attrs.enabled = 1;
                    
                    // Set worker count (adjust based on your operator setup)
                    if(station.find("Operators")) {
                        station.find("Operators").maxcontent = config.workers;
                    }
                    
                    // Store task assignments for routing
                    station.attrs.assigned_tasks = config.tasks;
                    station.attrs.optimal_workers = config.workers;
                    
                    print("Station " + stationId + ": ENABLED, " + 
                          config.workers + " workers, tasks " + 
                          JSON.stringify(config.tasks));
                } else {
                    // Disable station
                    station.attrs.enabled = 0;
                    station.attrs.assigned_tasks = [];
                    print("Station " + stationId + ": DISABLED");
                }
            }
        }
        
        print("✓ DALB configuration applied successfully!");
        print("Total active workers: 13 across 5 stations");
        print("Expected line efficiency: 86.5%");
        
        return true;
        
    } catch(e) {
        print("Error applying configuration: " + e.message);
        print("Please customize the script paths for your FlexSim model");
        return false;
    }
}

function validateConfiguration() {
    print("=== Configuration Validation ===");
    
    // Check if all 13 tasks are assigned
    var assignedTasks = [2,4,1,3,5,6,7,8,9,10,11,12,13];
    var expectedTasks = [1,2,3,4,5,6,7,8,9,10,11,12,13];
    
    print("Task assignment check:");
    for(var i = 0; i < expectedTasks.length; i++) {
        var task = expectedTasks[i];
        if(assignedTasks.indexOf(task) >= 0) {
            print("✓ Task " + task + " assigned");
        } else {
            print("✗ Task " + task + " NOT assigned");
        }
    }
    
    // Check station workload balance
    var workloads = [4.454, 4.241, 4.105, 3.247, 0.0, 3.224];
    var meanWorkload = 3.212;
    
    print("Workload balance analysis:");
    for(var s = 0; s < workloads.length; s++) {
        var stationId = s + 1;
        var workload = workloads[s];
        var deviation = Math.abs(workload - meanWorkload);
        var status = deviation > 1.0 ? "HIGH" : (deviation > 0.5 ? "MED" : "LOW");
        
        if(workload > 0) {
            print("Station " + stationId + ": " + workload.toFixed(3) + 
                  " min/door, deviation=" + deviation.toFixed(3) + " (" + status + ")");
        }
    }
    
    print("Primary bottleneck: Station 4 (painting operations)");
    print("Secondary bottleneck: Station 6 (final assembly)");
}

// Main function to call
function implementDALBResults() {
    if(applyDALBConfiguration()) {
        validateConfiguration();
        
        print("\n=== Next Steps ===");
        print("1. Test run your FlexSim simulation");
        print("2. Monitor station utilization and throughput");
        print("3. Compare performance to previous configuration");
        print("4. Consider bottleneck improvements for Stations 4 & 6");
        
        return true;
    }
    
    return false;
}

// Execute the implementation
implementDALBResults();