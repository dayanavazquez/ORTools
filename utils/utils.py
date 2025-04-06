from distances.distance_type import DistanceType
from problems.problem_type import ProblemType
from load_data.instance_type import process_files, InstanceType
import numpy as np
from pathlib import Path

# Constants and configuration
PROBLEMS_DATA = [
    {
        "instance_type": InstanceType.TSP,
        "problem_type": ProblemType.TSP,
        "distance_type": DistanceType.CHEBYSHEV,
        "path": ['../../instances/tsp_instances']
    },
    {
        "instance_type": InstanceType.BHCVRP,
        "problem_type": ProblemType.CVRP,
        "distance_type": DistanceType.CHEBYSHEV,
        "path": ['../../instances/bhcvrp_instances']
    },
    {
        "instance_type": InstanceType.HFVRP,
        "problem_type": ProblemType.CVRP,
        "distance_type": DistanceType.CHEBYSHEV,
        "path": ['../../instances/hfvrp_instances']
    },
    {
        "instance_type": InstanceType.VRPTW,
        "problem_type": ProblemType.CVRP,
        "distance_type": DistanceType.CHEBYSHEV,
        "path": ['../../instances/vrptw_instances']
    },
    {
        "instance_type": InstanceType.VRPTW,
        "problem_type": ProblemType.VRPTW,
        "distance_type": DistanceType.CHEBYSHEV,
        "path": ['../../instances/vrptw_instances']
    },
    {
        "instance_type": InstanceType.VRPTW,
        "problem_type": ProblemType.VRPTW,
        "distance_type": DistanceType.CHEBYSHEV,
        "path": ['../../instances/vrptw_instances']
    },
    {
        "instance_type": InstanceType.MDCVRP,
        "problem_type": ProblemType.MDVRP,
        "distance_type": DistanceType.CHEBYSHEV,
        "path": ['../../instances/mdcvrp_instances/C-mdvrp']
    },
    {
        "instance_type": InstanceType.BHCVRP,
        "problem_type": ProblemType.VRPPD,
        "distance_type": DistanceType.CHEBYSHEV,
        "path": ['../../instances/bhcvrp_instances']
    },
    {
        "instance_type": InstanceType.MDCVRP,
        "problem_type": ProblemType.VRPPD,
        "distance_type": DistanceType.CHEBYSHEV,
        "path": ['../../instances/mdcvrp_instances/C-mdvrp']
    }
]

RESULT_TEMPLATES = {
    "TSP": {
        "Instance": [],
        "Nodes": [],
        "Objective": [],
        "Method": [],
        "Time": [],
        "Routes": [],
        "Average Distance Between Nodes": [],
        "Number of Jumps/Edges": []
    },
    "CVRP": {
        "Instance": [],
        "Vehicles": [],
        "Vehicles Capacity": [],
        "Demands": [],
        "Nodes": [],
        "Objective": [],
        "Method": [],
        "Time": [],
        "Routes": [],
        "Average Distance Between Nodes": [],
        "Number of Jumps/Edges": []
    },
    "VRPTW": {
        "Instance": [],
        "Vehicles": [],
        "Vehicles Capacity": [],
        "Avg Time Window Start": [],
        "Avg Time Window Length": [],
        "Max Time Window Length": [],
        "Min Time Window Length": [],
        "Time Window Coverage": [],
        "Demands": [],
        "Nodes": [],
        "Objective": [],
        "Method": [],
        "Time": [],
        "Routes": [],
        "Average Distance Between Nodes": [],
        "Number of Jumps/Edges": []
    },
    "MDVRP": {
        "Instance": [],
        "Vehicles": [],
        "Vehicles Capacity": [],
        "Starts Dispersion": [],
        "Ends Dispersion": [],
        "Demands": [],
        "Nodes": [],
        "Objective": [],
        "Method": [],
        "Time": [],
        "Routes": [],
        "Average Distance Between Nodes": [],
        "Number of Jumps/Edges": []
    },
    "VRPPD": {
        "Instance": [],
        "Vehicles": [],
        "Vehicles Capacity": [],
        "Num_Pickup_Delivery_Pairs": [],
        "Avg_Distance_Pickup_to_Delivery": [],
        "Max_Distance_Pickup_to_Delivery": [],
        "Demands": [],
        "Nodes": [],
        "Objective": [],
        "Method": [],
        "Time": [],
        "Routes": [],
        "Average Distance Between Nodes": [],
        "Number of Jumps/Edges": []
    }
}

_RESULTS_CACHE = None


def get_data_for_predictions(force_reload=False):
    global _RESULTS_CACHE

    if _RESULTS_CACHE is not None and not force_reload:
        return _RESULTS_CACHE

    results = {key: {k: [] for k in v} for key, v in RESULT_TEMPLATES.items()}

    for row in PROBLEMS_DATA:
        problem_type = row["problem_type"].value.upper()
        if problem_type not in results:
            continue

        instance_data = process_files(
            row["instance_type"],
            row["distance_type"],
            None, None, None,
            row["path"]
        )

        # Read solutions file
        solutions_path = Path(
            f"../../problems/solutions/{row['distance_type'].value}/"
            f"solutions_{row['problem_type'].value}/"
            f"all_solutions_{row['problem_type'].value}_{row['distance_type'].value}.txt"
        )

        try:
            with open(solutions_path, "r") as file:
                lines = file.readlines()
        except FileNotFoundError:
            continue

        header = lines[0].strip().split(";")
        records = [dict(zip(header, line.strip().split(";"))) for line in lines[1:]]

        for record in records:
            avg_distance = 0.0
            instance_name = record["Instance"]
            if instance_name not in instance_data:
                continue

            data = instance_data[instance_name]
            result = results[problem_type]

            result["Instance"].append(instance_name)
            result["Nodes"].append(
                int(data["num_locations"] if "num_locations" in data else len(data["distance_matrix"]))
            )
            result["Objective"].append(float(record["Objective"]))
            result["Method"].append(f"{record['Heuristic']}and{record['Metaheuristic']}")
            result["Time"].append(float(record["Time"]))
            result["Routes"].append(max(1, int(record["Routes"])))

            if problem_type != 'TSP':
                result["Vehicles"].append(int(data["num_vehicles"]))
                result["Demands"].append(np.mean(data["demands"]))
                result["Vehicles Capacity"].append(np.mean(data["vehicle_capacities"]))

            if problem_type == 'VRPTW':
                time_windows = data["time_windows"]
                starts, ends = zip(*time_windows)
                lengths = [e - s for s, e in time_windows]

                result["Avg Time Window Start"].append(np.mean(starts))
                result["Avg Time Window Length"].append(np.mean(lengths))
                result["Max Time Window Length"].append(np.max(lengths))
                result["Min Time Window Length"].append(np.min(lengths))
                result["Time Window Coverage"].append(
                    (max(ends) - min(starts)) / len(time_windows)
                )
            elif problem_type == 'MDVRP':
                starts = data["starts"]
                ends = data["ends"]
                dist_matrix = data["distance_matrix"]

                starts_dispersion = np.std([dist_matrix[i][j] for i in starts for j in starts if i != j])
                ends_dispersion = np.std([dist_matrix[i][j] for i in ends for j in ends if i != j])

                result["Starts Dispersion"].append(starts_dispersion)
                result["Ends Dispersion"].append(ends_dispersion)

            elif problem_type == 'VRPPD':
                pd_pairs = data['pickups_deliveries']
                dist_matrix = data['distance_matrix']
                pd_distances = [dist_matrix[p][d] for p, d in pd_pairs]

                result["Num_Pickup_Delivery_Pairs"].append(len(pd_pairs))
                result["Avg_Distance_Pickup_to_Delivery"].append(np.mean(pd_distances))
                result["Max_Distance_Pickup_to_Delivery"].append(np.max(pd_distances))

            elif problem_type == "TSP":
                locations = data.get("locations", [])
                n = len(locations)
                if n > 1:
                    distances = [
                        np.hypot(locations[i][0] - locations[j][0], locations[i][1] - locations[j][1])
                        for i in range(n) for j in range(i + 1, n)
                    ]
                avg_distance = np.mean(distances)
            result["Average Distance Between Nodes"].append(avg_distance)
            result["Number of Jumps/Edges"].append(n)
    _RESULTS_CACHE = results
    return results


def get_cached_results():
    if _RESULTS_CACHE:
        return _RESULTS_CACHE


def clear_cache():
    global _RESULTS_CACHE
    _RESULTS_CACHE = None
