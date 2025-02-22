from distances.distance_type import DistanceType
from problems.problem_type import ProblemType
from load_data.instance_type import process_files, InstanceType

problems_data = [
    {
        "instance_type": InstanceType.TSP,
        "problem_type": ProblemType.TSP,
        "distance_type": DistanceType.EUCLIDEAN,
        "path": ['../../instances/tsp_instances']
    },
    {
        "instance_type": InstanceType.BHCVRP,
        "problem_type": ProblemType.CVRP,
        "distance_type": DistanceType.EUCLIDEAN,
        "path": ['../../instances/bhcvrp_instances']
    },
    {
        "instance_type": InstanceType.HFVRP,
        "problem_type": ProblemType.CVRP,
        "distance_type": DistanceType.EUCLIDEAN,
        "path": ['../../instances/hfvrp_instances']
    },
    {
        "instance_type": InstanceType.VRPTW,
        "problem_type": ProblemType.CVRP,
        "distance_type": DistanceType.EUCLIDEAN,
        "path": ['../../instances/vrptw_instances']
    },
    {
        "instance_type": InstanceType.VRPTW,
        "problem_type": ProblemType.VRPTW,
        "distance_type": DistanceType.EUCLIDEAN,
        "path": ['../../instances/vrptw_instances']
    },
    {
        "instance_type": InstanceType.VRPTW,
        "problem_type": ProblemType.VRPTW,
        "distance_type": DistanceType.EUCLIDEAN,
        "path": ['../../instances/vrptw_instances']
    },
    {
        "instance_type": InstanceType.MDCVRP,
        "problem_type": ProblemType.MDVRP,
        "distance_type": DistanceType.EUCLIDEAN,
        "path": ['../../instances/mdcvrp_instances/C-mdvrp']
    },
    {
        "instance_type": InstanceType.BHCVRP,
        "problem_type": ProblemType.VRPPD,
        "distance_type": DistanceType.EUCLIDEAN,
        "path": ['../../instances/bhcvrp_instances']
    },
    {
        "instance_type": InstanceType.MDCVRP,
        "problem_type": ProblemType.VRPPD,
        "distance_type": DistanceType.EUCLIDEAN,
        "path": ['../../instances/mdcvrp_instances/C-mdvrp']
    }
]


def get_data_for_predictions():
    results = {
        "CVRP": {
            "Instance": [],
            "Vehicles": [],
            "Vehicles Capacity": [],
            "Demands": [],
            "Nodes": [],
            "Objective": [],
            "Method": [],
            "Time": [],
            "Routes": []
        },
        "VRPTW": {
            "Instance": [],
            "Vehicles": [],
            "Vehicles Capacity": [],
            "Demands": [],
            "Nodes": [],
            "Objective": [],
            "Method": [],
            "Time": [],
            "Routes": []
        },

    }

    for row in problems_data:
        problem_type = row["problem_type"].value.upper()
        if problem_type in results:
            instance_data = process_files(row["instance_type"], row["distance_type"], None,
                                          None, None, row["path"])
            with open(
                    f"../../problems/solutions/{row['distance_type'].value}/solutions_{row['problem_type'].value}/all_solutions_{row['problem_type'].value}_{row['distance_type'].value}.txt",
                    "r") as file:
                lines = file.readlines()
            header = lines[0].strip().split(";")
            records = [dict(zip(header, line.strip().split(";"))) for line in lines[1:]]
            for record in records:
                for instance, data in instance_data.items():
                    if record["Instance"] == instance:
                        results[problem_type]["Instance"].append(instance)
                        results[problem_type]["Vehicles"].append(int(data["num_vehicles"]))
                        if problem_type != 'TSP':
                            results[problem_type]["Demands"].append(
                                float(((sum(row for row in data["demands"])) / len(data["demands"])))
                            )
                            results[problem_type]["Vehicles Capacity"].append(
                                float(((sum(row for row in data["vehicle_capacities"])) / len(
                                    data["vehicle_capacities"])))
                            )
                        results[problem_type]["Nodes"].append(
                            int(data["num_locations"] if "num_locations" in data else len(data["distance_matrix"])))
                        results[problem_type]["Objective"].append(float(record["Objective"]))
                        results[problem_type]["Method"].append(f"{record['Heuristic']}and{record['Metaheuristic']}")
                        results[problem_type]["Time"].append(float(record["Time"]))
                        results[problem_type]["Routes"].append(
                            int(record["Routes"]) if int(record["Routes"]) > 0 else 1)
    return results