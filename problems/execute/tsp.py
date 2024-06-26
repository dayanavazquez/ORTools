import os
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from load_data.instance_type import process_files, InstanceType
from distances.distance_type import calculate_distance, DistanceType


# [START distance_callback]
def compute_euclidean_distance_matrix(locations):
    """Creates callback to return distance between points."""
    distances = {}
    for from_counter, from_node in enumerate(locations):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(locations):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                # Euclidean distance
                distances[from_counter][to_counter] = int(
                    calculate_distance(from_node, to_node, DistanceType.EUCLIDEAN)
                )
    return distances


def save_solution(manager, routing, solution, instance, heuristic, metaheuristic):
    """Saves solution to a text file."""
    solutions_dir = os.path.join(f"problems/tsp/solutions_{heuristic}_{metaheuristic}")
    try:
        os.makedirs(solutions_dir, exist_ok=True)
        print(f"Directory {solutions_dir} created successfully or already exists.")
    except OSError as error:
        print(f"Error creating directory {solutions_dir}: {error}")
        return
    file_name = os.path.join(solutions_dir, f"solution_{instance}")
    try:
        with open(file_name, 'w') as f:
            f.write(f"Instance: {instance}\n\n")
            f.write(f"Heuristic: {heuristic}\n\n")
            f.write(f"Metaheuristic: {metaheuristic}\n\n")
            f.write(f"Objective: {solution.ObjectiveValue()}\n\n")
            index = routing.Start(0)
            plan_output = "Route:\n"
            route_distance = 0
            while not routing.IsEnd(index):
                plan_output += f" {manager.IndexToNode(index)} ->"
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
            plan_output += f" {manager.IndexToNode(index)}\n"
            f.write(plan_output)
            f.write(f"Objective: {route_distance}m\n")
        print(f"Solution saved successfully in {file_name}")
    except OSError as error:
        print(f"Error writing to file {file_name}: {error}")


def execute():
    """Entry point of the program."""
    # Instantiate the data problem.
    instances_data = process_files(InstanceType.TSP)
    for instance, data in instances_data.items():
        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(
            len(data["locations"]), data["num_vehicles"], data["depot"]
        )
        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)
        distance_matrix = compute_euclidean_distance_matrix(data["locations"])

        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        # Setting first solution heuristic.
        first_solution_strategies = [
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
            routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC,
            routing_enums_pb2.FirstSolutionStrategy.EVALUATOR_STRATEGY,
            routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
            routing_enums_pb2.FirstSolutionStrategy.SWEEP,
            routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES,
            routing_enums_pb2.FirstSolutionStrategy.ALL_UNPERFORMED,
            routing_enums_pb2.FirstSolutionStrategy.BEST_INSERTION,
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
            routing_enums_pb2.FirstSolutionStrategy.SEQUENTIAL_CHEAPEST_INSERTION,
            routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION,
            routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_COST_INSERTION,
            routing_enums_pb2.FirstSolutionStrategy.GLOBAL_CHEAPEST_ARC,
            routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_ARC,
            routing_enums_pb2.FirstSolutionStrategy.FIRST_UNBOUND_MIN_VALUE,
            routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
        ]

        local_search_metaheuristics = [
            routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT,
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
            routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
            routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
            routing_enums_pb2.LocalSearchMetaheuristic.GENERIC_TABU_SEARCH,
            routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC
        ]

        for first_solution_strategy in first_solution_strategies:
            for local_search_metaheuristic in local_search_metaheuristics:
                search_parameters = pywrapcp.DefaultRoutingSearchParameters()
                search_parameters.first_solution_strategy = first_solution_strategy
                search_parameters.local_search_metaheuristic = local_search_metaheuristic
                search_parameters.time_limit.FromSeconds(15)
                solution = routing.SolveWithParameters(search_parameters)

                # Print solution on console.
                if solution:
                    save_solution(manager, routing, solution, instance, search_parameters.first_solution_strategy, search_parameters.local_search_metaheuristic)
                else:
                    print("No solution found !")