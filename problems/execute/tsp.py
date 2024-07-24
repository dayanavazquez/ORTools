import os
import time
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from load_data.instance_type import process_files
from distances.distance_type import calculate_distance, DistanceType
from problems.strategy_type import HeuristicType, MetaheuristicType
from utils.get_strategies import get_strategies


# [START distance_callback]
def compute_euclidean_distance_matrix(locations, distance_type):
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
                    calculate_distance(from_node, to_node, distance_type)
                )
    return distances


def save_solution(manager, routing, solution, instance, heuristic, metaheuristic, elapsed_time, i):
    """Saves solution to a text file."""
    if heuristic and not metaheuristic:
        solutions_dir = os.path.join(f"problems/tsp/solutions_tsp_{i}/solutions_{heuristic}")
    elif metaheuristic and not heuristic:
        solutions_dir = os.path.join(f"problems/tsp/solutions_tsp_{i}/solutions_{metaheuristic}")
    else:
        solutions_dir = os.path.join(f"problems/tsp/solutions_tsp_{i}/solutions_{heuristic}_&_{metaheuristic}")
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
            f.write(f"Objective: {solution.ObjectiveValue()}\n\n")
            f.write(f"Execution Time: {elapsed_time}\n\n")
            f.write(f"Heuristic: {heuristic}\n\n")
            f.write(f"Metaheuristic: {metaheuristic}\n\n")
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


def execute(i, instance_type, time_limit, distance_type: DistanceType = None, heuristic: HeuristicType = None,
            metaheuristic: MetaheuristicType = None):
    """Entry point of the program."""
    # Instantiate the data problem.
    instances_data = process_files(instance_type)
    for instance, data in instances_data.items():
        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(
            len(data["locations"]), data["num_vehicles"], data["depot"]
        )
        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)
        distance_matrix = compute_euclidean_distance_matrix(data["locations"], distance_type)

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

        first_solution_strategies, local_search_metaheuristics = get_strategies(heuristic, metaheuristic)
        if not first_solution_strategies and local_search_metaheuristics:
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.local_search_metaheuristic = getattr(
                routing_enums_pb2.LocalSearchMetaheuristic, local_search_metaheuristics[0]
            )
            get_solutions(i, search_parameters, routing, time_limit, data, manager, instance,
                          local_search_metaheuristics[0])
        elif not local_search_metaheuristics and first_solution_strategies:
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = getattr(
                routing_enums_pb2.FirstSolutionStrategy, first_solution_strategies[0]
            )
            get_solutions(i, search_parameters, routing, time_limit, data, manager, instance,
                          first_solution_strategies[0])
        else:
            for first_solution_strategy in first_solution_strategies:
                for local_search_metaheuristic in local_search_metaheuristics:
                    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
                    search_parameters.first_solution_strategy = getattr(
                        routing_enums_pb2.FirstSolutionStrategy, first_solution_strategy
                    )
                    search_parameters.local_search_metaheuristic = getattr(
                        routing_enums_pb2.LocalSearchMetaheuristic, local_search_metaheuristic
                    )
                    get_solutions(i, search_parameters, routing, time_limit, data, manager, instance,
                                  first_solution_strategy, local_search_metaheuristic)


def get_solutions(i, search_parameters, routing, time_limit, data, manager, instance, first_solution_strategy=None,
                  local_search_metaheuristic=None):
    search_parameters.time_limit.FromSeconds(time_limit)
    start_time = time.time()
    solution = routing.SolveWithParameters(search_parameters)
    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time  # Calculate elapsed time

    # Save solution on console.
    if solution:
        save_solution(data, manager, routing, solution, instance, first_solution_strategy,
                      local_search_metaheuristic,
                      elapsed_time, i)
    else:
        print("No solution found !")
