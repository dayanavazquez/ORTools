from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from problems.strategy_type import HeuristicType, MetaheuristicType
from distances.distance_type import DistanceType
from load_data.instance_type import process_files
from utils.get_strategies import get_strategies
import os
import time


def save_solution(data, manager, routing, solution, instance, heuristic, metaheuristic, elapsed_time, i):
    """Saves solution to a text file."""
    if heuristic and not metaheuristic:
        solutions_dir = os.path.join(f"problems/manhattan/solutions_cvrp_{i}/solutions_{heuristic}")
    elif metaheuristic and not heuristic:
        solutions_dir = os.path.join(f"problems/manhattan/solutions_cvrp_{i}/solutions_{metaheuristic}")
    else:
        solutions_dir = os.path.join(f"problems/manhattan/solutions_cvrp_{i}/solutions_{heuristic}_&_{metaheuristic}")
    try:
        os.makedirs(solutions_dir, exist_ok=True)
        print(f"Directory {solutions_dir} created successfully or already exists.")
    except OSError as error:
        print(f"Error creating directory {solutions_dir}: {error}")
        return

    file_name = os.path.join(solutions_dir, f"solution_{instance}")
    try:
        with open(file_name, 'w') as file:
            file.write(f"Instance: {instance}\n\n")
            file.write(f"Objective: {solution.ObjectiveValue()}\n\n")
            file.write(f"Execution Time: {elapsed_time}\n\n")
            file.write(f"Heuristic: {heuristic}\n\n")
            file.write(f"Metaheuristic: {metaheuristic}\n\n")
            total_distance = 0
            total_load = 0
            for vehicle_id in range(data["num_vehicles"]):
                index = routing.Start(vehicle_id)
                plan_output = f"Route for vehicle {vehicle_id}:\n"
                route_distance = 0
                route_load = 0
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    route_load += data["demands"][node_index]
                    plan_output += f" {node_index} Load({route_load}) -> "
                    previous_index = index
                    index = solution.Value(routing.NextVar(index))
                    route_distance += routing.GetArcCostForVehicle(
                        previous_index, index, vehicle_id
                    )
                plan_output += f" {manager.IndexToNode(index)} Load({route_load})\n"
                plan_output += f"Distance of the route: {route_distance}m\n"
                plan_output += f"Load of the route: {route_load}\n\n"
                file.write(plan_output)
                total_distance += route_distance
                total_load += route_load
            file.write(f"Total distance of all routes: {total_distance}m\n")
            file.write(f"Total load of all routes: {total_load}\n")
        print(f"Solution saved successfully in {file_name}")
    except OSError as error:
        print(f"Error writing to file {file_name}: {error}")


def execute(i, instance_type, time_limit, distance_type: DistanceType = None, heuristic: HeuristicType = None,
            metaheuristic: MetaheuristicType = None):
    """Solve the manhattan problem."""
    # Instantiate the data problem.
    instances_data = process_files(instance_type, distance_type)
    for instance, data in instances_data.items():
        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
        )
        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        # Create and register a transit callback.
        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["distance_matrix"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Capacity constraint.
        def demand_callback(from_index):
            """Returns the demand of the node."""
            # Convert from routing variable Index to demands NodeIndex.
            from_node = manager.IndexToNode(from_index)
            return data["demands"][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            data["vehicle_capacities"],  # vehicle maximum capacities
            True,  # start cumul to zero
            "Capacity",
        )
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
