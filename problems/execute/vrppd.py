from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from load_data.instance_type import process_files
import os
import time
from distances.distance_type import DistanceType
from problems.strategy_type import HeuristicType, MetaheuristicType
from utils.get_strategies import get_strategies


def save_solution(data, manager, routing, solution, instance, heuristic, metaheuristic, elapsed_time, i):
    """Saves solution to a file."""
    if heuristic and not metaheuristic:
        output_dir = os.path.join(f"problems/vrppd/solutions_vrppd_{i}/solutions_{heuristic}")
    elif metaheuristic and not heuristic:
        output_dir = os.path.join(f"problems/vrppd/solutions_vrppd_{i}/solutions_{metaheuristic}")
    else:
        output_dir = os.path.join(f"problems/vrppd/solutions_vrppd_{i}/solutions_{heuristic}_&_{metaheuristic}")
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Directory {output_dir} created successfully or already exists.")
    except OSError as error:
        print(f"Error creating directory {output_dir}: {error}")
        return
    filename = os.path.join(output_dir, f'{instance}')
    try:
        with open(filename, 'w') as f:
            f.write(f"Instance: {instance}\n\n")
            f.write(f"Objective: {solution.ObjectiveValue()}\n\n")
            f.write(f"Execution Time: {elapsed_time}\n\n")
            f.write(f"Heuristic: {heuristic}\n\n")
            f.write(f"Metaheuristic: {metaheuristic}\n\n")
            total_distance = 0
            for vehicle_id in range(data["num_vehicles"]):
                index = routing.Start(vehicle_id)
                plan_output = f"Route for vehicle {vehicle_id}:\n"
                route_distance = 0
                while not routing.IsEnd(index):
                    plan_output += f" {manager.IndexToNode(index)} -> "
                    previous_index = index
                    index = solution.Value(routing.NextVar(index))
                    route_distance += routing.GetArcCostForVehicle(
                        previous_index, index, vehicle_id
                    )
                plan_output += f"{manager.IndexToNode(index)}\n"
                plan_output += f"Distance of the route: {route_distance}m\n\n"
                f.write(plan_output)
                total_distance += route_distance
            f.write(f"Total Distance of all routes: {total_distance}m")
        print(f"Solution saved successfully in {filename}")
    except OSError as error:
        print(f"Error writing to file {filename}: {error}")


def execute(i, instance_type, time_limit, vehicle_maximum_travel_distance, distance_type: DistanceType = None, heuristic: HeuristicType = None,
            metaheuristic: MetaheuristicType = None):
    """Entry point of the program."""
    # Instantiate the data problem.
    instances_data = process_files(instance_type, distance_type)
    for instance, data in instances_data.items():
        manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
        )
        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)
        routing = pywrapcp.RoutingModel(manager)

        # Define cost of each arc.
        def distance_callback(from_index, to_index):
            """Returns the manhattan distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["distance_matrix"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        # Add Distance constraint.
        dimension_name = "Distance"
        routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            vehicle_maximum_travel_distance,  # vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name,
        )
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)
        # Define Transportation Requests.
        for request in data["pickups_deliveries"]:
            pickup_index = manager.NodeToIndex(request[0])
            delivery_index = manager.NodeToIndex(request[1])
            routing.AddPickupAndDelivery(pickup_index, delivery_index)
            routing.solver().Add(
                routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index)
            )
            routing.solver().Add(
                distance_dimension.CumulVar(pickup_index)
                <= distance_dimension.CumulVar(delivery_index)
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
