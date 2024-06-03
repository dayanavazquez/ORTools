"""Simple Vehicles Routing Problem."""
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from load_data.instance_type import process_files
from load_data.instance_type import InstanceType
import os


def save_solution_to_file(data, manager, routing, solution, instance):
    """Saves solution to a file."""
    output_dir = 'solutions_mdvrp'
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
            max_route_distance = 0
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
                max_route_distance = max(route_distance, max_route_distance)
            f.write(f"Maximum of the route distances: {max_route_distance}m\n")
        print(f"Solution saved successfully in {filename}")
    except OSError as error:
        print(f"Error writing to file {filename}: {error}")


def execute():
    """Entry point of the program."""
    # Instantiate the data problem.
    instances_data = process_files(InstanceType.MDCVRP)
    for instance, data in instances_data.items():
        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), data["num_vehicles"], data["starts"], data["ends"]
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
        # Add Distance constraint.
        dimension_name = "Distance"
        routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            2000,  # vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name,
        )
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)
        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)
        # Print solution on console.
        if solution:
            save_solution_to_file(data, manager, routing, solution, instance)
