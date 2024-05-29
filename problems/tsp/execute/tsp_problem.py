import os
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from load_data.instance_type import process_files
from distances.distance_type import calculate_distance, DistanceType
from load_data.instance_type import InstanceType


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


def save_solution(manager, routing, solution, instance):
    """Saves solution to a text file."""
    solutions_dir = os.path.join("../solutions")
    os.makedirs(solutions_dir, exist_ok=True)
    file_name = os.path.join(solutions_dir, f"solution_{instance}")
    with open(file_name, 'w') as f:
        f.write(f"Instance: {instance}\n\n")
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
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)
        # Print solution on console.
        if solution:
            save_solution(manager, routing, solution, instance)


if __name__ == "__main__":
    execute()
