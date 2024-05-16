import os
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from tsp.import_data import process_files


# [START distance_callback]
def create_distance_callback(data, manager):
    """Creates callback to return distance between points."""
    distances_ = {}
    index_manager_ = manager
    # precompute distance between location to have distance callback in O(1)
    for from_counter, from_node in enumerate(data["locations"]):
        distances_[from_counter] = {}
        for to_counter, to_node in enumerate(data["locations"]):
            if from_counter == to_counter:
                distances_[from_counter][to_counter] = 0
            else:
                distances_[from_counter][to_counter] = abs(
                    from_node[0] - to_node[0]
                ) + abs(from_node[1] - to_node[1])

    def distance_callback(from_index, to_index):
        """Returns the manhattan distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = index_manager_.IndexToNode(from_index)
        to_node = index_manager_.IndexToNode(to_index)
        return distances_[from_node][to_node]

    return distance_callback


def save_solution(manager, routing, assignment, instance):
    """Saves assignment to a text file."""
    # Create the solutions_bhcvrp directory if it doesn't exist
    solutions_dir = os.path.join("../solutions")
    os.makedirs(solutions_dir, exist_ok=True)
    file_name = os.path.join(solutions_dir, f"solution_{instance}")
    with open(file_name, "w") as file:
        file.write(f"Instance: {instance}\n\n")
        file.write(f"Objective: {assignment.ObjectiveValue()}\n\n")
        index = routing.Start(0)
        plan_output = "Route for vehicle 0:\n"
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += f" {manager.IndexToNode(index)} ->"
            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        plan_output += f" {manager.IndexToNode(index)}\n"
        plan_output += f"Distance of the route: {route_distance}m\n\n"
        file.write(plan_output)


def main():
    """Entry point of the program."""
    # Instantiate the data problem.
    instances_data = process_files()
    for instance, data in instances_data.items():
        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(
            len(data["locations"]), data["num_vehicles"], data["depot"]
        )
        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)
        # Create and register a transit callback.
        distance_callback = create_distance_callback(data, manager)
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        # Solve the problem.
        assignment = routing.SolveWithParameters(search_parameters)
        # Save solution on console.
        if assignment:
            save_solution(manager, routing, assignment, instance)


if __name__ == "__main__":
    main()
