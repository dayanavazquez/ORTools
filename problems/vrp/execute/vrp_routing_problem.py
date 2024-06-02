import weakref
import os
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from load_data.instance_type import process_files
from load_data.instance_type import InstanceType


def save_solution(routing_manager, routing_model, instance):
    """Saves solution to a text file."""
    # Create the solutions_vrp_01 directory if it doesn't exist
    solutions_dir = os.path.join("../solutions/solutions_vrp")
    os.makedirs(solutions_dir, exist_ok=True)
    file_name = os.path.join(solutions_dir, f"solution_{instance}")
    with open(file_name, "w") as file:
        file.write(f"Instance: {instance}\n\n")
        file.write(f"Solution objective: {routing_model.CostVar().Value()}\n\n")
        total_distance = 0
        for vehicle_id in range(routing_manager.GetNumberOfVehicles()):
            index = routing_model.Start(vehicle_id)
            plan_output = f"Route for vehicle {vehicle_id}:\n"
            route_distance = 0
            while not routing_model.IsEnd(index):
                plan_output += f" {routing_manager.IndexToNode(index)} ->"
                previous_index = index
                index = routing_model.NextVar(index).Value()
                route_distance += routing_model.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )
            plan_output += f" {routing_manager.IndexToNode(index)}\n"
            plan_output += f"Distance of the route: {route_distance}m\n\n"
            file.write(plan_output)
            total_distance += route_distance
        file.write(f"Total Distance of all routes: {total_distance}m\n")


class SolutionCallback:
    """Create a solution callback."""

    def __init__(self, manager, model, limit, instance):
        # We need a weak ref on the routing model to avoid a cycle.
        self._routing_manager_ref = weakref.ref(manager)
        self._routing_model_ref = weakref.ref(model)
        self._counter = 0
        self._counter_limit = limit
        self.objectives = []
        self.instance = instance

    def __call__(self):
        objective = int(self._routing_model_ref().CostVar().Value())
        if not self.objectives or objective < self.objectives[-1]:
            self.objectives.append(objective)
            self._counter += 1
        if self._counter >= self._counter_limit:
            save_solution(self._routing_manager_ref(), self._routing_model_ref(), self.instance)
            self._routing_model_ref().solver().FinishCurrentSearch()


def execute():
    """Entry point of the program."""
    # Instantiate the data problem.
    instances_data = process_files(InstanceType.BHCVRP)
    for instance, data in instances_data.items():
        # Create the routing index manager.
        routing_manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
        )
        # Create Routing Model.
        routing_model = pywrapcp.RoutingModel(routing_manager)

        # Create and register a transit callback.
        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = routing_manager.IndexToNode(from_index)
            to_node = routing_manager.IndexToNode(to_index)
            return data["distance_matrix"][from_node][to_node]

        transit_callback_index = routing_model.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing_model.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Distance constraint.
        dimension_name = "Distance"
        routing_model.AddDimension(
            transit_callback_index,
            0,  # no slack
            500,  # vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name,
        )
        distance_dimension = routing_model.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)

        # Attach a solution callback.
        solution_callback = SolutionCallback(routing_manager, routing_model, 15, instance)
        routing_model.AddAtSolutionCallback(solution_callback)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.FromSeconds(5)

        # Solve the problem.
        routing_model.SolveWithParameters(search_parameters)
