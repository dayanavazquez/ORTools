import weakref
import os
import time
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from load_data.instance_type import process_files
from distances.distance_type import DistanceType
from problems.strategy_type import HeuristicType, MetaheuristicType
from utils.get_strategies import get_strategies
from utils.utils import get_distance_and_solution_name


def save_solution(routing_manager, routing_model, instance, heuristic, metaheuristic, elapsed_time, i, distance_type):
    """Saves solution to a text file."""
    # Create the solutions_vrp_01 directory if it doesn't exist
    distance_type, solution_name = get_distance_and_solution_name(distance_type, heuristic, metaheuristic)
    solutions_dir = os.path.join(f"problems/{distance_type}/solutions_dvrp_{i}/solutions_{solution_name}")
    try:
        os.makedirs(solutions_dir, exist_ok=True)
        print(f"Directory {solutions_dir} created successfully or already exists.")
    except OSError as error:
        print(f"Error creating directory {solutions_dir}: {error}")
        return
    file_name = os.path.join(solutions_dir, f"solution_{instance}")
    try:
        with open(file_name, "w") as file:
            file.write(f"Instance: {instance}\n\n")
            file.write(f"Objective: {routing_model.CostVar().Value()}\n\n")
            file.write(f"Execution Time: {elapsed_time}\n\n")
            file.write(f"Heuristic: {heuristic}\n\n")
            file.write(f"Metaheuristic: {metaheuristic}\n\n")
            file.write(f"Distance type: {distance_type}\n\n")
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
        print(f"Solution saved successfully in {file_name}")
    except OSError as error:
        print(f"Error writing to file {file_name}: {error}")


class SolutionCallback:
    """Create a solution callback."""

    def __init__(self, manager, model, limit, instance, search_parameters, first_solution_strategy,
                 local_search_metaheuristic, i, distance_type, elapsed_time):

        # We need a weak ref on the routing model to avoid a cycle.
        self._routing_manager_ref = weakref.ref(manager)
        self._routing_model_ref = weakref.ref(model)
        self._counter = 0
        self._counter_limit = limit
        self.objectives = []
        self.instance = instance
        self.search_parameters = search_parameters
        self.first_solution_strategy = first_solution_strategy
        self.local_search_metaheuristic = local_search_metaheuristic
        self.elapsed_time = elapsed_time
        self.i = i
        self.distance_type = distance_type

    def __call__(self):
        objective = int(self._routing_model_ref().CostVar().Value())
        if not self.objectives or objective < self.objectives[-1]:
            self.objectives.append(objective)
            self._counter += 1
        if self._counter >= self._counter_limit:
            save_solution(self._routing_manager_ref(), self._routing_model_ref(), self.instance,
                          self.first_solution_strategy, self.local_search_metaheuristic, self.elapsed_time, self.i,
                          self.distance_type)
            self._routing_model_ref().solver().FinishCurrentSearch()


def execute(i, instance_type, time_limit, vehicle_maximum_travel_distance, distance_type: DistanceType = None,
            heuristic: HeuristicType = None,
            metaheuristic: MetaheuristicType = None, initial_routes=None):
    """Entry point of the program."""
    # Instantiate the data problem.
    instances_data = process_files(instance_type, distance_type)
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
            vehicle_maximum_travel_distance,  # vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name,
        )
        distance_dimension = routing_model.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)

        first_solution_strategies, local_search_metaheuristics = get_strategies(heuristic, metaheuristic)

        if not first_solution_strategies and local_search_metaheuristics:
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.local_search_metaheuristic = getattr(
                routing_enums_pb2.LocalSearchMetaheuristic, local_search_metaheuristics[0]
            )
            get_solutions(i, initial_routes, distance_type, search_parameters, routing_model, time_limit,
                          routing_manager, instance,
                          local_search_metaheuristics[0])
        elif not local_search_metaheuristics and first_solution_strategies:
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = getattr(
                routing_enums_pb2.FirstSolutionStrategy, first_solution_strategies[0]
            )
            get_solutions(i, initial_routes, distance_type, search_parameters, routing_model, time_limit,
                          routing_manager, instance,
                          first_solution_strategies[0])
        else:
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = getattr(
                routing_enums_pb2.FirstSolutionStrategy, first_solution_strategies[0]
            )
            search_parameters.local_search_metaheuristic = getattr(
                routing_enums_pb2.LocalSearchMetaheuristic, local_search_metaheuristics[0]
            )
            get_solutions(i, initial_routes, distance_type, search_parameters, routing_model, time_limit,
                          routing_manager, instance,
                          first_solution_strategies[0], local_search_metaheuristics[0])


def get_solutions(
        i, initial_routes, distance_type, search_parameters, routing_model, time_limit, routing_manager,
        instance,
        first_solution_strategy=None,
        local_search_metaheuristic=None
):
    search_parameters.time_limit.FromSeconds(time_limit)
    # Measure the time taken to solve the problem
    start_time = time.time()
    # Attach a solution callback.
    solution_callback = SolutionCallback(
        routing_manager, routing_model, 20, instance, search_parameters,
        first_solution_strategy, local_search_metaheuristic,
        i, distance_type, elapsed_time=0
    )
    routing_model.AddAtSolutionCallback(solution_callback)
    # Solve the problem.
    if initial_routes:
        filtered_routes = [route for route in initial_routes if len(route) > 2]
        print("Initial Routes:", filtered_routes)
        initial_solution = routing_model.ReadAssignmentFromRoutes(filtered_routes, True)
        routing_model.SolveFromAssignmentWithParameters(
            initial_solution, search_parameters
        )
    else:
        routing_model.SolveWithParameters(search_parameters)
    end_time = time.time()
    # Update the elapsed time in the solution callback
    solution_callback.elapsed_time = end_time - start_time
