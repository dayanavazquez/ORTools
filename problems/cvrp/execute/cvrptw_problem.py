from functools import partial
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
from load_data.instance_type import process_files
from distances.distance_type import calculate_distance
from load_data.instance_type import InstanceType
import os


def create_distance_evaluator(data):
    """Creates callback to return distance between points."""
    _distances = {}
    # precompute distance between location to have distance callback in O(1)
    for from_node in range(data['num_locations']):
        _distances[from_node] = {}
        for to_node in range(data['num_locations']):
            if from_node == to_node:
                _distances[from_node][to_node] = 0
            # Forbid start/end/reload node to be consecutive.
            elif from_node in range(6) and to_node in range(6):
                _distances[from_node][to_node] = data['vehicle_max_distance']
            else:
                _distances[from_node][to_node] = (calculate_distance(
                    data['locations'][from_node], data['locations'][to_node]))

    def distance_evaluator(manager, from_node, to_node):
        """Returns the manhattan distance between the two nodes"""
        return _distances[manager.IndexToNode(from_node)][manager.IndexToNode(
            to_node)]

    return distance_evaluator


def add_distance_dimension(routing, manager, data, distance_evaluator_index):
    """Add Global Span constraint"""
    del manager
    distance = 'Distance'
    routing.AddDimension(
        distance_evaluator_index,
        0,  # null slack
        data['vehicle_max_distance'],  # maximum distance per vehicle
        True,  # start cumul to zero
        distance)
    distance_dimension = routing.GetDimensionOrDie(distance)
    # Try to minimize the max distance among vehicles.
    # /!\ It doesn't mean the standard deviation is minimized
    distance_dimension.SetGlobalSpanCostCoefficient(100)


def create_demand_evaluator(data):
    """Creates callback to get demands at each location."""
    _demands = data['demands']

    def demand_evaluator(manager, from_node):
        """Returns the demand of the current node"""
        return _demands[manager.IndexToNode(from_node)]

    return demand_evaluator


def add_capacity_constraints(routing, manager, data, demand_evaluator_index):
    """Adds capacity constraint"""
    vehicle_capacity = data['vehicle_capacity']
    capacity = 'Capacity'
    routing.AddDimension(
        demand_evaluator_index,
        vehicle_capacity,
        vehicle_capacity,
        True,  # start cumul to zero
        capacity)

    # Add Slack for reseting to zero unload depot nodes.
    # e.g. vehicle with load 10/15 arrives at node 1 (depot unload)
    # so we have CumulVar = 10(current load) + -15(unload) + 5(slack) = 0.
    capacity_dimension = routing.GetDimensionOrDie(capacity)
    # Allow to drop reloading nodes with zero cost.
    for node in [1, 2, 3, 4, 5]:
        node_index = manager.NodeToIndex(node)
        routing.AddDisjunction([node_index], 0)

    # Allow to drop regular node with a cost.
    for node in range(6, len(data['demands'])):
        node_index = manager.NodeToIndex(node)
        capacity_dimension.SlackVar(node_index).SetValue(0)
        routing.AddDisjunction([node_index], 100_000)


def create_time_evaluator(data):
    def travel_time(data, from_node, to_node):
        """Gets the travel times between two locations."""
        if from_node == to_node:
            travel_time = 0
        else:
            travel_time = calculate_distance(
                data['locations'][from_node],
                data['locations'][to_node]) / data['vehicle_speed']
        return travel_time

    _total_time = {}
    # precompute total time to have time callback in O(1)
    for from_node in range(data['num_locations']):
        _total_time[from_node] = {}
        for to_node in range(data['num_locations']):
            if from_node == to_node:
                _total_time[from_node][to_node] = 0
            else:
                _total_time[from_node][to_node] = int(
                    data['service_time'][from_node] +
                    travel_time(data, from_node, to_node))

    def time_evaluator(manager, from_node, to_node):
        """Returns the total time between the two nodes"""
        return _total_time[manager.IndexToNode(from_node)][manager.IndexToNode(
            to_node)]

    return time_evaluator


def add_time_window_constraints(routing, manager, data, time_evaluator):
    """Add Time windows constraint"""
    time = 'Time'
    max_time = data['vehicle_max_time']
    routing.AddDimension(
        time_evaluator,
        max_time,  # allow waiting time
        max_time,  # maximum time per vehicle
        False,  # don't force start cumul to zero since we are giving TW to start nodes
        time)
    time_dimension = routing.GetDimensionOrDie(time)
    # Add time window constraints for each location except depot
    # and 'copy' the slack var in the solution object (aka Assignment) to print it
    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx == 0:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
        routing.AddToAssignment(time_dimension.SlackVar(index))
    # Add time window constraints for each vehicle start node
    # and 'copy' the slack var in the solution object (aka Assignment) to print it
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(data['time_windows'][0][0],
                                                data['time_windows'][0][1])
        routing.AddToAssignment(time_dimension.SlackVar(index))
        # Warning: Slack var is not defined for vehicle's end node
        # routing.AddToAssignment(time_dimension.SlackVar(self.routing.End(vehicle_id)))


def save_solution_to_file(data, manager, routing, assignment, instance, heuristic, metaheuristic):  # pylint:disable=too-many-locals
    """Prints assignment on console"""
    output_dir = os.path.join(f"problems/cvrp/solutions/solutions_cvrptw/solutions_{heuristic}_{metaheuristic}")
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Directory {output_dir} created successfully or already exists.")
    except OSError as error:
        print(f"Error creating directory {output_dir}: {error}")
        return
    filename = os.path.join(output_dir, f'{instance}')
    try:
        with open(filename, 'w') as f:
            f.write(f'Instance: {instance}\n\n')
            f.write(f'Objective: {assignment.ObjectiveValue()}\n\n')
            f.write(f"Heuristic: {heuristic}\n\n")
            f.write(f"Metaheuristic: {metaheuristic}\n\n")
            total_distance = 0
            total_load = 0
            total_time = 0
            capacity_dimension = routing.GetDimensionOrDie('Capacity')
            time_dimension = routing.GetDimensionOrDie('Time')
            dropped = []
            for order in range(6, routing.nodes()):
                index = manager.NodeToIndex(order)
                if assignment.Value(routing.NextVar(index)) == index:
                    dropped.append(order)
            f.write(f'dropped orders: {dropped}\n\n')
            for reload in range(1, 6):
                index = manager.NodeToIndex(reload)
                if assignment.Value(routing.NextVar(index)) == index:
                    dropped.append(reload)
            f.write(f'dropped reload stations: {dropped}\n\n')

            for vehicle_id in range(data['num_vehicles']):
                index = routing.Start(vehicle_id)
                plan_output = f'Route for vehicle {vehicle_id}:\n'
                distance = 0
                while not routing.IsEnd(index):
                    load_var = capacity_dimension.CumulVar(index)
                    time_var = time_dimension.CumulVar(index)
                    plan_output += (
                        f' {manager.IndexToNode(index)} '
                        f'Load({assignment.Min(load_var)}) '
                        f'Time({assignment.Min(time_var)},{assignment.Max(time_var)}) ->'
                    )
                    previous_index = index
                    index = assignment.Value(routing.NextVar(index))
                    distance += routing.GetArcCostForVehicle(previous_index, index,
                                                             vehicle_id)
                load_var = capacity_dimension.CumulVar(index)
                time_var = time_dimension.CumulVar(index)
                plan_output += (
                    f' {manager.IndexToNode(index)} '
                    f'Load({assignment.Min(load_var)}) '
                    f'Time({assignment.Min(time_var)},{assignment.Max(time_var)})\n')
                plan_output += f'Distance of the route: {distance}m\n'
                plan_output += f'Load of the route: {assignment.Min(load_var)}\n'
                plan_output += f'Time of the route: {assignment.Min(time_var)}min\n\n'
                f.write(plan_output)
                total_distance += distance
                total_load += assignment.Min(load_var)
                total_time += assignment.Min(time_var)
            f.write(f'Total Distance of all routes: {total_distance}m')
            f.write(f'Total Load of all routes: {total_load}')
            f.write(f'Total Time of all routes: {total_time}min')
        print(f"Solution saved successfully in {filename}")
    except OSError as error:
        print(f"Error writing to file {filename}: {error}")


def execute():
    # Instantiate the data problem.
    instances_data = process_files(InstanceType.CVRPTW)
    for instance, data in instances_data.items():
        # Create the routing index manager
        manager = pywrapcp.RoutingIndexManager(data['num_locations'],
                                               data['num_vehicles'], data['depot'])
        # Create Routing Model
        routing = pywrapcp.RoutingModel(manager)

        # Define weight of each edge
        distance_evaluator_index = routing.RegisterTransitCallback(
            partial(create_distance_evaluator(data), manager))
        routing.SetArcCostEvaluatorOfAllVehicles(distance_evaluator_index)

        # Add Distance constraint to minimize the longuest route
        add_distance_dimension(routing, manager, data, distance_evaluator_index)

        # Add Capacity constraint
        demand_evaluator_index = routing.RegisterUnaryTransitCallback(
            partial(create_demand_evaluator(data), manager))
        add_capacity_constraints(routing, manager, data, demand_evaluator_index)

        # Add Time Window constraint
        time_evaluator_index = routing.RegisterTransitCallback(
            partial(create_time_evaluator(data), manager))
        add_time_window_constraints(routing, manager, data, time_evaluator_index)

        # Setting first solution heuristic (cheapest addition).
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC
        )
        search_parameters.time_limit.FromSeconds(15)
        solution = routing.SolveWithParameters(search_parameters)
        if solution:
            save_solution_to_file(data, manager, routing, solution, instance, search_parameters.first_solution_strategy, search_parameters.local_search_metaheuristic)
        else:
            print("No solution found !")
