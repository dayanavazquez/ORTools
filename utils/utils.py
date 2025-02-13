import time
from utils.get_strategies import get_strategies
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def get_distance_and_solution_name(distance_type, heuristic, metaheuristic):
    distance_type = distance_type.value if distance_type else "manhattan"
    solution_name = heuristic if heuristic and not metaheuristic else metaheuristic if metaheuristic and not heuristic else f"{heuristic}_&_{metaheuristic}"
    return distance_type, solution_name


def get_solutions(
        initial_routes, save_solution, i, distance_type, search_parameters, routing, time_limit, data, manager,
        instance,
        first_solution_strategy=None,
        local_search_metaheuristic=None
):
    try:
        search_parameters.time_limit.FromSeconds(time_limit)
        start_time = time.time()
        if initial_routes:
            filtered_routes = [route for route in initial_routes if len(route) > 2]
            print("Initial Routes:", filtered_routes)
            initial_solution = routing.ReadAssignmentFromRoutes(filtered_routes, True)
            solution = routing.SolveFromAssignmentWithParameters(
                initial_solution, search_parameters
            )
        else:
            solution = routing.SolveWithParameters(search_parameters)
        end_time = time.time()
        elapsed_time = end_time - start_time
        if solution:
            save_solution(
                data, manager, routing, solution, instance, first_solution_strategy,
                local_search_metaheuristic,
                elapsed_time, i, distance_type)
        else:
            print("No solution found !")
    except OSError as error:
        print(f"Error writing in the initial routes {initial_routes}: {error}")


def execute_solution(
        save_solution, heuristic, metaheuristic, i, distance_type, routing, time_limit, data, manager, instance,
        initial_routes
):
    first_solution_strategies, local_search_metaheuristics = get_strategies(heuristic, metaheuristic)
    if not first_solution_strategies and local_search_metaheuristics:
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.local_search_metaheuristic = getattr(
            routing_enums_pb2.LocalSearchMetaheuristic, local_search_metaheuristics[0]
        )
        get_solutions(
            initial_routes, save_solution, i, distance_type, search_parameters, routing, time_limit, data, manager,
            instance,
            local_search_metaheuristics[0]
        )
    elif not local_search_metaheuristics and first_solution_strategies:
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = getattr(
            routing_enums_pb2.FirstSolutionStrategy, first_solution_strategies[0]
        )
        get_solutions(
            initial_routes, save_solution, i, distance_type, search_parameters, routing, time_limit, data, manager,
            instance,
            first_solution_strategies[0]
        )
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
                get_solutions(
                    initial_routes, save_solution, i, distance_type, search_parameters, routing, time_limit, data,
                    manager,
                    instance, first_solution_strategy, local_search_metaheuristic
                )
