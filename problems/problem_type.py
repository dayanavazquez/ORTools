from enum import Enum
from problems.strategy_type import HeuristicType, MetaheuristicType
from distances.distance_type import DistanceType
from problems.execute import cvrp, mdvrp, vrppd, vrptw, tsp


class ProblemType(Enum):
    CVRP = 'cvrp'
    VRPTW = 'vrptw'
    TSP = 'tsp'
    MDVRP = 'mdvrp'
    VRPPD = 'vrppd'


def execute(
        problem_type: ProblemType, instance, distance_type: DistanceType = None,
        time_limit=None, executions=None, vehicle_maximum_travel_distance=None, vehicle_max_time=None,
        vehicle_speed=None, heuristic: HeuristicType = None, metaheuristic: MetaheuristicType = None,
        initial_routes=None
):
    if not executions:
        executions = 1
    if not time_limit:
        time_limit = 20
    if not vehicle_maximum_travel_distance:
        vehicle_maximum_travel_distance = 500
    for i in range(7, executions):
        execute_problem(
            i, problem_type, instance, distance_type, time_limit, vehicle_maximum_travel_distance,
            vehicle_max_time, vehicle_speed, heuristic, metaheuristic, initial_routes
        )


def execute_problem(
        i, problem_type: ProblemType, instance, distance_type: DistanceType = None,
        time_limit=None, vehicle_maximum_travel_distance=None, vehicle_max_time=None, vehicle_speed=None,
        heuristic: HeuristicType = None, metaheuristic: MetaheuristicType = None, initial_routes=None
):
    if problem_type == ProblemType.CVRP:
        return cvrp.execute(i, instance, time_limit, distance_type, heuristic, metaheuristic, initial_routes)
    if problem_type == ProblemType.TSP:
        return tsp.execute(i, instance, time_limit, distance_type, heuristic, metaheuristic, initial_routes)
    if problem_type == ProblemType.VRPTW:
        return vrptw.execute(i, instance, time_limit, vehicle_maximum_travel_distance, vehicle_max_time, vehicle_speed,
                             distance_type, heuristic, metaheuristic, initial_routes)
    if problem_type == ProblemType.MDVRP:
        return mdvrp.execute(i, instance, time_limit, vehicle_maximum_travel_distance, distance_type, heuristic,
                             metaheuristic, initial_routes)
    if problem_type == ProblemType.VRPPD:
        return vrppd.execute(i, instance, time_limit, vehicle_maximum_travel_distance, distance_type, heuristic,
                             metaheuristic, initial_routes)
    return print("The problem type is not supported")
