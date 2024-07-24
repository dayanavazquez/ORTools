from enum import Enum
from problems.strategy_type import HeuristicType, MetaheuristicType
from distances.distance_type import DistanceType
from load_data.instance_type import InstanceType
from problems.execute import cvrp, dvrp, mdvrp, vrppd, vrptw, tsp


class ProblemType(Enum):
    CVRP = 'cvrp'
    VRPTW = 'vrptw'
    DVRP = 'dvrp'
    TSP = 'tsp'
    MDVRP = 'mdvrp'
    VRPPD = 'vrppd'


def execute_problem(problem_type: ProblemType, instance, executions=None, distance_type: DistanceType = None,
                    time_limit=None, vehicle_maximum_travel_distance=None, vehicle_max_time=None, vehicle_speed=None, heuristic: HeuristicType = None, metaheuristic: MetaheuristicType = None):
    if not time_limit:
        time_limit = 20
    if not vehicle_maximum_travel_distance:
        vehicle_maximum_travel_distance = 500
    if not executions:
        executions = 1
    for i in range(executions):
        if problem_type == ProblemType.CVRP:
            return cvrp.execute(i, instance, time_limit, distance_type, heuristic, metaheuristic)
        if problem_type == ProblemType.DVRP:
            return dvrp.execute(i, instance, time_limit, vehicle_maximum_travel_distance, distance_type, heuristic, metaheuristic)
        if problem_type == ProblemType.TSP:
            return tsp.execute(i, instance, time_limit, distance_type, heuristic, metaheuristic)
        if problem_type == ProblemType.VRPTW:
            return vrptw.execute(i, instance, time_limit, vehicle_maximum_travel_distance, vehicle_max_time, vehicle_speed, distance_type, heuristic, metaheuristic)
        if problem_type == ProblemType.MDVRP:
            return mdvrp.execute(i, instance, time_limit, vehicle_maximum_travel_distance, distance_type, heuristic, metaheuristic)
        if problem_type == ProblemType.VRPPD:
            return vrppd.execute(i, instance, time_limit, vehicle_maximum_travel_distance, distance_type, heuristic, metaheuristic)
        return "The problem type is not supported"
