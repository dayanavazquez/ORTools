from enum import Enum
from problems.cvrp.execute import cvrptw_problem
from problems.cvrp.execute import vrp_capacity_problem
from problems.tsp.execute import tsp_problem
from problems.vrp.execute import vrp_multiple_depots_problem
from problems.vrp.execute import vrp_routing_problem
from problems.vrp.execute import vrp_pickup_delivery_problem


class ProblemType(Enum):
    CVRP = 'cvrp'
    CVRPTW = 'cvrptw'
    VRP = 'vrp'
    TSP = 'tsp'
    MDVRP = 'mdvrp'
    VRPPD = 'vrppd'


def execute_problem(problem_type: ProblemType):
    if problem_type == ProblemType.CVRP:
        return vrp_capacity_problem.execute()
    if problem_type == ProblemType.VRP:
        vrp_routing_problem.execute()
    if problem_type == ProblemType.TSP:
        tsp_problem.execute()
    if problem_type == ProblemType.CVRPTW:
        return cvrptw_problem.execute()
    if problem_type == ProblemType.MDVRP:
        return vrp_multiple_depots_problem.execute()
    if problem_type == ProblemType.VRPPD:
        return vrp_pickup_delivery_problem.execute()
    return "The problem type is not supported"
