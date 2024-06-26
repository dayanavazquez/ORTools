from enum import Enum
from problems.execute import cvrp, dvrp, mdvrp, vrppd, cvrptw, tsp


class ProblemType(Enum):
    CVRP = 'cvrp'
    CVRPTW = 'cvrptw'
    DVRP = 'dvrp'
    TSP = 'tsp'
    MDVRP = 'mdvrp'
    VRPPD = 'vrppd'


def execute_problem(problem_type: ProblemType):
    if problem_type == ProblemType.CVRP:
        return cvrp.execute()
    if problem_type == ProblemType.DVRP:
        return dvrp.execute()
    if problem_type == ProblemType.TSP:
        return tsp.execute()
    if problem_type == ProblemType.CVRPTW:
        return cvrptw.execute()
    if problem_type == ProblemType.MDVRP:
        return mdvrp.execute()
    if problem_type == ProblemType.VRPPD:
        return vrppd.execute()
    return "The problem type is not supported"
