from enum import Enum
from problems.execute import cvrp, dvrp, mdvrp, vrppd, vrptw, tsp


class ProblemType(Enum):
    CVRP = 'cvrp'
    VRPTW = 'vrptw'
    DVRP = 'dvrp'
    TSP = 'solutions_vrppd_10'
    MDVRP = 'mdvrp'
    VRPPD = 'vrppd'


def execute_problem(problem_type: ProblemType, i):
    if problem_type == ProblemType.CVRP:
        return cvrp.execute(i)
    if problem_type == ProblemType.DVRP:
        return dvrp.execute()
    if problem_type == ProblemType.TSP:
        return tsp.execute(i)
    if problem_type == ProblemType.VRPTW:
        return vrptw.execute(i)
    if problem_type == ProblemType.MDVRP:
        return mdvrp.execute()
    if problem_type == ProblemType.VRPPD:
        return vrppd.execute(i)
    return "The problem type is not supported"
