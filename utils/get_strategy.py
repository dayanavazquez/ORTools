from ortools.constraint_solver import routing_enums_pb2


# Utilidad para obtener el nombre de la estrategia
def get_first_solution_strategy_name(value):
    return routing_enums_pb2.FirstSolutionStrategy.Name(value)


def get_local_search_metaheuristic_name(value):
    return routing_enums_pb2.LocalSearchMetaheuristic.Name(value)
