from problems.problem_type import ProblemType, execute
from distances.distance_type import DistanceType
from load_data.instance_type import InstanceType
from problems.strategy_type import HeuristicType, MetaheuristicType

############
#  RUN
############
# Arguments:
# 1. problem_type => variante VRP que se desea ejecutar de las 6 disponibles
# 2. instance => si desea un conjunto de instancias del problema elegir de InstanceType.el grupo que desee, si es una única instancia poner la ruta relativa. Ejemplo: ./instances/hfvrp_instances/CVRP_1.txt
# 3. distance_type => elegir el tipo de distancia que se desea utilizar de las 4 disponibles (si no se elige por defecto se usa Manhattan)
# 4. executions => cantidad de ejecuciones (si no se elige, se pone 1 por defecto)
# 5. time_limit => tiempo máximo en segundos en que se va a demorar el algoritmo en devolver la solución
# 6. vehicle_maximum_travel_distance => distancia máxima que puede recorrer un vehículo (si no se elige se pone 500 por defecto, excepto a VRPTW que se pone 1000) (restricción válida para todos los problemas menos TSP y CVRP)
# 7. vehicle_max_time => máximo tiempo en que un vehículo debe completar su ruta (solo para VRPTW, se pone 1500 por defecto)
# 8. vehicle_speed => velocidad del vehículo (solo para VRPTW, se pone 83.33 km/h por defecto)
# 9. heuristic => heurística específica que se desea utilizar
# 10. metaheuristic => metaheurística específica que se desea utilizar
# 11. initial_routes => rutas iniciales para reoptimizar en tiempo real
# (si no se elige una heurística ni una metaheurística entonces se ejecutan todas por defecto)


execute(
    problem_type=ProblemType.CVRP,
    instance=InstanceType.VRPTW,
    distance_type=DistanceType.HAVERSINE,
    time_limit=60,
    executions=10,
    vehicle_maximum_travel_distance=None,
    vehicle_max_time=None,
    vehicle_speed=None,
    heuristic=None,
    metaheuristic=None,
    initial_routes=None,
    #[
    #   [8, 16, 14, 13, 12, 11],
    #   [3, 4, 9, 10],
    #   [15, 1],
    #   [7, 5, 2, 6],
    #],
)
