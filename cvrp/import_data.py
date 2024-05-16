import os
import math


#######################
# DATA CVRPMD
#######################


def calculate_distance(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    return int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + 0.5)


def read_file_md(file):
    locations = []
    demands = []
    num_locations = 0

    with open(file, 'r') as f:
        lines = f.readlines()
        num_vehicles = int(lines[0].split()[2])
        capacities = []
        for i in range(1, num_vehicles + 1):
            capacities.extend(map(int, lines[i].split()))

        for line in lines[1:]:
            if len(line.strip()) == 0:
                break
            data = line.split()
            if len(data) == 4:
                x_coord = float(data[1])
                y_coord = float(data[2])
                locations.append((x_coord, y_coord))
                demand = int(data[3]) if data[3] else 0
                demands.append(demand)
                num_locations += 1

    distance_matrix = []
    for i, coord1 in enumerate(locations):
        row = []
        for j, coord2 in enumerate(locations):
            if i == j:
                row.append(0)
            else:
                distance = calculate_distance(coord1, coord2)
                row.append(distance)
        distance_matrix.append(row)

    return {"distance_matrix": distance_matrix, "num_vehicles": num_vehicles, "vehicle_capacities": capacities,
            "demands": demands, "depot": 0}


def process_files_md():
    total_data = {}
    path = ["../../instances/cvrpmd_instances(Nanda)/C-mdvrp"]
    for directory in path:
        files = os.listdir(directory)
        for file in files:
            if file.endswith(".txt"):
                route = os.path.join(directory, file)
                data = read_file_md(route)
                total_data[file] = data
    return total_data


#######################
# DATA BHCVRP
#######################


def read_file(file):
    locations = []
    demands = []
    num_locations = 0

    with open(file, 'r') as f:
        lines = f.readlines()
        capacity = int(float(lines[0].split()[0]))
        num_vehicles = int(lines[0].split()[3])
        capacities = [capacity] * num_vehicles

        for line in lines[1:]:
            if len(line.strip()) == 0 or line.strip().startswith("EOF"):
                break
            data = line.split()
            if len(data) == 5:
                x_coord = float(data[1])
                y_coord = float(data[2])
                locations.append((x_coord, y_coord))
                demand = float(data[3]) if data[3] else 0
                demands.append(demand)
                num_locations += 1

    distance_matrix = []
    for i, coord1 in enumerate(locations):
        row = []
        for j, coord2 in enumerate(locations):
            if i == j:
                row.append(0)
            else:
                distance = calculate_distance(coord1, coord2)
                row.append(distance)
        distance_matrix.append(row)

    return {"distance_matrix": distance_matrix, "num_vehicles": num_vehicles, "vehicle_capacities": capacities,
            "demands": demands, "depot": 0}


def process_files():
    total_data = {}
    path = ['../../instances/bhcvrp_instances']
    for directory in path:
        files = os.listdir(directory)
        for file in files:
            if file.endswith(".txt"):
                route = os.path.join(directory, file)
                data = read_file(route)
                total_data[file] = data
    return total_data
