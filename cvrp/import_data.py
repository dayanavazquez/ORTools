import os
import math


#######################
# DATA NANDA
#######################

def calculate_distance(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    return int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + 0.5)


def read_file(file):
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


def process_files(directories):
    total_data = {}
    for directory in directories:
        files = os.listdir(directory)
        for file in files:
            if file.endswith(".txt"):
                route = os.path.join(directory, file)
                data = read_file(route)
                total_data[file] = data
    return total_data


path = ["../instances/cvrp_instances"]
data = process_files(path)
print(data)
