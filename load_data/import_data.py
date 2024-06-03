import math
import random
import json


#######################
# DATA BSS
#######################


def read_file_bss(file):
    with open(file, 'r') as file:
        json_data = json.load(file)
        locations = []
        for depot in json_data["depots"]:
            depot_coords = (depot["coordinate_x"], depot["coordinate_y"])
            locations.append(depot_coords)
        for bus_stop in json_data["bus_stops"]:
            bus_stop_coords = (bus_stop["coordinate_x"], bus_stop["coordinate_y"])
            locations.append(bus_stop_coords)

    size = len(locations)
    distance_matrix = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if i != j:
                distance_matrix[i][j] = calculate_distance(locations[i], locations[j])
    return {
        "locations": locations,
        "distance_matrix": distance_matrix,
        "num_vehicles": 6,
        "vehicle_capacities": [7, 7, 3, 5, 4, 6],
        "demands": [0, 2, 2, 2, 3, 3, 3, 3, 3],
        "depot": 0
    }

#######################
# DATA TSP
#######################

def read_file_tsp(file):
    with open(file, "r") as f:
        lines = f.readlines()
        node_coord_section_index = lines.index("NODE_COORD_SECTION\n")
        locations = []
        for line in lines[node_coord_section_index + 1:]:
            if line.strip() == "EOF":
                break
            parts = line.split()
            x_coord = float(parts[1])
            y_coord = float(parts[2])
            locations.append((x_coord, y_coord))
    return {
        "locations": locations,
        "num_vehicles": 1,
        "depot": 0
    }


#######################
# DATA CVRPMD
#######################

def read_file_md(file):
    locations = []
    demands = []
    num_locations = 0
    starts = []
    ends = []
    with open(file, 'r') as f:
        lines = f.readlines()
        num_depots = int(lines[0].split()[0])
        num_vehicles = int(lines[0].split()[2])
        capacities = []
        for i in range(1, num_vehicles + 1):
            capacities.append(int(lines[i].strip()))
        temp_lines = []
        for line in lines[num_vehicles + 1:]:
            if len(line.strip()) > 0:
                temp_lines.append(line)
        for line in temp_lines:
            data = line.split()
            if len(data) >= 3:
                x_coord = float(data[1])
                y_coord = float(data[2])
                locations.append((x_coord, y_coord))
                if len(data) == 4:
                    demand = int(data[3])
                else:
                    demand = 0
                demands.append(demand)
                num_locations += 1
        depot_indices = temp_lines[-num_depots:]
        for line in depot_indices:
            data = line.split()
            if len(data) >= 3:
                depot_index = int(data[0]) - 1
                starts.append(depot_index)
                ends.append(depot_index)
    all_ids = list(range(num_locations))
    all_ids.remove(0)
    random.shuffle(all_ids)
    pickups = all_ids[:num_locations // 2]
    deliveries = all_ids[num_locations // 2:]
    pickups_deliveries = [list(pair) for pair in zip(pickups, deliveries)]
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
    return {
        "distance_matrix": distance_matrix,
        "num_vehicles": num_vehicles,
        "vehicle_capacities": capacities,
        "demands": demands,
        "starts": starts,
        "ends": ends,
        "pickups_deliveries": pickups_deliveries,
        "depot": 0
    }


#######################
# DATA BHCVRP
#######################

def read_file_bh(file):
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
    all_ids = list(range(num_locations))
    all_ids.remove(0)
    random.shuffle(all_ids)
    pickups = all_ids[:num_locations // 2]
    deliveries = all_ids[num_locations // 2:]
    pickups_deliveries = [list(pair) for pair in zip(pickups, deliveries)]
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
            "demands": demands, "depot": 0, "pickups_deliveries": pickups_deliveries}


#######################
# DATA CVRPTW
#######################

def read_file_tw(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        num_vehicles, vehicle_capacities = map(int, lines[0].split())
        service_times = []
        locations = []
        demands = []
        time_windows = []
        for line in lines[2:]:
            parts = line.split()
            if len(parts) == 7:
                locations.append((int(parts[1]), int(parts[2])))
                demands.append(int(parts[3]))
                time_windows.append((int(parts[4]), int(parts[5])))
                service_times.append(int(parts[6]))
        num_locations = len(locations)
        all_ids = list(range(num_locations))
        all_ids.remove(0)
        random.shuffle(all_ids)
        pickups = all_ids[:num_locations // 2]
        deliveries = all_ids[num_locations // 2:]
        pickups_deliveries = [list(pair) for pair in zip(pickups, deliveries)]
        distance_matrix = [[0] * num_locations for _ in range(num_locations)]
        for i in range(num_locations):
            for j in range(num_locations):
                if i != j:
                    distance_matrix[i][j] = calculate_distance(locations[i], locations[j])
    return {
        "num_vehicles": num_vehicles,
        "vehicle_capacity": vehicle_capacities,
        "vehicle_capacities": [vehicle_capacities] * num_vehicles,
        "demands": demands,
        "distance_matrix": distance_matrix,
        "pickups_deliveries": pickups_deliveries,
        "locations": locations,
        "num_locations": num_locations,
        "service_time": service_times,
        "time_windows": time_windows,
        "vehicle_max_distance": 10_000,
        "vehicle_max_time": 1_500,
        "vehicle_speed": 5 * 60 / 3.6,
        "depot": 0
    }


def calculate_distance(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    return int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + 0.5)
