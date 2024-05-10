import os
import math
import random


###########################
# DATA TSPLIB
###########################

def calculate_distance(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    return int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + 0.5)


def extract_vrp_data(folder_path):
    vrp_data = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".vrp"):
            with open(os.path.join(folder_path, filename), "r") as file:
                lines = file.readlines()
                coordinates = []
                node_coord_section_index = lines.index("NODE_COORD_SECTION\n")
                demand_section_index = lines.index("DEMAND_SECTION\n")

                for line in lines[node_coord_section_index + 1:demand_section_index]:
                    if line.strip() == "-1":
                        break
                    parts = line.split()
                    x_coord = float(parts[1])
                    y_coord = float(parts[2])
                    coordinates.append((x_coord, y_coord))

                distance_matrix = []
                for i, coord1 in enumerate(coordinates):
                    row = []
                    for j, coord2 in enumerate(coordinates):
                        if i == j:
                            row.append(0)
                        else:
                            distance = calculate_distance(coord1, coord2)
                            row.append(distance)
                    distance_matrix.append(row)

                vrp_data[filename] = {
                    "distance_matrix": distance_matrix,
                    "num_vehicles": random.randint(1, 10),
                    "depot": 0
                }

    return vrp_data


path = "../instances/vrp_instances"
vrp_data = extract_vrp_data(path)
print(vrp_data)
