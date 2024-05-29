import os
import math


###########################
# DATA TSPLIB
###########################

def process_files():
    tsp_data = {}
    folder_path = "../../instances/tsp_instances"
    for filename in os.listdir(folder_path):
        if filename.endswith(".tsp"):
            with open(os.path.join(folder_path, filename), "r") as file:
                lines = file.readlines()
                node_coord_section_index = lines.index("NODE_COORD_SECTION\n")
                locations = []
                for line in lines[node_coord_section_index + 1:]:
                    if line.strip() == "EOF":
                        break
                    parts = line.split()
                    x_coord = float(parts[1])
                    y_coord = float(parts[2])
                    locations.append((x_coord, y_coord))

                tsp_data[filename] = {
                    "locations": locations,
                    "num_vehicles": 1,
                    "depot": 0
                }
    return tsp_data


def calculate_distance(coord1, coord2):
    """Calculates the Euclidean distance between two coordinates."""
    x1, y1 = coord1
    x2, y2 = coord2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)