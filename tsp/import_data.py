import os
import pprint


###########################
# DATA TSPLIB
###########################

def extract_tsp_data(folder_path):
    tsp_data = {}

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


path = "../instances/tsp_instances"
data = extract_tsp_data(path)
print(data)
