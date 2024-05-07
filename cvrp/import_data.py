import os


def read_file(file):
    locations = []
    demands = []
    num_locations = 0
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if len(line.strip()) == 0:
                break
            data = line.split()
            if len(data) == 4:
                location = (float(data[1]), float(data[2]))
                demand = int(data[3]) if data[3] else 0
                locations.append(location)
                demands.append(demand)
                num_locations += 1
    return {"locations": locations, "num_locations": num_locations, "demands": demands}


def process_files(directories):
    total_data = []
    for directory in directories:
        files = os.listdir(directory)
        for file in files:
            if file.endswith(".txt"):
                route = os.path.join(directory, file)
                data = read_file(route)
                total_data.append(data)
    return total_data


path = ["../instances/cvrp_instances"]
data = process_files(path)
print(data)
