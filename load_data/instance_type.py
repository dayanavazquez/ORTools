from enum import Enum
import os
from import_data import read_file_tsp, read_file_bh, read_file_tw, read_file_md


class InstanceType(Enum):
    TSP = 'tsp'
    BHCVRP = 'bhcvrp'
    MDCVRP = 'mdcvrp'
    CVRPTW = 'cvrptw'


def process_files(instance_type: InstanceType):
    problem_data = {
        InstanceType.TSP: {
            'read_function': read_file_tsp,
            'path': ['../../../instances/tsp_instances']
        },
        InstanceType.BHCVRP: {
            'read_function': read_file_bh,
            'path': ['../../../instances/bhcvrp_instances']
        },
        InstanceType.MDCVRP: {
            'read_function': read_file_md,
            'path': ['../../../instances/mdcvrp_instances(Nanda)/C-mdvrp']
        },
        InstanceType.CVRPTW: {
            'read_function': read_file_tw,
            'path': ['../../../instances/cvrptw_instances']
        }
    }
    total_data = {}
    read_function = problem_data[instance_type]['read_function']
    for directory in problem_data[instance_type]['path']:
        files = os.listdir(directory)
        for file in files:
            if file.endswith('.txt'):
                route = os.path.join(directory, file)
                data = read_function(route)
                total_data[file] = data
    return total_data
