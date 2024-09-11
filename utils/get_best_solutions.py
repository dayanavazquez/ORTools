import os
import re
import csv


def extract_info_from_txt(file_path):
    routes_count = 0
    execution_time = 0
    cost = 0

    with open(file_path, 'r') as file:
        content = file.read()
        instance_match = re.search(r'Instance:\s*(\S+)', content)
        objective_match = re.search(r'Objective:\s*(\d+)', content)
        time_match = re.search(r'Execution Time:\s*([\d\.]+)', content)
        load_matches = re.findall(r'Distance of the route:\s*(\d+)', content)
        cost_matches = re.findall(r'Total Distance of all routes:\s*(\d+)', content)
        if load_matches:
            for load in load_matches:
                if load != '0':
                    routes_count += 1
                    if not cost_matches:
                        cost += int(load)
        if instance_match:
            instance = instance_match.group(1)
        if objective_match:
            objective = int(objective_match.group(1))
        if time_match:
            execution_time = float(time_match.group(1))
        if cost_matches:
            cost = int(cost_matches[0])
        if objective != 0:
            if cost != objective and cost != 0:
                objective = cost
        return instance, objective, execution_time, routes_count
    return None, None, None, None


def process_solutions_folder(base_folder):
    results = []
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path):
            for f_name in os.listdir(folder_path):
                f_path = os.path.join(folder_path, f_name)
                if os.path.isdir(f_path):
                    f_name = f_name[len("solutions_"):]
                    if '&' in f_name:
                        heuristic, metaheuristic = map(str.strip, f_name.split('&'))
                    else:
                        heuristic, metaheuristic = map(int, f_name.split('_'))

                    for file_name in os.listdir(f_path):
                        if file_name.endswith('.txt'):
                            file_path = os.path.join(f_path, file_name)
                            instance, objective, execution_time, routes_count = extract_info_from_txt(file_path)
                            if instance and objective is not None:
                                results.append(
                                    (instance, objective, heuristic, metaheuristic, execution_time, routes_count))
    return results


def filter_best_solutions(results):
    best_results = {}

    for instance, objective, heuristic, metaheuristic, execution_time, routes_count in results:
        if instance not in best_results:
            best_results[instance] = [
                (instance, objective, heuristic, metaheuristic, execution_time, routes_count)]
        else:
            best_results[instance].append(
                (instance, objective, heuristic, metaheuristic, execution_time, routes_count))

    final_best_results = {}
    for instance, candidates in best_results.items():
        min_objective = min(candidates, key=lambda x: x[1])[1]
        min_candidates = [c for c in candidates if c[1] == min_objective]
        if len(min_candidates) > 1:
            best_candidate = min(min_candidates, key=lambda x: x[4])
        else:
            best_candidate = min_candidates[0]
        final_best_results[instance] = best_candidate

    return final_best_results


def write_best_solutions(output_file, best_results):
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['Instance', 'Objective', 'Heuristic', 'Metaheuristic', 'Time', 'Routes'])

        for row in best_results.values():
            writer.writerow(tuple(row))


def main():
    base_folder = '../problems/vrptw'
    output_file = '../problems/vrptw/best_solutions_vrppd.txt'

    results = process_solutions_folder(base_folder)
    best_results = filter_best_solutions(results)
    write_best_solutions(output_file, best_results)


if __name__ == "__main__":
    main()
