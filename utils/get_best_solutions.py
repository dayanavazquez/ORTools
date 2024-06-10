import os
import re
import csv


def extract_info_from_txt(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        instance_match = re.search(r'Instance:\s*(\S+)', content)
        objective_match = re.search(r'Objective:\s*(\d+)', content)

        if instance_match and objective_match:
            instance = instance_match.group(1)
            objective = int(objective_match.group(1))
            return instance, objective
    return None, None


def process_solutions_folder(base_folder):
    results = []

    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)

        if os.path.isdir(folder_path):
            heuristic, metaheuristic = map(int, folder_name.split('_')[1:])

            for file_name in os.listdir(folder_path):
                if file_name.endswith('.txt'):
                    file_path = os.path.join(folder_path, file_name)
                    instance, objective = extract_info_from_txt(file_path)

                    if instance and objective:
                        results.append((instance, objective, heuristic, metaheuristic))

    return results


def filter_best_objectives(results):
    best_results = {}

    for instance, objective, heuristic, metaheuristic in results:
        if instance not in best_results:
            best_results[instance] = [(instance, objective, heuristic, metaheuristic)]
        else:
            current_best_objective = best_results[instance][0][1]
            if objective < current_best_objective:
                best_results[instance] = [(instance, objective, heuristic, metaheuristic)]
            elif objective == current_best_objective:
                best_results[instance].append((instance, objective, heuristic, metaheuristic))

    return best_results


def write_best_solutions(output_file, best_results):
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['Instance', 'Objective', 'Heuristic', 'Metaheuristic'])

        for rows in best_results.values():
            for row in rows:
                writer.writerow(row)


def main():
    base_folder = 'solutions_vrppd'
    output_file = '../../vrp/best_solutions_vrppd.txt'

    results = process_solutions_folder(base_folder)
    best_results = filter_best_objectives(results)
    write_best_solutions(output_file, best_results)


if __name__ == "__main__":
    main()
