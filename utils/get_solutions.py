import os
import re
import csv
import pandas as pd


def get_cost(data, routes_count, cost_matches):
    cost = 0
    for load in data:
        if load != '0':
            routes_count += 1
            if not cost_matches:
                cost += int(load)
    return cost, routes_count


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
        other_cost_matches = re.findall(r'Total load of all routes:\s*(\d+)', content)
        load_route_matches = re.findall(r'Load of the route:\s*(\d+)', content)
        if load_matches:
            cost, routes_count = get_cost(load_matches, routes_count, cost_matches)
        if not cost and load_route_matches:
            cost, routes_count = get_cost(load_route_matches, routes_count, cost_matches)
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
        elif other_cost_matches:
            objective = other_cost_matches[0]
        return instance, objective, execution_time, routes_count
    return None, None, None, None


def obtain_technique(f_path, f_name):
    for file_name in os.listdir(f_path):
        file_path = os.path.join(f_path, file_name)
        with open(file_path, 'r') as file:
            for line in file:
                if 'Heuristic:' in line:
                    heuristic = line.split(':')[1].strip()
                elif 'Metaheuristic:' in line:
                    metaheuristic = line.split(':')[1].strip()
        if heuristic == 'None':
            metaheuristic = f_name
        elif metaheuristic == 'None':
            heuristic = f_name
    return heuristic, metaheuristic


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
                    elif f_name.replace('_', '').isdigit():  # Verifica si todos los componentes de f_name son numéricos
                        heuristic, metaheuristic = map(int, f_name.split('_'))
                    else:
                        heuristic, metaheuristic = obtain_technique(f_path, f_name)
                    for file_name in os.listdir(f_path):
                        if file_name.endswith('.txt'):
                            file_path = os.path.join(f_path, file_name)
                            instance, objective, execution_time, routes_count = extract_info_from_txt(file_path)
                            if instance and objective is not None:
                                results.append(
                                    (instance, objective, heuristic, metaheuristic, execution_time, routes_count))
    return results


def filter_best_solutions_per_algorithm(results):
    best_results = {}
    for instance, objective, heuristic, metaheuristic, execution_time, routes_count in results:
        algorithm = heuristic + "and" + metaheuristic
        if instance not in best_results:
            best_results[instance] = {}
        if algorithm not in best_results[instance]:
            best_results[instance][algorithm] = []

        best_results[instance][algorithm].append(
            (objective, execution_time, routes_count, heuristic, metaheuristic)
        )
    final_best_results = []
    for instance, algorithms in best_results.items():
        for algorithm, candidates in algorithms.items():
            min_objective = min(candidates, key=lambda x: x[0])[0]
            min_candidates = [c for c in candidates if c[0] == min_objective]
            if len(min_candidates) > 1:
                best_candidate = min(min_candidates, key=lambda x: x[1])
            else:
                best_candidate = min_candidates[0]
            final_best_results.append(
                (
                instance, best_candidate[0], best_candidate[3], best_candidate[4], best_candidate[1], best_candidate[2])
            )

    return final_best_results


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


def write_solutions(output_file, results, best_solution, is_csv):
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        if not is_csv:
            writer.writerow(['Instance', 'Objective', 'Heuristic', 'Metaheuristic', 'Time', 'Routes'])
            if best_solution:
                results = filter_best_solutions(results).values()
            for row in results:
                writer.writerow(row)
        else:
            if best_solution:
                results = filter_best_solutions_per_algorithm(results)
            data = []
            for row in results:
                instance = row[0].split(".")[0]
                heuristic = row[2]
                metaheuristic = row[3]
                objective = row[1]
                algorithm = heuristic + "&" + metaheuristic
                data.append([instance, objective, heuristic, metaheuristic, algorithm])

            df = pd.DataFrame(data, columns=["Instance", "Objective", "Heuristic", "Metaheuristic", "Algorithm"])
            algorithms = df["Algorithm"].unique()

            # Crear diccionario con instancias como claves
            result_dict = {"Instance": []}
            for algorithm in algorithms:
                result_dict[algorithm] = []

            for instance in df["Instance"].unique():
                result_dict["Instance"].append(instance)
                for algorithm in algorithms:
                    value = df.loc[(df["Algorithm"] == algorithm) & (df["Instance"] == instance), "Objective"]
                    result_dict[algorithm].append(value.iloc[0] if not value.empty else "")

            df_result = pd.DataFrame(result_dict)
            df_result.to_csv(output_file, sep=";", index=False)


def main(best_solution=False, csv=False):
    base_folder = '../problems/solutions/manhattan/solutions_tsp'
    output_file = '../problems/solutions/manhattan/solutions_tsp/best_solutions_tsp_manhattan_per_algorithm.csv'

    results = process_solutions_folder(base_folder)
    write_solutions(output_file, results, best_solution, csv)


if __name__ == "__main__":
    main(best_solution=True, csv=True)
