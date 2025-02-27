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


def filter_average_solutions_per_algorithm(results):
    average_results = {}
    for instance, objective, heuristic, metaheuristic, execution_time, routes_count in results:
        algorithm = heuristic.strip() + "and" + metaheuristic.strip()
        if instance not in average_results:
            average_results[instance] = {}
        if algorithm not in average_results[instance]:
            average_results[instance][algorithm] = {
                "objectives": [],
                "execution_times": [],
                "routes_counts": [],
            }
        average_results[instance][algorithm]["objectives"].append(objective)
        average_results[instance][algorithm]["execution_times"].append(execution_time)
        average_results[instance][algorithm]["routes_counts"].append(routes_count)
    final_average_results = []
    for instance, algorithms in average_results.items():
        for algorithm, data in algorithms.items():
            avg_objective = sum(data["objectives"]) / len(data["objectives"])
            avg_execution_time = sum(data["execution_times"]) / len(data["execution_times"])
            avg_routes_count = sum(data["routes_counts"]) / len(data["routes_counts"])
            final_average_results.append(
                (
                    instance,
                    avg_objective,
                    algorithm.split("and")[0],
                    algorithm.split("and")[1],
                    algorithm,
                    avg_execution_time,
                    avg_routes_count,
                )
            )
    return final_average_results


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


def generate_csv_files(results, best_solution, output_folder, filtered=None):
    if best_solution:
        results = filter_average_solutions_per_algorithm(results)
    data = []
    for row in results:
        instance = row[0].split(".")[0]
        heuristic = row[2]
        metaheuristic = row[3]
        objective = row[1]
        execution_time = row[5]
        algorithm = row[4]
        data.append([instance, objective, execution_time, heuristic, metaheuristic, algorithm])

    df = pd.DataFrame(
        data,
        columns=["Instance", "Objective", "Time", "Heuristic", "Metaheuristic", "Algorithm"]
    )
    if not filtered:
        metaheuristics = df["Metaheuristic"].unique()
        for metaheuristic in metaheuristics:
            df_meta = df[df["Metaheuristic"] == metaheuristic]
            objective_dict = {"Instance": []}
            time_dict = {"Instance": []}
            algorithms = df_meta["Algorithm"].unique()
            for algorithm in algorithms:
                objective_dict[algorithm] = []
                time_dict[algorithm] = []
            for instance in df_meta["Instance"].unique():
                objective_dict["Instance"].append(instance)
                time_dict["Instance"].append(instance)
                for algorithm in algorithms:
                    objective_value = df_meta.loc[
                        (df_meta["Algorithm"] == algorithm) & (df_meta["Instance"] == instance), "Objective"]
                    objective_dict[algorithm].append(objective_value.iloc[0] if not objective_value.empty else "")
                    time_value = df_meta.loc[
                        (df_meta["Algorithm"] == algorithm) & (df_meta["Instance"] == instance), "Time"]
                    time_dict[algorithm].append(time_value.iloc[0] if not time_value.empty else "")
            df_objective = pd.DataFrame(objective_dict)
            df_time = pd.DataFrame(time_dict)
            objective_file = f"{output_folder}/{metaheuristic}_objective.csv"
            time_file = f"{output_folder}/{metaheuristic}_time.csv"
            df_objective.to_csv(objective_file, sep=";", index=False)
            df_time.to_csv(time_file, sep=";", index=False)
            print(f"Generated files: {objective_file}, {time_file}")
    else:
        df_filtered = df[df["Algorithm"].isin(filtered)]
        result_dict = {"Instance": []}
        for algorithm in filtered:
            result_dict[algorithm] = []
        for instance in df_filtered["Instance"].unique():
            result_dict["Instance"].append(instance)
            for algorithm in filtered:
                value = df_filtered.loc[
                    (df_filtered["Algorithm"] == algorithm) & (df_filtered["Instance"] == instance), "Objective"]
                result_dict[algorithm].append(value.iloc[0] if not value.empty else "")
        df_result = pd.DataFrame(result_dict)
        file = f"{output_folder}/best_algorithms_objective.csv"
        df_result.to_csv(file, sep=";", index=False)
        print(f"Archivo generado: {file}")


def write_solutions(output_file, results, best_solution, is_csv, filtered):
    if not is_csv:
        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(['Instance', 'Objective', 'Heuristic', 'Metaheuristic', 'Time', 'Routes'])
            if best_solution:
                results = filter_best_solutions(results).values()
            for row in results:
                writer.writerow(row)
    else:
        generate_csv_files(results, best_solution, output_file, filtered)


def main(best_solution=False, csv=False, filtered=None):
    base_folder = '../problems/solutions/manhattan/solutions_cvrp'
    output_file = '../friedman/manhattan/cvrp'

    results = process_solutions_folder(base_folder)
    write_solutions(output_file, results, best_solution, csv, filtered)


if __name__ == "__main__":
    main(
        best_solution=True,
        csv=True,
        filtered=[
            "PATH_CHEAPEST_ARC_and_TABU_SEARCH",
            "PATH_CHEAPEST_ARC_and_SIMULATED_ANNEALING",
            "PATH_CHEAPEST_ARC_and_GUIDED_LOCAL_SEARCH",
            "SAVINGS_and_GREEDY_DESCENT",
            "SEQUENTIAL_CHEAPEST_INSERTION_and_GREEDY_DESCENT",
            "LOCAL_CHEAPEST_ARC_and_GENERIC_TABU_SEARCH",
            "BEST_INSERTION_and_GENERIC_TABU_SEARCH",
            "PATH_MOST_CONSTRAINED_ARC_and_GENERIC_TABU_SEARCH"
        ]
    )
