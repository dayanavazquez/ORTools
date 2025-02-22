import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
import os


def calculate_rankings(data):
    rankings = pd.DataFrame(index=data.index, columns=data.columns)
    for col in data.columns:
        rankings[col] = data[col].rank(method='average', ascending=True)
    rankings['Average Ranking'] = rankings.mean(axis=1)
    return rankings


def rank_algorithms(data):
    rankings = calculate_rankings(data)
    ranked_algorithms = rankings['Average Ranking'].sort_values().index.tolist()
    return ranked_algorithms


def save_rankings_to_txt(ranked_algorithms, output_file):
    with open(output_file, 'w') as file:
        for i, algorithm in enumerate(ranked_algorithms, start=1):
            file.write(f"{i}. {algorithm}\n")


def load_file_and_analyze():
    file_path = "../../problems/solutions/manhattan/solutions_tsp/best_solutions_tsp_manhattan_per_algorithm.csv"
    df = pd.read_csv(file_path, sep=";")
    df.columns = df.columns.str.strip()
    print("Columns of the DataFrame:", df.columns.tolist())
    data = df.set_index('Instance').transpose()
    print("\nTransposed data (rows = algorithms, columns = instances):")
    print(data)
    ranked_algorithms = rank_algorithms(data)
    print("\nRanked algorithms (from best to worst):")
    for i, algorithm in enumerate(ranked_algorithms, start=1):
        print(f"{i}. {algorithm}")

    friedman_data = [data[col].values for col in data.columns]

    result = friedmanchisquare(*friedman_data)
    print("\nFriedman test result:", result)

    output_file = os.path.splitext(file_path)[0] + ".txt"
    save_rankings_to_txt(ranked_algorithms, output_file)
    print(f"\nRanked algorithms saved to: {output_file}")


load_file_and_analyze()
