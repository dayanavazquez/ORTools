import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from distances.distance_type import DistanceType
from load_data.instance_type import InstanceType
from utils.utils import get_data_for_predictions
from problems.problem_type import ProblemType


def save_rules_to_txt(rules, file_path="association_rules.txt"):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write("Association rules:\n")

        for idx, row in rules.iterrows():
            antecedents = ', '.join(list(row['antecedents']))
            consequents = ', '.join(list(row['consequents']))
            support = round(row['support'], 3)
            confidence = round(row['confidence'], 3)
            lift = round(row['lift'], 3)

            file.write(f"\nRegla {idx + 1}:\n")
            file.write(f"Si tienes: {antecedents}\n")
            file.write(f"Entonces, es probable que tengas: {consequents}\n")
            file.write(f"Con un soporte de {support} (probabilidad de que ambas condiciones ocurran juntas)\n")
            file.write(f"Confianza de {confidence} (probabilidad de que el consecuente ocurra dado el antecedente)\n")
            file.write(f"Y un Lift de {lift} (indicación de la fuerza de la regla respecto a la independencia)\n")


def get_association_rules():
    data = get_data_for_predictions(ProblemType.DVRP, InstanceType.BHCVRP, DistanceType.EUCLIDEAN, None, None, None,
                                    ['../instances/bhcvrp_instances'])
    dataset = pd.DataFrame(data)
    dataset["Time_Bin"] = pd.cut(dataset["Time"], bins=3, labels=["Short", "Medium", "High"])
    dataset["Objective_Bin"] = pd.cut(dataset["Objective"], bins=3, labels=["Short", "Medium", "High"])
    dataset["Routes_Bin"] = pd.cut(dataset["Routes"], bins=2, labels=["Few", "Many"])
    dataset["Vehicles_Bin"] = pd.cut(dataset["Vehicles"], bins=2, labels=["Few", "Many"])
    dataset["Vehicles Capacity_Bin"] = pd.cut(dataset["Vehicles Capacity"], bins=2, labels=["Low", "High"])
    dataset["Demands_Bin"] = pd.cut(dataset["Demands"], bins=2, labels=["Few", "Many"])
    dataset["Nodes_Bin"] = pd.cut(dataset["Nodes"], bins=2, labels=["Few", "Many"])
    dataset = dataset.drop(columns=["Time", "Objective", "Routes", "Vehicles", "Vehicles Capacity", "Demands", "Nodes"])
    df = pd.get_dummies(dataset)
    df = df.astype(bool)
    frequent_itemsets = apriori(df, min_support=0.08, use_colnames=True)
    if frequent_itemsets.empty:
        print("No se encontraron conjuntos frecuentes con el soporte especificado.")
        return
    try:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    except ZeroDivisionError:
        print("Error: Se produjo una división por cero en la generación de reglas de asociación.")
        return
    rules.replace([np.inf, -np.inf], np.nan, inplace=True)
    rules.dropna(subset=['confidence', 'lift'], inplace=True)
    if rules.empty:
        print("No se encontraron reglas válidas después de eliminar valores NaN e infinitos.")
        return
    filtered_rules = rules[rules['antecedents'].apply(lambda x: 'Method' in str(x)) |
                           rules['consequents'].apply(lambda x: 'Method' in str(x))]
    if filtered_rules.empty:
        print("No se encontraron reglas de asociación relevantes con 'Method'.")
        return
    save_rules_to_txt(filtered_rules, f"./association_rules/association_rules_dvrp.txt")


get_association_rules()
