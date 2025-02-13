import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def get_association_rules():
    data = {
        "Time": [120, 95, 110, 130, 100],  # Tiempo de ejecución en segundos
        "Cost": [500, 450, 470, 520, 480],  # Costo de la solución
        "Vehicles": [3, 2, 3, 4, 2],  # Cantidad de vehículos usados
        "Method": ["Heuristic A", "Heuristic B", "Metaheuristic A", "Metaheuristic B", "Metaheuristic C"]
        # Métodos usados
    }
    dataset = pd.DataFrame(data)
    dataset["Time_Bin"] = pd.cut(dataset["Time"], bins=3, labels=["Short", "Medium", "High"])
    dataset["Cost_Bin"] = pd.cut(dataset["Cost"], bins=3, labels=["Short", "Medium", "High"])
    dataset["Vehicles_Bin"] = pd.cut(dataset["Vehicles"], bins=2, labels=["Few", "Many"])
    dataset = dataset.drop(columns=["Time", "Cost", "Vehicles"])

    df = pd.get_dummies(dataset)
    frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

    print("Reglas de Asociación Encontradas:")
    for idx, row in rules.iterrows():
        antecedents = ', '.join(list(row['antecedents']))
        consequents = ', '.join(list(row['consequents']))
        support = round(row['support'], 3)
        confidence = round(row['confidence'], 3)
        lift = round(row['lift'], 3)

        print(f"\nRegla {idx + 1}:")
        print(f"Si tienes: {antecedents}")
        print(f"Entonces, es probable que tengas: {consequents}")
        print(f"Con un soporte de {support} (probabilidad de que ambas condiciones ocurran juntas)")
        print(f"Confianza de {confidence} (probabilidad de que el consecuente ocurra dado el antecedente)")
        print(f"Y un Lift de {lift} (indicación de la fuerza de la regla respecto a la independencia)")

    return rules


get_association_rules()
