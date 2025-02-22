import matplotlib
from utils.utils import get_data_for_predictions
import joblib
import numpy as np
matplotlib.use("TkAgg")
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import pandas as pd
from sklearn.metrics import accuracy_score
import graphviz
from PIL import Image
import plotly.express as px


def get_decision_tree():
    results = get_data_for_predictions()
    for key, data in results.items():
        dataset = pd.DataFrame(data)
        dataset.dropna(subset=["Method"], inplace=True)

        # Calcular el score para determinar el mejor método
        dataset["Score"] = (
                0.5 * dataset["Objective"] +
                0.3 * dataset["Time"] +
                0.2 * dataset["Routes"]
        )

        # Seleccionar el mejor método para cada instancia
        best_methods = dataset.groupby("Instance", group_keys=False).apply(
            lambda x: x.loc[x["Score"].idxmin(), "Method"], include_groups=False
        ).reset_index(drop=True)

        dataset["Method"] = best_methods
        dataset["Method"] = dataset["Method"].astype(str)

        # Dividir los datos en características (x) y etiquetas (y)
        x = dataset.drop(columns=["Method", "Score", "Instance", "Objective", "Time", "Routes"])
        y = dataset["Method"]

        # Dividir en conjuntos de entrenamiento y prueba
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Configurar la búsqueda de hiperparámetros
        param_grid = {
            "max_depth": [5, 10, None],  # Profundidad máxima del árbol
            "min_samples_split": [2, 5, 10],  # Mínimo de muestras para dividir un nodo
            "min_samples_leaf": [1, 2, 4],  # Mínimo de muestras en una hoja
        }

        # Usar GridSearchCV para encontrar los mejores hiperparámetros
        grid_search = GridSearchCV(
            DecisionTreeClassifier(random_state=42),
            param_grid,
            cv=2,
            scoring="accuracy",
            n_jobs=-1,
            verbose=2,
        )
        grid_search.fit(x_train, y_train)
        model = grid_search.best_estimator_

        # Guardar el modelo
        joblib.dump(model, f"decision_tree_model_{key}.pkl")

        # Importancia de características
        importances = model.feature_importances_
        feature_names = x_train.columns
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        print(f"Feature Importances for {key}:")
        print(importance_df)

        # Visualizar importancia de características
        fig = px.bar(importance_df, x="Feature", y="Importance", title=f"Importancia de las características ({key})")
        fig.update_layout(xaxis_title="Características", yaxis_title="Importancia", xaxis_tickangle=-90)
        fig.show()

        # Limpiar los nombres de las clases (reemplazar "&" con "_and_")
        y_train_cleaned = y_train.str.replace("&", "_and_")
        y_test_cleaned = y_test.str.replace("&", "_and_")
        model.classes_ = [cls.replace("&", "_and_") for cls in model.classes_]

        # Exportar y visualizar el árbol de decisión
        export_graphviz(
            model,
            out_file=f"tree_{key}.dot",
            feature_names=x_train.columns,
            class_names=model.classes_,
            filled=True,
            rounded=True,
            special_characters=True,
            max_depth=3  # Limitar la profundidad para una mejor visualización
        )
        with open(f"tree_{key}.dot") as f:
            dot_graph = f.read()
        graph = graphviz.Source(dot_graph)
        graph.render(f"tree_{key}", format="png", cleanup=True)
        img = Image.open(f"tree_{key}.png")
        img.show()


get_decision_tree()