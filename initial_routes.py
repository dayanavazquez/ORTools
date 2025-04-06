import matplotlib
from utils.utils import get_data_for_predictions
import joblib
import numpy as np
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
import graphviz


def get_random_forest():
    data = get_data_for_predictions()
    dataset = pd.DataFrame(data)
    dataset.dropna(subset=["Method"], inplace=True)

    dataset = pd.get_dummies(dataset, columns=["Problem"])

    dataset["Score"] = (
            0.5 * dataset["Objective"] +
            0.3 * dataset["Time"] +
            0.2 * dataset["Routes"]
    )

    best_methods = dataset.groupby("Instance", group_keys=False).apply(
        lambda x: x.loc[x["Score"].idxmin(), "Method"], include_groups=False
    ).reset_index(drop=True)

    dataset["Method"] = best_methods

    dataset["Method"] = dataset["Method"].astype(str)

    x = dataset.drop(columns=["Method", "Score", "Instance", "Objective", "Time", "Routes"])
    y = dataset["Method"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    param_grid = {
        "n_estimators": [500],
        "max_depth": [5],
        "min_samples_split": [10],
        "min_samples_leaf": [1],
        "bootstrap": [True],
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=2,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2,
    )

    grid_search.fit(x_train, y_train)

    model = grid_search.best_estimator_

    importances = model.feature_importances_
    feature_names = x_train.columns

    print("Feature Importances:")
    for feature, importance in zip(feature_names, importances):
        print(f"{feature}: {importance:.4f}")

    y_pred = model.predict(x_test)
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    scores = cross_val_score(model, x, y, cv=2, scoring="accuracy")
    print(f"Cross Val Score: {scores.mean() * 100:.2f}%")

    # Guardar el modelo
    joblib.dump(model, "random_forest_model.pkl")

    # Limpiar los nombres de las clases
    y_train_cleaned = y_train.str.replace("&", "_and_")
    y_test_cleaned = y_test.str.replace("&", "_and_")
    model.classes_ = [cls.replace("&", "_and_") for cls in model.classes_]

    # Contar el número de ejemplos por clase
    class_counts = y_train_cleaned.value_counts()

    # Filtrar clases con menos de 2 ejemplos
    valid_classes = class_counts[class_counts >= 2].index

    # Restablecer índices para asegurar la alineación
    x_train = x_train.reset_index(drop=True)
    y_train_cleaned = y_train_cleaned.reset_index(drop=True)

    # Crear una máscara booleana para filtrar
    mask = y_train_cleaned.isin(valid_classes)

    # Aplicar la máscara a ambos x_train y y_train_cleaned
    x_train = x_train[mask]
    y_train_cleaned = y_train_cleaned[mask]

    # Iterar sobre todos los árboles del bosque
    for i, estimator in enumerate(model.estimators_):
        # Exportar el árbol a un archivo .dot
        export_graphviz(
            estimator,
            out_file=f"tree_{i}.dot",
            feature_names=x_train.columns,
            class_names=model.classes_,
            filled=True,
            rounded=True,
            special_characters=True,
            max_depth=3,  # Limitar la profundidad para facilitar la visualización
            proportion=True,  # Mostrar proporciones en lugar de conteos
        )

        # Convertir el archivo .dot a una imagen
        with open(f"tree_{i}.dot") as f:
            dot_graph = f.read()

        graph = graphviz.Source(dot_graph)

        # Guardar el gráfico como PNG
        graph.render(f"tree_{i}", format="png")
        print(f"Árbol {i} guardado como tree_{i}.png")

    # Visualizar la importancia de las características
    importances = model.feature_importances_
    feature_names = x_train.columns
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Importancia de las características")
    plt.bar(range(x_train.shape[1]), importances[indices], align="center")
    plt.xticks(range(x_train.shape[1]), feature_names[indices], rotation=90)
    plt.xlabel("Características")
    plt.ylabel("Importancia")
    plt.tight_layout()
    plt.show()

get_random_forest()