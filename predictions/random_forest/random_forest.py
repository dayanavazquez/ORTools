from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score, \
    roc_auc_score, accuracy_score, classification_report
import pandas as pd
import joblib
from sklearn.preprocessing import label_binarize
from utils.utils import get_data_for_predictions
from sklearn.tree import export_graphviz
from PIL import Image
import plotly.express as px
from imblearn.over_sampling import SMOTE
from collections import Counter
from scipy.stats import randint
import os
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import re


def save_results_to_txt(filename, accuracy, balanced_acc, kappa, f1, precision, recall, roc_auc,
                        best_params, cv_scores, classification_rep, importance_df):
    with open(filename, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(f"Balanced Accuracy: {balanced_acc:.4f}\n\n")
        f.write(f"Cohen's Kappa: {kappa:.4f}\n\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        f.write(f"Precision: {precision:.4f}\n\n")
        f.write(f"Recall: {recall:.4f}\n\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n\n")
        f.write("Best Parameters:\n")
        f.write(f"{best_params}\n\n")
        f.write("Cross-Validation Scores:\n")
        f.write(f"{cv_scores}\n")
        f.write(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(f"{classification_rep}\n\n")
        f.write("Feature Importances:\n")
        f.write(importance_df.to_string(index=False))


def get_random_forest():
    results = get_data_for_predictions()
    for key, data in results.items():
        dataset = pd.DataFrame(data)
        dataset.dropna(subset=["Method"], inplace=True)

        # Cálculo del Score: se ajusta según el problema
        dataset["Score"] = (
                0.6 * dataset["Objective"] +
                0.2 * dataset["Time"] +
                0.2 * dataset["Routes"]
        ) if key != "TSP" else (
                0.7 * dataset["Objective"] +
                0.3 * dataset["Time"]
        )

        # Seleccionar el método óptimo por cada instancia
        optimal_methods = dataset.groupby("Instance").apply(
            lambda x: x.loc[x["Score"].idxmin(), "Method"], include_groups=False
        ).reset_index(name="Optimal_Method")

        dataset = dataset.merge(optimal_methods, on="Instance")

        # Definir las columnas a eliminar
        columns_set = (
            ["Method", "Score", "Instance", "Objective", "Time", "Routes", "Optimal_Method"]
            if key != "TSP"
            else ["Method", "Score", "Instance", "Objective", "Time", "Optimal_Method"]
        )

        # Separar características y target
        x = dataset.drop(columns=columns_set)
        y = dataset["Optimal_Method"]

        # Identificar columnas numéricas y la categórica principal "Distance"
        numeric_features = x.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = ['Distance']  # Columna categórica

        # Preprocesamiento: aplicar OneHotEncoder sobre la variable categórica y pasar las numéricas
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )

        x_processed = preprocessor.fit_transform(x)

        # Obtener los nombres de las columnas para la variable Distance
        encoder = preprocessor.named_transformers_['cat']
        distance_categories = encoder.categories_[0]
        distance_feature_names = [f"Distance_{cat}" for cat in distance_categories]

        numeric_feature_names = numeric_features.copy()
        feature_names = numeric_feature_names + distance_feature_names

        # Reportar la distribución de clases
        class_distribution = Counter(y)
        print(f"Class distribution for {key}: {class_distribution}")

        # Filtrado de clases con muy pocas muestras
        classes_to_keep = [cls for cls, count in class_distribution.items() if count >= 2]
        if len(classes_to_keep) < len(class_distribution):
            print(f"Warning: Some classes have less than 2 samples in {key}. Removing those classes.")
            mask = y.isin(classes_to_keep)
            x_processed = x_processed[mask]
            y = y[mask]

        if len(y) == 0:
            print(f"Error: Not enough samples to train the model in {key}. Skipping this dataset.")
            continue

        # División de datos en entrenamiento y test
        x_train, x_test, y_train, y_test = train_test_split(
            x_processed, y, test_size=0.2, random_state=42, stratify=y
        )

        # Balanceo de clases: SMOTE para aumentar la representatividad en el entrenamiento
        smote = SMOTE(random_state=42)
        x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

        # Selección de características: usar SelectKBest
        selector = SelectKBest(f_classif, k=min(10, x_train_smote.shape[1]))
        x_train_selected = selector.fit_transform(x_train_smote, y_train_smote)
        x_test_selected = selector.transform(x_test)

        selected_feature_indices = selector.get_support(indices=True)
        selected_feature_names = [feature_names[i] for i in selected_feature_indices]

        # *** Ajuste de hiperparámetros para evitar sobreajuste y mejorar precisión ***
        # Se restringe la complejidad (max_depth, min_samples_leaf, etc.)
        param_dist = {
            "n_estimators": randint(200, 800),  # Aseguramos un número suficiente de árboles, pero no excesivo
            "max_depth": [5, 10, 15, 20, 25],  # Limitar la profundidad para prevenir sobreajuste
            "min_samples_split": randint(5, 20),  # Incrementar el mínimo de muestras para dividir un nodo
            "min_samples_leaf": randint(2, 10),  # Incrementar el mínimo de muestras en cada hoja
            "bootstrap": [True],  # Usar bootstrapping para estabilizar el modelo
            "max_features": ['sqrt', 'log2'],  # Mantener la selección de características
            "criterion": ['gini', 'entropy']
        }

        # Se utiliza RandomizedSearchCV con validación cruzada de 5 pliegues
        random_search = RandomizedSearchCV(
            RandomForestClassifier(class_weight="balanced", random_state=42),
            param_distributions=param_dist,
            n_iter=100,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            verbose=2,
            random_state=42
        )

        random_search.fit(x_train_selected, y_train_smote)
        model = random_search.best_estimator_

        joblib.dump(model, f"random_forest_model_{key}.pkl")

        # Luego de ajustar el modelo con validación, se evaluará en el conjunto de test
        y_pred = model.predict(x_test_selected)

        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

        # Para ROC-AUC en clasificación multiclase necesitas usar `label_binarize`
        classes = model.classes_
        y_test_bin = label_binarize(y_test, classes=classes)
        y_pred_bin = label_binarize(y_pred, classes=classes)

        roc_auc = roc_auc_score(y_test_bin, y_pred_bin, average="macro", multi_class="ovo")

        classification_rep = classification_report(y_test, y_pred, zero_division=0)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for {key}: {accuracy:.4f}")

        # Validación cruzada
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, x_processed, y, cv=cv, scoring="accuracy")
        print(f"Cross-validation scores for {key}: {cv_scores}")
        print(f"Cross-validation mean accuracy for {key}: {cv_scores.mean():.4f}")

        # Reporte de clasificación
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Importancia de características
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": selected_feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        # Identifica si una característica es original o derivada de una categórica
        importance_df = importance_df[~importance_df["Feature"].str.startswith("Distance_")]

        importance_df = importance_df[importance_df["Importance"] > 0.01]

        print(f"Feature Importances for {key}:")
        print(importance_df)

        save_results_to_txt(
            f"results_{key}.txt",
            accuracy, balanced_acc, kappa, f1, precision, recall, roc_auc,
            random_search.best_params_,
            cv_scores, classification_rep, importance_df
        )

        # Visualización de importancia de características
        fig = px.bar(importance_df, x="Feature", y="Importance", title=f"Feature Importances ({key})")
        fig.update_layout(xaxis_title="Features", yaxis_title="Importance", xaxis_tickangle=-90)
        fig.write_image(f"feature_importances_{key}.jpg")

        try:
            estimator = model.estimators_[0]
            dot_data = export_graphviz(
                estimator,
                out_file=None,
                feature_names=selected_feature_names,
                class_names=model.classes_,
                filled=True,
                rounded=True,
                special_characters=True,
                impurity=False,
                proportion=True,
                precision=0,
                node_ids=True,
                rotate=False,
            )
            import html
            dot_data = html.unescape(dot_data)

            # Patrón para capturar "Distance_<tipo> ≤ <valor>"
            distance_pattern = re.compile(
                r"Distance_([a-zA-Z]+)\s*(?:&le;|<=|≤|&#8804;|&amp;le;)\s*([-+]?\d*\.?\d+)",
                re.IGNORECASE
            )

            def replace_distance_condition(match):
                try:
                    category = match.group(1).lower()  # Ej: "euclidean"
                    threshold = float(match.group(2))
                    print(f"Coincidencia encontrada: {match.group(0)} -> categoría: {category}, threshold: {threshold}")
                    if threshold < 0.5:
                        replacement = f"Distance != '{category}'"
                    else:
                        replacement = f"Distance == '{category}'"
                    print(f"Reemplazo: {replacement}")
                    return replacement
                except Exception as err:
                    print(f"Error al procesar la coincidencia {match.group(0)}: {err}")
                    return match.group(0)

            matches = list(distance_pattern.finditer(dot_data))
            print(f"{len(matches)} coincidencia(s) encontradas:")
            for m in matches:
                print(f" - {m.group(0)}")

            dot_data_converted = distance_pattern.sub(replace_distance_condition, dot_data)

            # Guardar el archivo DOT
            dot_file = f"tree_{key}.dot"
            with open(dot_file, "w") as f:
                f.write(dot_data_converted)

            png_file = f"tree_{key}.png"
            os.system(f"dot -Tpng {dot_file} -o {png_file}")

            if os.path.exists(png_file):
                print(f"Tree visualization saved as {png_file}")
                img = Image.open(png_file)
                img.show()
            else:
                print(f"Error: Failed to generate {png_file}")
        except Exception as e:
            print(f"Error visualizing the tree for {key}: {e}")


if __name__ == "__main__":
    get_random_forest()
