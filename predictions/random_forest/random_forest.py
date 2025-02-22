from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from imblearn.combine import SMOTEENN
from collections import Counter
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import get_data_for_predictions


def save_results_to_txt(filename, accuracy, f1, precision, recall, roc_auc, best_params, cv_scores, classification_rep, importance_df, class_mapping):
    with open(filename, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        f.write(f"Precision: {precision:.4f}\n\n")
        f.write(f"Recall: {recall:.4f}\n\n")
        f.write(f"ROC-AUC: {roc_auc if roc_auc is not None else 'N/A'}\n\n")
        f.write("Best Parameters:\n")
        f.write(f"{best_params}\n\n")
        f.write("Cross-Validation Scores:\n")
        f.write(f"{cv_scores}\n")
        f.write(f"Mean Cross-Validation F1-score: {cv_scores.mean():.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(f"{classification_rep}\n\n")
        f.write("Feature Importances:\n")
        f.write(importance_df.to_string(index=False))
        f.write("\n\nClass Mapping:\n")
        for class_name, class_value in class_mapping.items():
            f.write(f"{class_name}: {class_value}\n")


def get_xgboost():
    results = get_data_for_predictions()

    for key, data in results.items():
        dataset = pd.DataFrame(data)
        dataset.dropna(subset=["Method"], inplace=True)

        dataset["Score"] = (
                0.6 * dataset["Objective"] +
                0.2 * dataset["Time"] +
                0.2 * dataset["Routes"]
        )

        optimal_methods = dataset.groupby("Instance").apply(
            lambda x: x.loc[x["Score"].idxmin(), "Method"], include_groups=False
        ).reset_index(name="Optimal_Method")

        dataset = dataset.merge(optimal_methods, on="Instance")

        x = dataset.drop(columns=["Method", "Score", "Instance", "Objective", "Time", "Routes", "Optimal_Method"])
        y = dataset["Optimal_Method"]

        # Codificar las etiquetas de clase
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Crear un diccionario con el mapeo de clases
        class_mapping = {class_name: class_value for class_value, class_name in enumerate(label_encoder.classes_)}

        class_distribution = Counter(y_encoded)
        print(f"Class distribution for {key}: {class_distribution}")

        if len(set(y_encoded)) < 2:
            print(f"Error: Only one class present in {key}. Skipping dataset.")
            continue

        x = x.values

        x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42)

        # Normalizar las características
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # Eliminar características con baja varianza
        variance_threshold = VarianceThreshold(threshold=0.01)
        x_train = variance_threshold.fit_transform(x_train)
        x_test = variance_threshold.transform(x_test)

        # Usar SMOTEENN para balancear las clases
        smote_enn = SMOTEENN(random_state=42)
        x_train, y_train = smote_enn.fit_resample(x_train, y_train)

        # Verificar y reindexar las etiquetas después de SMOTEENN
        unique_classes = np.unique(y_train)
        if not np.array_equal(unique_classes, np.arange(len(unique_classes))):
            print("Warning: Some classes were removed by SMOTEENN. Reindexing labels.")
            label_encoder_after_smote = LabelEncoder()
            y_train = label_encoder_after_smote.fit_transform(y_train)
            class_mapping = {class_name: class_value for class_value, class_name in enumerate(label_encoder_after_smote.classes_)}

        # Selección de características con SelectFromModel
        selector = SelectFromModel(XGBClassifier(random_state=42), threshold='median')
        x_train_selected = selector.fit_transform(x_train, y_train)
        x_test_selected = selector.transform(x_test)

        # Optimización de hiperparámetros con GridSearchCV
        param_grid = {
            "n_estimators": [100, 300, 500, 700, 1000],
            "max_depth": [3, 5, 7, 10, 15],
            "learning_rate": [0.001, 0.01, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "gamma": [0, 0.1, 0.2, 0.5],
            "reg_alpha": [0, 0.1, 0.5],
            "reg_lambda": [0, 0.1, 0.5],
        }

        grid_search = GridSearchCV(
            XGBClassifier(random_state=42, eval_metric='mlogloss'),
            param_grid,
            cv=5,
            scoring="f1_weighted",
            n_jobs=-1,
            verbose=2
        )

        grid_search.fit(x_train_selected, y_train)
        model = grid_search.best_estimator_

        joblib.dump(model, f"xgboost_model_{key}.pkl")

        best_params = grid_search.best_params_
        print(f"Best parameters: {best_params}")

        y_pred = model.predict(x_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

        # Calcular ROC-AUC solo si todas las clases están presentes en y_test
        if len(np.unique(y_test)) == len(model.classes_):
            roc_auc = roc_auc_score(y_test, model.predict_proba(x_test_selected), multi_class='ovr')
        else:
            print("Warning: Not all classes are present in y_test. Skipping ROC-AUC calculation.")
            roc_auc = None

        print(f"Accuracy for {key}: {accuracy:.4f}")
        print(f"F1 Score for {key}: {f1:.4f}")
        print(f"Precision for {key}: {precision:.4f}")
        print(f"Recall for {key}: {recall:.4f}")
        print(f"ROC-AUC for {key}: {roc_auc if roc_auc is not None else 'N/A'}")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, x, y_encoded, cv=cv, scoring="f1_weighted")
        print(f"Cross-validation scores for {key}: {cv_scores}")
        print(f"Cross-validation mean F1-score for {key}: {cv_scores.mean():.4f}")

        # Decodificar las etiquetas para el reporte de clasificación
        y_test_decoded = label_encoder.inverse_transform(y_test)
        y_pred_decoded = label_encoder.inverse_transform(y_pred)

        classification_rep = classification_report(y_test_decoded, y_pred_decoded, zero_division=0)
        print("Classification Report:")
        print(classification_rep)

        # Importancia de las características
        importances = model.feature_importances_
        feature_names = dataset.drop(
            columns=["Method", "Score", "Instance", "Objective", "Time", "Routes", "Optimal_Method"]).columns.tolist()
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        importance_df = importance_df[importance_df["Importance"] > 0.01]

        print(f"Feature Importances for {key}:")
        print(importance_df)

        # Guardar resultados en un archivo de texto
        save_results_to_txt(
            f"results_{key}.txt",
            accuracy,
            f1,
            precision,
            recall,
            roc_auc,
            best_params,
            cv_scores,
            classification_rep,
            importance_df,
            class_mapping  # Incluir el mapeo de clases
        )

        # Graficar la curva de aprendizaje
        train_sizes, train_scores, test_scores = learning_curve(model, x_train_selected, y_train, cv=5, scoring="f1_weighted")
        plt.plot(train_sizes, train_scores.mean(axis=1), label="Training score")
        plt.plot(train_sizes, test_scores.mean(axis=1), label="Cross-validation score")
        plt.xlabel("Training examples")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.show()


get_xgboost()