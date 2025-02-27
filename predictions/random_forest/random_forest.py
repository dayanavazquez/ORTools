from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
import pandas as pd
import joblib
from utils.utils import get_data_for_predictions
from sklearn.tree import export_graphviz
import plotly.express as px
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import os
import re
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA

def save_results_to_txt(filename, accuracy, cv_scores, importance_df, best_params, classification_report_str):
    with open(filename, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Cross-Validation Scores:\n")
        f.write(f"{cv_scores}\n")
        f.write(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.4f}\n\n")
        f.write("Best Parameters:\n")
        f.write(f"{best_params}\n\n")
        f.write("Classification Report:\n")
        f.write(f"{classification_report_str}\n\n")
        f.write("Feature Importances:\n")
        f.write(importance_df.to_string(index=False))

def get_random_forest():
    results = get_data_for_predictions()
    for key, data in results.items():
        dataset = pd.DataFrame(data)
        dataset.dropna(subset=["Method"], inplace=True)

        # Feature Engineering: Crear nuevas características
        dataset["Objective_Time_Ratio"] = dataset["Objective"] / (dataset["Time"] + 1e-6)
        dataset["Routes_Time_Ratio"] = dataset["Routes"] / (dataset["Time"] + 1e-6)

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

        class_distribution = Counter(y)
        print(f"Class distribution for {key}: {class_distribution}")

        classes_to_keep = [cls for cls, count in class_distribution.items() if count >= 2]
        if len(classes_to_keep) < len(class_distribution):
            print(f"Warning: Some classes have less than 2 samples in {key}. Removing those classes.")
            mask = y.isin(classes_to_keep)
            x = x[mask]
            y = y[mask]

        if len(y) == 0:
            print(f"Error: Not enough samples to train the model in {key}. Skipping this dataset.")
            continue

        x = x.values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Balanceo de clases con SMOTE, ADASYN y RandomUnderSampler
        smote = SMOTE(random_state=42)
        adasyn = ADASYN(random_state=42)
        rus = RandomUnderSampler(random_state=42)

        # Probar diferentes técnicas de balanceo
        x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)  # SMOTE por defecto
        # x_train_balanced, y_train_balanced = adasyn.fit_resample(x_train, y_train)  # ADASYN
        # x_train_balanced, y_train_balanced = rus.fit_resample(x_train, y_train)  # RandomUnderSampler

        # Feature Selection
        feature_selector = SelectFromModel(RandomForestClassifier(random_state=42))
        x_train_selected = feature_selector.fit_transform(x_train_balanced, y_train_balanced)
        x_test_selected = feature_selector.transform(x_test)

        # Obtener los nombres de las características seleccionadas
        feature_names = dataset.drop(
            columns=["Method", "Score", "Instance", "Objective", "Time", "Routes", "Optimal_Method"]).columns.tolist()
        selected_feature_names = [feature_names[i] for i in feature_selector.get_support(indices=True)]

        # Model Pipeline con Polynomial Features para Feature Engineering
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),  # Añadir interacciones entre características
            ('pca', PCA(n_components=0.95)),  # Reducción de dimensionalidad
            ('clf', RandomForestClassifier(class_weight="balanced", random_state=42))
        ])

        # Hiperparámetros para GridSearchCV (rango más amplio)
        param_grid = {
            "clf__n_estimators": [50, 100, 200, 300, 400, 500],
            "clf__max_depth": [None, 5, 10, 20, 30, 50, 100],
            "clf__min_samples_split": [2, 5, 10, 20],
            "clf__min_samples_leaf": [1, 2, 4, 10],
            "clf__bootstrap": [True, False],
            "clf__max_features": ['sqrt', 'log2', None],
            "clf__criterion": ['gini', 'entropy']
        }

        grid_search = GridSearchCV(
            model,
            param_grid=param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            verbose=2,
            error_score='raise'
        )

        grid_search.fit(x_train_selected, y_train_balanced)
        model = grid_search.best_estimator_

        joblib.dump(model, f"random_forest_model_{key}.pkl")

        y_pred = model.predict(x_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, model.predict_proba(x_test_selected), multi_class='ovr')

        print(f"Accuracy for {key}: {accuracy:.4f}")
        print(f"F1 Score for {key}: {f1:.4f}")
        print(f"ROC-AUC Score for {key}: {roc_auc:.4f}")

        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, x, y, cv=cv, scoring="accuracy")
        print(f"Cross-validation scores for {key}: {cv_scores}")
        print(f"Cross-validation mean accuracy for {key}: {cv_scores.mean():.4f}")

        classification_report_str = classification_report(y_test, y_pred, zero_division=0)
        print("Classification Report:")
        print(classification_report_str)

        # Obtener las importancias de las características
        importances = model.named_steps['clf'].feature_importances_

        # Crear el DataFrame con las características seleccionadas y sus importancias
        importance_df = pd.DataFrame({
            "Feature": selected_feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        # Filtrar características con importancia mayor a 0.01
        importance_df = importance_df[importance_df["Importance"] > 0.01]

        print(f"Feature Importances for {key}:")
        print(importance_df)

        # Guardar resultados en un archivo de texto
        save_results_to_txt(f"results_{key}.txt", accuracy, cv_scores, importance_df, grid_search.best_params_, classification_report_str)

        fig = px.bar(importance_df, x="Feature", y="Importance", title=f"Feature Importances ({key})")
        fig.update_layout(xaxis_title="Features", yaxis_title="Importance", xaxis_tickangle=-90)
        fig.write_image(f"feature_importances_{key}.jpg")

        # Ensemble Methods: Usar VotingClassifier
        log_reg = LogisticRegression(max_iter=1000, random_state=42)
        svm = SVC(probability=True, random_state=42)
        ensemble_model = VotingClassifier(
            estimators=[
                ('rf', model.named_steps['clf']),
                ('lr', log_reg),
                ('svm', svm)
            ],
            voting='soft'
        )

        ensemble_model.fit(x_train_selected, y_train_balanced)
        y_pred_ensemble = ensemble_model.predict(x_test_selected)
        accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
        print(f"Ensemble Model Accuracy for {key}: {accuracy_ensemble:.4f}")

get_random_forest()