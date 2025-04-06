#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, \
    roc_auc_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.preprocessing import PolynomialFeatures
from imblearn.over_sampling import ADASYN, SMOTE
import pandas as pd
import joblib
import numpy as np
from utils.utils import get_data_for_predictions
from sklearn.tree import export_graphviz
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import randint
import seaborn as sns
import pydotplus
from io import StringIO


def save_feature_importance_as_jpg(importance_df, key):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', hue='Feature', legend=False)
    plt.title(f'Feature Importances - {key}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(f"feature_importance_{key}.jpg")
    plt.close()


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
        f.write(f"Mean Cross-Validation F1-score: {cv_scores.mean():.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(f"{classification_rep}\n\n")
        f.write("Feature Importances:\n")
        f.write(importance_df.to_string(index=False))


def get_random_forest():
    results = get_data_for_predictions()
    print("aqui//////////////////////////////////////////")
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

        feature_names = dataset.drop(
            columns=["Method", "Score", "Instance", "Objective", "Time", "Routes", "Optimal_Method"]
        ).columns.tolist()

        x = dataset[feature_names].values
        y = dataset["Optimal_Method"]

        class_distribution = Counter(y)
        print(f"Class distribution for {key}: {class_distribution}")

        min_samples_per_class = 10
        classes_to_keep = [cls for cls, count in class_distribution.items() if count >= min_samples_per_class]
        if len(classes_to_keep) < len(class_distribution):
            print(f"Warning: Removing classes with less than {min_samples_per_class} samples in {key}.")
            mask = y.isin(classes_to_keep)
            x = x[mask]
            y = y[mask]

        if len(y) == 0:
            print(f"Error: Not enough samples to train the model in {key}. Skipping.")
            continue

        poly = PolynomialFeatures(degree=1, interaction_only=True, include_bias=False)
        x_poly = poly.fit_transform(x)
        feature_names_poly = poly.get_feature_names_out(feature_names)

        x_train, x_test, y_train, y_test = train_test_split(x_poly, y, test_size=0.4, random_state=42)

        try:
            sampler = ADASYN(random_state=42)
            x_train_res, y_train_res = sampler.fit_resample(x_train, y_train)
        except RuntimeError as e:
            print(f"ADASYN failed: {e}. Using SMOTE instead.")
            sampler = SMOTE(random_state=42, k_neighbors=min(3, len(np.unique(y_train)) - 1))
            x_train_res, y_train_res = sampler.fit_resample(x_train, y_train)

        selector_constant = VarianceThreshold()
        x_train_res = selector_constant.fit_transform(x_train_res)
        x_test = selector_constant.transform(x_test)
        feature_names_poly = [name for i, name in enumerate(feature_names_poly)
                              if selector_constant.get_support()[i]]

        selector = SelectKBest(f_classif, k=min(10, len(feature_names_poly)))
        x_train_selected = selector.fit_transform(x_train_res, y_train_res)
        x_test_selected = selector.transform(x_test)

        selected_feature_names = [feature_names_poly[i] for i in selector.get_support(indices=True)]

        param_dist = {
            "n_estimators": randint(50, 200),
            "max_depth": [3, 5, 7, 10],
            "min_samples_leaf": randint(5, 20),
            "min_samples_split": randint(10, 50),
            "bootstrap": [True],
            "max_features": ['sqrt', 0.3, 0.5],
            "criterion": ['gini'],
            "class_weight": ['balanced', None],
            "ccp_alpha": [0.0, 0.01, 0.05]
        }

        random_search = RandomizedSearchCV(
            RandomForestClassifier(random_state=42),
            param_distributions=param_dist,
            n_iter=50,
            cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
            scoring='balanced_accuracy',
            n_jobs=-1,
            verbose=2,
            random_state=42,
            error_score='raise'
        )
        random_search.fit(x_train_selected, y_train_res)

        best_params = random_search.best_params_
        param_grid = {
            'max_depth': [max(5, best_params['max_depth'] - 5), best_params['max_depth'],
                          best_params['max_depth'] + 5],
            'min_samples_split': [max(2, best_params['min_samples_split'] - 2),
                                  best_params['min_samples_split'],
                                  best_params['min_samples_split'] + 2],
            'ccp_alpha': [max(0, best_params['ccp_alpha'] - 0.005),
                          best_params['ccp_alpha'],
                          min(0.1, best_params['ccp_alpha'] + 0.005)]
        }

        grid_search = GridSearchCV(
            random_search.best_estimator_,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
            scoring='balanced_accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(x_train_selected, y_train_res)
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        final_model = BaggingClassifier(
            estimator=model,
            n_estimators=30,
            max_samples=0.5,
            max_features=0.5,
            random_state=42,
            n_jobs=-1
        )
        final_model.fit(x_train_selected, y_train_res)

        joblib.dump(final_model, f"random_forest_model_{key}.pkl")

        y_pred = final_model.predict(x_test_selected)

        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(
            y_test,
            final_model.predict_proba(x_test_selected),
            multi_class='ovr'
        ) if len(np.unique(y_test)) > 2 else roc_auc_score(y_test, final_model.predict_proba(x_test_selected)[:, 1])

        print(f"\nPerformance for {key}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"Cohen's Kappa: {kappa:.4f}")
        print(f"F1 Score: {f1:.4f}")

        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = cross_val_score(final_model, x_train_selected, y_train_res, cv=cv, scoring='balanced_accuracy')
        print(f"Mean CV Balanced Accuracy: {cv_scores.mean():.4f}")

        classification_rep = classification_report(y_test, y_pred, zero_division=0)
        print("\nClassification Report:")
        print(classification_rep)

        importances = model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": selected_feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        importance_df = importance_df[importance_df["Importance"] > 0.01]
        print("\nFeature Importances:")
        print(importance_df)

        save_feature_importance_as_jpg(importance_df, key)

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - {key}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(f"confusion_matrix_{key}.png")
        plt.close()

        importances_all = [tree.feature_importances_.sum() for tree in model.estimators_]
        best_tree_index = np.argmax(importances_all)
        best_tree = model.estimators_[best_tree_index]

        dot_data = StringIO()
        export_graphviz(
            best_tree,
            out_file=dot_data,
            feature_names=selected_feature_names,
            class_names=np.unique(y_train_res).astype(str),
            filled=True,
            rounded=True,
            special_characters=True
        )
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_png(f"best_tree_{key}.png")

        save_results_to_txt(f"results_{key}.txt", accuracy, balanced_acc, kappa, f1, precision, recall, roc_auc,
                            best_params, cv_scores, classification_rep, importance_df)


get_random_forest()
