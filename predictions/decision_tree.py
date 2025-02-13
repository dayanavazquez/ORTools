import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import pandas as pd


def get_decision_tree():
    data = {
        "Time": [120, 95, 110, 130, 100],
        "Cost": [500, 450, 470, 520, 480],
        "Vehicles": [3, 2, 3, 4, 2],
        "Method": ["Heuristic A", "Heuristic B", "Metaheuristic A", "Metaheuristic B", "Metaheuristic C"]
    }

    dataset = pd.DataFrame(data)
    x = dataset.drop(columns=["Method"])
    y = dataset["Method"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)

    plt.figure(figsize=(12, 8))
    plot_tree(
        clf,
        feature_names=x.columns,
        class_names=clf.classes_,
        filled=True,
        rounded=True,
        fontsize=12,
        impurity=False,
        proportion=True,
        label='all',
        precision=2
    )

    plt.title('Decision Tree', fontsize=16, fontweight='bold', color='darkblue')
    plt.show()
    return clf


get_decision_tree()
