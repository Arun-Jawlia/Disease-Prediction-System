from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42


MODEL_CONFIGS = {
    "logistic_regression": {
        "model": LogisticRegression(
            max_iter=2000
        ),
        "params": {
            "classifier__C": [
                0.01,
                0.1,
                1,
                10,
                100,
            ],
            "classifier__solver": [
                "liblinear",
                "lbfgs",
            ],
        },
    },

    "decision_tree": {
        "model": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),
        "params": {
            "classifier__criterion": [
                "gini",
                "entropy",
                "log_loss",
            ],
            "classifier__max_depth": [
                None,
                3,
                5,
                7,
                10,
            ],
            "classifier__min_samples_split": [
                2,
                5,
                10,
                20,
            ],
            "classifier__min_samples_leaf": [
                1,
                2,
                5,
                10,
            ],
        },
    },

    "random_forest": {
        "model": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "params": {
            "classifier__n_estimators": [
                100,
                200,
                300,
            ],
            "classifier__max_depth": [
                None,
                5,
                10,
                15,
            ],
            "classifier__min_samples_split": [
                2,
                5,
                10,
            ],
            "classifier__min_samples_leaf": [
                1,
                2,
                4,
            ],
            "classifier__max_features": [
                "sqrt",
                "log2",
            ],
        },
    },

    "svm": {
        "model": SVC(
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "params": {
            "classifier__C": [
                0.1,
                1,
                10,
                100,
            ],
            "classifier__kernel": [
                "linear",
                "rbf",
            ],
            "classifier__gamma": [
                "scale",
                "auto",
            ],
        },
    },

    "knn": {
        "model": KNeighborsClassifier(),
        "params": {
            "classifier__n_neighbors": [
                3,
                5,
                7,
                9,
                11,
                15,
            ],
            "classifier__weights": [
                "uniform",
                "distance",
            ],
            "classifier__metric": [
                "euclidean",
                "manhattan",
                "minkowski",
            ],
        },
    },
}