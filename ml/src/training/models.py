from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42


def get_baseline_models():

    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000
        ),

        "decision_tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),

        "random_forest": RandomForestClassifier(
            random_state=RANDOM_STATE
        ),

        "svm": SVC(
            probability=True,
            random_state=RANDOM_STATE
        ),

        "knn": KNeighborsClassifier(),
    }