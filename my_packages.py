import json
import numpy as np
import inflect


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
import xgboost as xgb

def encoder(dataframe):
    mappings = {}
    for col in dataframe.columns:
        if dataframe[col].dtype == 'object':
            unique_vals = dataframe[col].dropna().unique()
            col_mapping = {val: idx for idx, val in enumerate(unique_vals)}
            dataframe[col] = dataframe[col].map(col_mapping)
            mappings[col] = col_mapping
    return dataframe, mappings


def show(py_dict):
    print(json.dumps(py_dict, indent=4))

# Define the models
def get_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Support Vector Classifier": SVC(),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "LightGBM": lgb.LGBMClassifier(),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "KNeighbors": KNeighborsClassifier(),
        "Bagging": BaggingClassifier()
    }
    return models


# Define the best models
def get_best_models():
    best_models = {
        "Logistic Regression": LogisticRegression(
            penalty='l2',
            dual=False,
            C=0.9,
            fit_intercept=True,
            intercept_scaling=1.0,
            random_state=1000,
            solver="liblinear",
            max_iter=1000,
            multi_class="ovr",
            verbose=1,
            warm_start=False,
        ),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(
            warm_start=True,
            verbose=100,
            random_state=0,
            n_estimators=1000,
            min_weight_fraction_leaf=0.0,
            min_samples_split=10,
            min_samples_leaf=1,
            max_samples=None,
            max_leaf_nodes=None,
            max_features='log2',
            max_depth=10,
            criterion='entropy',
            bootstrap=False
        ),
        "Support Vector Classifier": SVC(),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "LightGBM": lgb.LGBMClassifier(),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "KNeighbors": KNeighborsClassifier(),
        "Bagging": BaggingClassifier()
    }
    return best_models


# Function to print evaluation metrics
def print_evaluation_metrics(y_true, y_pred, model_name):
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("-" * 80)


# run the models
def run_the_models(models, X_train, y_train, X_true, y_true):
    model_accuracies = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_true)
        accuracy = accuracy_score(y_true, y_pred)
        print_evaluation_metrics(y_true, y_pred, name)
        print(f"Accuracy for {name}: {accuracy}\n\n")
        model_accuracies[name] = accuracy
    print("All the accuracies:")
    print("*" * 100)
    print("*" * 100)
    show(model_accuracies)
    return model_accuracies


def kFold_cross_validation(kf, models, X, y):
    model_accuracies = {}
    for name, model in models.items():
        accuracies = []
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))
            print_evaluation_metrics(y_test, y_pred)
        average_accuracy = np.mean(accuracies)
        model_accuracies[name] = average_accuracy
        print(f"Average Accuracy for {name}: {average_accuracy}\n\n")
        print("All the accuracies:")
    print("*" * 100)
    print("*" * 100)
    show(model_accuracies)
    return model_accuracies


def highest_accuracy(dict_accuracy):
    highest_value_key = max(dict_accuracy, key=dict_accuracy.get)
    highest_value = dict_accuracy[highest_value_key]  
    print(f"The higest accuracy in this experiment:\n{highest_value_key} -->> {highest_value}")
    return(highest_value_key, highest_value)


def float_range(start, stop, step):
    while start < (stop):
        yield round(start, 3)
        start += step


def float_generator(start=0.0, stop=1.0, step=0.1, default=0.0):
    def_value = [default]
    float_list = list(float_range(start, stop, step))
    return sorted(set(def_value + float_list))


def int_generator(start=0, stop=100, step=10, default=0):
    def_value = [default]
    base_list = [0, 1, 10, 100, 1000]
    generated_list = list(range(start, (stop+1), step))
    final_list = list(set(def_value+ base_list + generated_list))
    return sorted(final_list)


def int_or_float(start=0, stop_float=1.0, stop_int=100, step_float=0.1, step_int=10, default=0):
    def_value = [default]
    int_list = int_generator(start, stop_int, step_int)
    float_list = float_generator(start, stop_float, step_float)
    return sorted(set(def_value + int_list + float_list))


def bool_type():
    return([True, False])



def calculate_combinations(param_dict):
    total_combinations = 1
    for key in param_dict:
        total_combinations *= len(param_dict[key])
    p = inflect.engine()
    total_in_words = p.number_to_words(total_combinations)
    return total_combinations, total_in_words