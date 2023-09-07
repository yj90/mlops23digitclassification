from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn import svm, datasets, metrics

def read_digits():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    return X, y 

def preprocess_data(data):
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data

def split_data(x, y, test_size, random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.5, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def train_model(x, y, model_params, model_type="svm"):
    if model_type == "svm":
        clf = svm.SVC(**model_params)
    model = clf
    model.fit(x, y)
    return model

def tune_hparams(X_train, y_train, X_dev, y_dev, X_test, y_test, param_combinations):
    best_acc_so_far = -1
    best_model = None
    best_hparams = None
    best_test_acc = -1
    best_train_acc = -1
    for param_combination in param_combinations:
        model_params = {"gamma": param_combination[0], "C": param_combination[1]}
        model = train_model(X_train, y_train, model_params)
        cur_dev_accuracy = predict_and_eval(model, X_dev, y_dev)
        cur_test_accuracy = predict_and_eval(model, X_test, y_test)
        cur_train_accuracy = predict_and_eval(model, X_train, y_train)
        if cur_dev_accuracy > best_acc_so_far:
            best_acc_so_far = cur_dev_accuracy
            best_model = model
            best_hparams = model_params
            best_test_acc = cur_test_accuracy
            best_train_acc = cur_train_accuracy
    return best_hparams, best_model, best_acc_so_far, best_test_acc, best_train_acc

def train_test_dev_split(X, y, test_size, dev_size):
    X_train_dev, X_test, Y_train_Dev, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    X_train, X_dev, y_train, y_dev = split_data(X_train_dev, Y_train_Dev, dev_size/(1-test_size), random_state=1)
    return X_train, X_test, X_dev, y_train, y_test, y_dev

def predict_and_eval(model, X_test, y_test):
    predicted = model.predict(X_test)
    return metrics.accuracy_score(y_test, predicted)
