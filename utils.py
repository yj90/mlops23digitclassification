# Import datasets, classifiers and performance metrics
# Import datasets, classifiers, and performance metrics
from sklearn import svm, datasets, metrics  # Added 'metrics' import
from sklearn.model_selection import train_test_split

# Read digits
def read_digits():
    digits = datasets.load_digits()
    x = digits.images
    y = digits.target
    return x, y

# We will define utils here:
def preprocess_data(data):
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data

# Split data into train, test, and dev subsets
def split_data(X, y, test_size=0.5, dev_size=0.2, random_state=1):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + dev_size, random_state=random_state
    )
    X_test, X_dev, y_test, y_dev = train_test_split(
        X_temp, y_temp, test_size=dev_size / (test_size + dev_size), random_state=random_state
    )
    return X_train, X_test, X_dev, y_train, y_test, y_dev

# Create a classifier: a support vector classifier
def train_model(X, y, model_params, model_type='svm'):
    if model_type == 'svm':
        clf = svm.SVC(**model_params)
    clf.fit(X, y)
    return clf

# Question1: Test, Train, Dev Split
def test_train_dev_split():
    # Read digits
    x, y = read_digits()
    
    # Split data into train, test, and dev subsets
    X_train, X_test, X_dev, y_train, y_test, y_dev = split_data(x, y, test_size=0.3, dev_size=0.2)
    
    return X_train, X_test, X_dev, y_train, y_test, y_dev

# Question2: Predict and Evaluate
def predict_and_eval(model, X_test, y_test):
    # Model Prediction
    predicted = model.predict(X_test)
    
    # Model Evaluation
    report = metrics.classification_report(y_test, predicted)
    confusion_matrix = metrics.confusion_matrix(y_test, predicted)
    
    return predicted, report, confusion_matrix
