from sklearn import metrics
from utils import preprocess_data, train_test_dev_split, tune_hparams, read_digits, predict_and_eval
import itertools

# 1. Get the dataset
X, y = read_digits()

# 3. Data splitting -- to create train and test sets
test_sizes = [0.1, 0.2, 0.3]
dev_sizes = [0.1, 0.2, 0.3]

for test_size, dev_size in itertools.product(test_sizes, dev_sizes):
    print(f"test_size={test_size} dev_size={dev_size} train_size={1 - test_size - dev_size}", end=' ')
    
    X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X, y, test_size=test_size, dev_size=dev_size)
    
    # 4. Data preprocessing
    X_train = preprocess_data(X_train)
    X_test = preprocess_data(X_test)
    X_dev = preprocess_data(X_dev)

    # HYPERPARAMETER TUNING
    # Create a list of tuples with all combinations of gamma and C
    gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
    C_ranges = [0.1, 1, 2, 5, 10]
    param_combinations = list(itertools.product(gamma_ranges, C_ranges))

    best_hparams, best_model, best_dev_acc, best_test_acc, best_train_acc = tune_hparams(X_train, y_train, X_dev, y_dev, X_test, y_test, param_combinations)

    print(f"dev_acc={best_dev_acc:.2f} test_acc={best_test_acc:.2f} train_acc={best_train_acc:.2f}")
    
    # Print the best hyperparameters found
    print(f"Best hyperparameters: gamma={best_hparams['gamma']}, C={best_hparams['C']}\n")
