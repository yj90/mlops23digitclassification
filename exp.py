"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Import datasets, classifiers and performance metrics
from sklearn import metrics, svm
from utils import preprocess_data, split_data, train_model, read_digits, predict_and_eval, train_test_dev_split

# 1. Get the dataset
X, y = read_digits()




# Add these statements to your code
# 2.1 The number of total samples in the dataset (train + test + dev)
total_samples = len(X)
print("Total samples in the dataset:", total_samples)

# 2.2 Size (height and width) of the images in the dataset
image_height, image_width = X[0].shape
print("Image size (height x width):", image_height, "x", image_width)

# 3. Data splitting -- to create train and test sets
X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X, y, test_size=0.3, dev_size=0.2)

# 4. Data preprocessing
X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)
X_dev = preprocess_data(X_dev)




# HYPER PARAMETER TUNING
# take all combinations of gamma and C
gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
C_ranges = [0.1, 1, 2, 5, 10]
best_acc_so_far=-1
best_model=None
for cur_gamma in gamma_ranges:
    for cur_C in C_ranges:
        #print("Running for gamma={} C={}".format(cur_gamma, cur_C))
        # train model with cur_gamma and cur_C
        # 5. Model training
        cur_model = train_model(X_train, y_train, {'gamma': cur_gamma, "C": cur_C}, model_type="svm")
        # get some performance metric on DEV set
        cur_accuracy = predict_and_eval(cur_model, X_dev, y_dev)
        # select the hparam that yield the best performance on DEV set
        if cur_accuracy > best_acc_so_far:
            print("New best accuracy:", cur_accuracy)
            best_acc_so_far = cur_accuracy
            optimal_gamma = cur_gamma
            optimal_C = cur_C
            best_model = cur_model


print("Optimal parameters gamma:", optimal_gamma, "C: ", optimal_C)

'''
# 5. Model training
model = train_model(X_train, y_train, {'gamma': optimal_gamma, "C":optimal_C}, model_type="svm")
'''

# 6. Getting model predictions on test set
# 7. Qualitative sanity check of the predictions
# 8. Evaluation
test_acc = predict_and_eval(best_model, X_test, y_test)
print("Test accuracy:", test_acc)