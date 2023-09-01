# This example shows how scikit-learn can be used to recognize images of hand-written digits, from 0-9.

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import metrics
from utils import preprocess_data, train_model, split_data, read_digits, test_train_dev_split, predict_and_eval

# The digits dataset consists of 8x8 pixel images of digits. The images attribute of the dataset stores 8x8 arrays of grayscale values for each image. We will use these arrays to visualize the first 4 images. The target attribute of the dataset stores the digit each image represents and this is included in the title of the 4 plots below.
# Note: if we were working from image files (e.g., ‘png’ files), we would load them using matplotlib.pyplot.imread.

# 1. Data Loading
x, y = read_digits()

# 2. Test, Train, Dev Split
X_train, X_test, X_dev, y_train, y_test, y_dev = test_train_dev_split()

# 3. Data Preprocessing
X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)
X_dev = preprocess_data(X_dev)

# 4. Train the data
model = train_model(X_train, y_train, {'gamma': 0.001}, model_type='svm')

# # 5. Model Prediction and Evaluation
# predicted, report, confusion_matrix = predict_and_eval(model, X_test, y_test)

# # Visualization code (unchanged)

# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, prediction in zip(axes, X_test, predicted):
#     ax.set_axis_off()
#     image = image.reshape(8, 8)
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title(f"Prediction: {prediction}")

# print(f"Classification report for classifier {model}:\n{report}\n")

# disp = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
# disp.figure_.suptitle("Confusion Matrix")
# print(f"Confusion matrix:\n{confusion_matrix}")

# plt.show()



# 5. Model Prediction and Evaluation
predicted, report, confusion_matrix = predict_and_eval(model, X_test, y_test)

# Visualization code (including displaying the confusion matrix)
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

print(f"Classification report for classifier {model}:\n{report}\n")

# Display the confusion matrix
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=range(10))
disp.plot(cmap=plt.cm.Blues, values_format=".0f")
plt.title("Confusion Matrix")

plt.show()