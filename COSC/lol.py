import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sklearn.datasets
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.metrics
from sklearn.svm import SVC
# Load dataset
iris = sklearn.datasets.load_iris()

# Separate labels y from the features/attributes X. Note that X and y are NumPy arrays and NOT Pandas dataframes
# Y represents labels
X = iris.data
y = iris.target

# Print the shape (# rows, # col) of the data X
X.shape

# Print the shape of the labels y
y.shape

# Divide data/labels into train/test sets
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2)

# print shapes of train/test sets for sanity
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Create a StandardScaler object
scaler = sklearn.preprocessing.StandardScaler()
# Use the .fit method on the *training* data so the object knows what transformation is needed
scaler.fit(X_train)
# Use the .transform method to scale both the training set and the test set
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
svc = sklearn.svm.SVC(kernel="linear")