'''Command to run:
python knn_as_classifier.py --featureVectors features_vectors_with_corresponding_labels.csv --neighbors 3
'''

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import argparse # For command line arguments
import pandas as pd
import numpy as np
ap = argparse.ArgumentParser()

#If we do not have a default value we set required to be True. 
ap.add_argument("-f", "--featureVectors", help="path to the file containing feature vectors and labels (.csv)", type=str, required=True)

# Since we have a default value, we do not need required=True
ap.add_argument("-k", "--neighbors", help="# of nearest neighbors for classification", type=int, default=1)

# Convert this to a dictionary
cmd_dict = vars(ap.parse_args())

df = pd.read_csv(cmd_dict["featureVectors"])
print(df)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
print("---------------Correctly read dataframe--------------------")
print(df)

dataset = np.array(df)
data = dataset[:,:-1]
print("----data[0]----")
print(data[0])
print("----data[0].shape----")
print(data[0].shape)

labels = dataset[:,-1]
print("----labels----")
print(labels)
print("----labels.shape----")
print(labels.shape)



# Map the string labels (class name) to integers.
le = LabelEncoder()
labels_int = le.fit_transform(labels) # le.classes_ attribute will have the corresponding string labels.
print("Example integer labels", labels_int[0:5])
print("----integer labels----")
print(labels_int)
print(np.array(list(labels_int)).dtype)


# partition the data into training and testing. 
# Generally, 75 percent is kept for training and 25 percent for testing.
# and the training and testing data consists of featureVectors.
(trainX, testX, trainY, testY) = train_test_split(data, labels_int, test_size=0.25, random_state = 42)
print("trainX", trainX.shape)
print("trainY", trainY.shape)
print("[INFO] evaluating k-NN classifier...")
#print(np.array(list(trainY)).dtype)
#print(np.array(list(trainX)).dtype)
# Just because we are using a library directly for knn, doesn't mean that you need not learn how the algorithm works.
# We use it because the implementation of this will be quite efficient.
model = KNeighborsClassifier(n_neighbors = cmd_dict["neighbors"])
model.fit(trainX, trainY)

# We are trying to predict on test data.
# Refer resources folder to learn about recall, precision and F1 as a performance measure. Support is same as frequency.
y_cap = model.predict(testX) # Predicted labels
y = testY # Actual labels
print(classification_report(y, y_cap, target_names = le.classes_))
