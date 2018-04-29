import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder # Convert string labels to whole numbers


df = pd.read_csv("feature_vectors_with_labels.csv")
df1 = df.loc[:, ~df.columns.str.contains('^Unnamed')]
print("---dataframe---")
print(df1)

# Converting to numpy array
dataset = np.array(df1)
print("---dataset---")
print(dataset)
print("---type(dataset)---")
print(type(dataset))
print("---dataset.shape---")
print(dataset.shape)

data = dataset[:,:-1]
print("---data---")
print(data)
labels = dataset[:,-1]
print("---labels---")
print(labels)

# Map the string labels (class name) to whole numbers.
le = LabelEncoder()
labels_whole = le.fit_transform(labels) # le.classes_ attribute will have the corresponding string labels.
print("Example whole number labels", labels_whole[0:5])
print("----whole numbers labels----")
print(labels_whole)
print(np.array(list(labels_whole)).dtype)
