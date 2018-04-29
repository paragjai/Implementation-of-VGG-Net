# To write to a csv which makes sense:
import pandas as pd
import numpy as np
fv1 = np.array([10,20,30,40]) # (4,)
label1 = 1

fv2 = np.array([50,60,70,80]) # (4,)
label2 = 2

fv1_with_label = np.append(fv1, label1) # (5,) # Last field of each feature vector now is the class/label the feature vector belongs to
fv2_with_label = np.append(fv2, label2) # (5,)

write_to_pandas = np.array([fv1_with_label, fv2_with_label])

df = pd.DataFrame(write_to_pandas)
print("----This is what will be written to the file----")
print(df)

df.to_csv("feature_vectors_with_labels.csv")
