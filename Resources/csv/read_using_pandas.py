import pandas as pd

df = pd.read_csv("feature_vectors_with_labels.csv")
print("--printing the dataframe---")
print(df)
df1 = df.loc[:, ~df.columns.str.contains('^Unnamed')]
print("---df1---")
print(df1)
print("--Converting dataframe to numpy array---")
print(df.as_matrix())
print("--Converting dataframe1 to numpy array---")
print(df1.as_matrix())