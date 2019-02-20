import pandas as pd
import numpy as np
import sklearn.preprocessing

LABEL_ORDER_DICT = {
    "education": ["primary", "secondary", "tertiary"],
    "default": ["no", "yes"],
    "housing": ["no", "yes"],
    "loan": ["no", "yes"],
    "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
}

ORDER_LABELS = ["education", "default", "housing", "loan", "month"]
CATEGORICAL_LABELS = ["job", "marital", "contact", "poutcome"]

# Read in the dataset.
raw_data = pd.read_csv("bank-full.csv", delimiter=';')
data_array = raw_data.get_values()

# Separate the ground truth from the rest of the dataset.
y_series = raw_data.loc[:, 'y']
X_dataframe = raw_data.iloc[:, :-1].drop(columns='duration')
X_dataframe = X_dataframe.replace({"unknown": np.nan})
ALL_LABELS = list(X_dataframe)
print(ALL_LABELS)

print(X_dataframe.get_values()[1])
# Replace those categorical variables that have an ordering using basic label encoding.
for label in ORDER_LABELS:
    X_dataframe[label] = X_dataframe[label].map(
        lambda x: np.nan if x is np.nan else LABEL_ORDER_DICT.get(label).index(x), na_action='ignore')

# One hot encoding of those categorical variables where a relative ordering is semantically incorrect.
X_dataframe_oneHot = X_dataframe.copy()
for label in CATEGORICAL_LABELS:
    X_dataframe_oneHot = pd.get_dummies(X_dataframe_oneHot, columns=[label], prefix=[label])

ONEHOT_LABELS = list(X_dataframe_oneHot)

# Using the mean value to encode missing data.
# Alternative method would be to build another model that could predict what the value may be,
# depending on the other available attributes for that row of data.
columnAvgs = {}
for label in ONEHOT_LABELS:
    columnAvgs[label] = X_dataframe_oneHot[label].mean(skipna=True)

for label in ALL_LABELS:
    sub_labels = filter(lambda sub: sub.startswith(label), X_dataframe_oneHot.columns)
    for one_hot_subLabel in sub_labels:
        X_dataframe_oneHot.loc[X_dataframe[label].isna(), one_hot_subLabel] = columnAvgs[one_hot_subLabel]
