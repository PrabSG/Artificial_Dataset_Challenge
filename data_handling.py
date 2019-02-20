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

# Read in the dataset, and shuffle it as it appears to be ordered by date.
raw_data = pd.read_csv("bank-full.csv", delimiter=';')
raw_data = raw_data.sample(frac=1).reset_index(drop=True)

# Separate the ground truth from the rest of the dataset.
y_series = raw_data.loc[:, 'y']
X_dataframe = raw_data.iloc[:, :-1].drop(columns='duration')
X_dataframe = X_dataframe.replace({"unknown": np.nan})
ALL_LABELS = list(X_dataframe)

# Replace those categorical variables that have an ordering using basic label encoding.
for label in ORDER_LABELS:
    X_dataframe[label] = X_dataframe[label].map(
        lambda x: np.nan if x is np.nan else LABEL_ORDER_DICT.get(label).index(x), na_action='ignore')

# One-hot encoding of those categorical variables where a relative ordering is semantically incorrect.
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

# Locating the Nan values in the original dataframe, and then replacing it with the average for
# the associated columns in the one-hot encoded data frame.
for label in ALL_LABELS:
    sub_labels = filter(lambda sub: sub.startswith(label), X_dataframe_oneHot.columns)
    for one_hot_subLabel in sub_labels:
        X_dataframe_oneHot.loc[X_dataframe[label].isna(), one_hot_subLabel] = columnAvgs[one_hot_subLabel]

X = X_dataframe_oneHot.get_values()
y = y_series.values

np.savez("bank-full-shuffled-onehot", trainingdata=X, labels=y)
