import pandas as pd
import numpy as np

VAR_ORDER_DICT = {
    "education": ["primary", "secondary", "tertiary"],
    "default": ["no", "yes"],
    "housing": ["no", "yes"],
    "loan": ["no", "yes"],
    "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
}

ORDER_VARS = ["education", "default", "housing", "loan", "month"]
CATEGORICAL_VARS = ["job", "marital", "contact", "poutcome"]

# Read in the dataset, and shuffle it as it appears to be ordered by date.
raw_data = pd.read_csv("bank-full.csv", delimiter=';')
raw_data = raw_data.sample(frac=1).reset_index(drop=True)

# Separate the ground truth from the rest of the dataset.
y_series = raw_data.loc[:, 'y']
X_dataframe = raw_data.iloc[:, :-1].drop(columns='duration')
X_dataframe = X_dataframe.replace({"unknown": np.nan})
ALL_VARS = list(X_dataframe)

# Replace those categorical variables that have an ordering using basic label encoding.
for col in ORDER_VARS:
    X_dataframe[col] = X_dataframe[col].map(
        lambda x: np.nan if x is np.nan else VAR_ORDER_DICT.get(col).index(x), na_action='ignore')

# One-hot encoding of those categorical variables where a relative ordering is semantically incorrect.
X_dataframe_oneHot = X_dataframe.copy()
for col in CATEGORICAL_VARS:
    X_dataframe_oneHot = pd.get_dummies(X_dataframe_oneHot, columns=[col], prefix=[col])

ONEHOT_VARS = list(X_dataframe_oneHot)

# Using the mean value to encode missing data.
# Alternative method would be to build another model that could predict what the value may be,
# depending on the other available attributes for that row of data.
columnAvgs = {}
for col in ONEHOT_VARS:
    columnAvgs[col] = X_dataframe_oneHot[col].mean(skipna=True)

# Locating the Nan values in the original dataframe, and then replacing it with the average for
# the associated columns in the one-hot encoded data frame.
for col in ALL_VARS:
    sub_labels = filter(lambda sub: sub.startswith(col), X_dataframe_oneHot.columns)
    for one_hot_subLabel in sub_labels:
        X_dataframe_oneHot.loc[X_dataframe[col].isna(), one_hot_subLabel] = columnAvgs[one_hot_subLabel]

# Encode the ground truth to be binary values
y_series = y_series.map({"no" : 0, "yes" : 1})

X = X_dataframe_oneHot.get_values()
y = y_series.values

np.savez("bank-full-shuffled-onehot", training_data=X, labels=y)
