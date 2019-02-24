import pandas as pd
import numpy as np

VAR_ORDER_DICT = {
    "default": ["no", "unknown", "yes"],
    "housing": ["no", "unknown", "yes"],
    "loan": ["no", "unknown", "yes"],
    "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
}

ORDER_VARS = ["default", "housing", "loan", "month"]
CATEGORICAL_VARS = ["job", "marital", "education", "contact", "poutcome"]

# Read in the dataset, and shuffle it as it appears to be ordered by date.
raw_data = pd.read_csv("bank-full.csv", delimiter=';')

# Heavily imbalanced dataset where there are significantly more data where y="no".
# So rebalancing the data now.
num_of_yes = raw_data["y"].value_counts()[1]
no_dataframe = raw_data[raw_data["y"] == "no"].sample(num_of_yes)
yes_dataframe = raw_data[raw_data["y"] == "yes"]
balanced_data = pd.concat([no_dataframe, yes_dataframe])

# Shuffle rows of data
balanced_data = balanced_data.sample(frac=1).reset_index(drop=True)

# Separate the ground truth from the rest of the dataset.
y_series = balanced_data.loc[:, 'y']
X_dataframe = balanced_data.iloc[:, :-1].drop(columns='duration')
ALL_VARS = list(X_dataframe)

# Replace those categorical variables that have an ordering using basic label encoding.
for col in ORDER_VARS:
    X_dataframe[col] = X_dataframe[col].map(
        lambda x: VAR_ORDER_DICT.get(col).index(x), na_action='ignore')

# One-hot encoding of those categorical variables where a relative ordering is semantically incorrect.
X_dataframe_oneHot = X_dataframe.copy()
for col in CATEGORICAL_VARS:
    X_dataframe_oneHot = pd.get_dummies(X_dataframe_oneHot, columns=[col], prefix=[col])

ONEHOT_VARS = list(X_dataframe_oneHot)

# Encode the ground truth to be binary values
y_series = y_series.map({"no" : 0, "yes" : 1})

X = X_dataframe_oneHot.get_values()
y = y_series.values

np.savez("bank-full-balanced-shuffled-onehot-unknowns", training_data=X, labels=y)
