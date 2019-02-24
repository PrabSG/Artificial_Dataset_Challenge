import pandas as pd
import numpy as np
from sklearn import svm

VAR_ORDER_DICT = {
    "education": ["primary", "secondary", "tertiary"],
    "default": ["no", "yes"],
    "housing": ["no", "yes"],
    "loan": ["no", "yes"],
    "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "y": ["no", "yes"]
}
# Here trying to make as many of the inputs numeric as possible,
# so unknown will be treated as an average value for the column
ORDER_VARS = ["education", "default", "housing", "loan", "month", "y"]

NUMERIC_COLS = ["age", "education", "default", "housing", "loan", "day", "month", "campaign", "pdays", "previous"]

raw_data = pd.read_csv("bank-full.csv", delimiter=';')
num_data = raw_data


# Replace those categorical variables that have an ordering using basic label encoding.
for col in ORDER_VARS:
    num_data[col] = num_data[col].map(
        lambda x: "unknown" if x == "unknown" else VAR_ORDER_DICT.get(col).index(x), na_action='ignore')

avg = {}

for categ in {"education", "default", "housing", "loan"}:
    catSeries = num_data[categ]
    catSeries = catSeries[catSeries != "unknown"]
    meanVal = np.sum(catSeries.get_values()) / (catSeries.__len__())
    avg[categ] = meanVal

for col in ORDER_VARS:
    num_data[col] = num_data[col].map(
        lambda x: avg[col] if x == "unknown" else x)

X_data = num_data[NUMERIC_COLS].get_values()
y_labels = num_data["y"].get_values().squeeze()

# Implement svm
clf = svm.SVC(gamma="scale")
clf.fit(X_data, y_labels)
