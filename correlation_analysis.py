import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


VAR_ORDER_DICT = {
    "default": ["no", "unknown", "yes"],
    "housing": ["no", "unknown", "yes"],
    "loan": ["no", "unknown", "yes"],
    "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "y": ["no", "yes"]
}

ORDER_VARS = ["default", "housing", "loan", "month", "y"]

raw_data = pd.read_csv("bank-full.csv", delimiter=';')
num_data = raw_data
# Replace those categorical variables that have an ordering using basic label encoding.
for col in ORDER_VARS:
    num_data[col] = num_data[col].map(
        lambda x: VAR_ORDER_DICT.get(col).index(x), na_action='ignore')


# Calculating the correlation between each of the inputs and the output of if the customer will subscribe to a term deposit
y_correlation = num_data.corr().filter(["y"]).drop(["y"])

corrCategs = list(y_correlation.index)
corrVals = y_correlation.get_values().squeeze()

plt.bar(list(range(len(corrCategs))), corrVals, align='center', alpha=0.5)
plt.xticks(list(range(len(corrCategs))), corrCategs)
plt.ylabel('Correlation Coefficient')
plt.title('Data category')
plt.show()

