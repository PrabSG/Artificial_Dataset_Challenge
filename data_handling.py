import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

LABEL_ORDER_DICT = {
    "education" :   ["primary", "secondary", "tertiary"],
    "default"   :   ["no", "yes"],
    "housing"   :   ["no", "yes"],
    "loan"      :   ["no", "yes"],
    "month"     :   ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
}

def labelEncodeOrderable(label, categoryVal):
    if not(categoryVal == "unknown"):
        return LABEL_ORDER_DICT.get(label).index(categoryVal)











# Read in the dataset
raw_data = pd.read_csv("bank-full.csv", delimiter=';')
data_array = raw_data.get_values()

# Separate the ground truth from the rest of the dataset
y_series = raw_data.loc[:,'y']
X_dataframe = raw_data.iloc[:, :-1].drop(columns='duration')




































print(list(X_dataframe[0]))
print(list(raw_data))