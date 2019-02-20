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

ORDER_LABELS = ["education", "default", "housing", "loan", "month"]

def labelEncodeOrderableFunc(label, categoryVal):
    if not(categoryVal is None):
        return LABEL_ORDER_DICT.get(label).index(categoryVal)

labelEncodeOrderable = lambda label, x : np.nan if x is np.nan else 5


# Read in the dataset
raw_data = pd.read_csv("bank-full.csv", delimiter=';')
data_array = raw_data.get_values()

# Separate the ground truth from the rest of the dataset
y_series = raw_data.loc[:,'y']
X_dataframe = raw_data.iloc[:, :-1].drop(columns='duration')
# X_dataframe.mask(X_dataframe == "unknown", np.nan)
# print(X_dataframe.get_values()[1])
#
# for label in ORDER_LABELS:
#     X_dataframe[label].apply(labelEncodeOrderable, args=(label,))
#

df = pd.DataFrame(np.arange(10).reshape(-1, 2), columns=['A', 'B'])
print(df)
df.replace(3, np.nan)
print(df)































print(list(raw_data))