from helpers import *
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import csv

# Load the data
data = pd.read_csv('melb_data.csv')

# Select target
y = data.Price

# To keep things simple, we'll use only numerical predictors
melb_predictors = data.drop(['Price'], axis=1)
X = melb_predictors.select_dtypes(exclude=['object'])

# handling missing data
# Imputation
my_impute = SimpleImputer()
X_imputed = pd.DataFrame(my_impute.fit_transform(X))

# Imputation removed column names; put them back
X_imputed.columns = X.columns

X = X_imputed.join(data['Type'])

# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

# handling categorical variables ['Type']
X_train, X_valid = new_encoder(X_train, X_valid)
X_train = X_train.rename(columns={'0': 'h', '1': 't', '2': 'u'})
X_valid = X_valid.rename(columns={'0': 'h', '1': 't', '2': 'u'})


# model training
my_model = XGBRegressor()
my_model.fit(X_train, y_train)

model_saver(my_model)

valid = my_model.predict(X_valid)
print(mean_absolute_error(y_valid, valid))
with open('MAE.csv', 'w') as f:
    writer = csv.writer(f)
    var = float(mean_absolute_error(y_valid, valid))
    writer.writerow(['MAE value', str(var)])

print("Done.")
