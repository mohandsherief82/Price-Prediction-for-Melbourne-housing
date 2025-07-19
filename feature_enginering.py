import matplotlib.pyplot as plt
from helpers import *
import seaborn as sns

# Read the data
data = pd.read_csv('melb_data.csv')

# Separate target from predictors
y = data.Price
X_train_full = data.drop(['Price'], axis=1)

# Drop columns with missing values (simplest approach)
cols_with_missing = [col for col in X_train_full.columns if X_train_full[col].isnull().any()]
X_train_full.drop(cols_with_missing, axis=1, inplace=True)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and
                        X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = low_cardinality_cols + numerical_cols
x = X_train_full[numerical_cols].copy()

# Label encoding for categorical
for colname in x.select_dtypes("object"):
    x[colname], _ = x[colname].factorize()

# All discrete features should now have integer dtypes (double-check this before using MI!)
# discrete_features = x.dtypes == int
#
# mi_scores = make_mi_scores(x, y, discrete_features)
# var = mi_scores[::3]
#
# # bar plot to make comparisons easier.
# plt.figure(dpi=100, figsize=(8, 5))
# plot_mi_scores(mi_scores)

# plot the data as scatter plot.
# sns.relplot(data=x, x='length', y='width', hue='Cluster')
# plot as catplot
# sns.catplot(x="Rooms", y="Type", data=x, kind="boxen", height=6)

plt.show()
