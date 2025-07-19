import pickle
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import csv
import matplotlib.pyplot as plt


#  obtain the final model object
def model_saver(model):
    # open a file in binary mode where you want to save the model
    with open('model.pkl', 'wb') as f:
        # use the pickle.dump() function to serialize the model object and save it to the file
        pickle.dump(model, f)

    # close the file object
    f.close()


# to use saved models
def model_user(model_path):
    with open(model_path, 'rb') as f:
        # use the pickle.load() function to deserialize the model object from the file
        model = pickle.load(f)

    # close the file object
    f.close()
    return model


# one hot encoder in one line.
def new_encoder(x_train, x_valid):
    # Apply one-hot encoder to each column with categorical data
    s = (x_train.dtypes == 'object')
    object_cols = list(s[s].index)
    oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    cols_train = pd.DataFrame(oh_encoder.fit_transform(x_train[object_cols]))
    cols_valid = pd.DataFrame(oh_encoder.transform(x_valid[object_cols]))

    # One-hot encoding removed index; put it back
    cols_train.index = x_train.index
    cols_valid.index = x_valid.index

    # Remove categorical columns (will replace with one-hot encoding)
    num_X_train = x_train.drop(object_cols, axis=1)
    num_X_valid = x_valid.drop(object_cols, axis=1)

    # Add one-hot encoded columns to numerical features
    OH_X_train = pd.concat([num_X_train, cols_train], axis=1)
    OH_X_valid = pd.concat([num_X_valid, cols_valid], axis=1)

    # Ensure all columns have string type
    OH_X_train.columns = OH_X_train.columns.astype(str)
    OH_X_valid.columns = OH_X_valid.columns.astype(str)

    # print("MAE from Approach 3 (One-Hot Encoding):")
    # print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
    return OH_X_train, OH_X_valid


def encode1(x, object_cols):
    # Apply one-hot encoder to each column with categorical data
    oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    cols_train = pd.DataFrame(oh_encoder.fit_transform(x[object_cols]))

    # One-hot encoding removed index; put it back
    cols_train.index = x.index

    # Remove categorical columns (will replace with one-hot encoding)
    num_X_train = x.drop(object_cols, axis=1)

    # Add one-hot encoded columns to numerical features
    new_x = pd.concat([num_X_train, cols_train], axis=1)

    # Ensure all columns have string type
    new_x.columns = new_x.columns.astype(str)

    # print("MAE from Approach 3 (One-Hot Encoding):")
    # print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
    return new_x


def get_value(value):
    print('''
    Note: 
    There could be an error value 
    and the given values are only predictions to what is given by user to a trained model. 
    Also, keep in mind that there other things that should be taken in consideration for housing prices. 
    ''')
    with open('MAE.csv', 'r') as f:
        reader = csv.reader(f)
        column_index = 1

        column_values = []
        for row in reader:
            column_values.append(row[column_index])

        error = float(column_values[0])

    max_value = value + error
    min_value = value - error

    print(f'Maximum Value: {max_value}')
    print(f'Minimum Value: {min_value}')


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


# one for real-valued targets (mutual_info_regression) and one for categorical targets (mutual_info_classif).
# computes the MI scores for our features and wraps them up in a nice dataframe
def make_mi_scores(x, y, discrete_features):
    mi_scores = mutual_info_regression(x, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=x.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_variance(pca, width=8, dpi=100):
    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )
    # Cumulative Variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )
    # Set up figure
    fig.set(figwidth=8, dpi=100)
    return axs
