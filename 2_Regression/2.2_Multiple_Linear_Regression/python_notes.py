import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def load_dataset():
    # Importing the dataset
    dataset = pd.read_csv('50_Startups.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    print(X)
    return X, y


def encode_categorical_data(X):
    ct = ColumnTransformer(
        transformers=[('encoder', OneHotEncoder(), [3])],
        remainder='passthrough'
    )
    # Converts categorical data to numerical form.
    # transformers = [(tag, numeric function, number of category data),
    # remainder='passthrough' -> Encode the text column, skip the other columns (the numeric ones)]

    X = np.array(ct.fit_transform(X))
    print(X)

    return X


def split_dataset(X, y):
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    # Training the Multiple Linear Regression model
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor


def predict_and_compare(regressor, X_test, y_test):
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)

    np.set_printoptions(precision=2)
    # Limit the decimal part to 2 digits.

    print(
        np.concatenate(
            (
                y_pred.reshape(len(y_pred), 1),  # Predicted values
                y_test.reshape(len(y_test), 1)   # Real values
            ),
            1
        )
    )
    # We create a table to compare the predicted result with the actual result side by side.

    # reshape(row, column)
    # It preserves the same data while only changing its format (row-column structure).

    # np.concatenate((array1, array2), axis) ->
    # It combines data of the same type along the specified axis.
    # array1, array2 → arrays to be combined
    # axis → the direction in which they will be combined
    # (0: One below the other (add row), 1: Side by side (add column))


def main():
    X, y = load_dataset()
    X = encode_categorical_data(X)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    regressor = train_model(X_train, y_train)
    predict_and_compare(regressor, X_test, y_test)


if __name__ == "__main__":
    main()
