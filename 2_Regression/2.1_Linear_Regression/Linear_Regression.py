# Before start, make sure to upload 'Salary_Data.csv'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def run_simple_linear_regression(data_path="Salary_Data.csv"):
    dataset = pd.read_csv(data_path)

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    regressor = LinearRegression()
    # We assign the regression algorithm, which we aim to learn, to the empty model.
    regressor.fit(X_train, y_train)
    # Training it with the train sets.
    # Math formula -> y = b0 + b1.x

    y_pred = regressor.predict(X_test)
    # By looking at the independent test element variables,
    # attempts are made to predict the y-test results.
    # The formula is calculated in here.
    # b0 : intercept
    # b1 : slope
    # b0 and b1 become constants.

    plt.scatter(X_train, y_train, color="red")
    plt.plot(X_train, regressor.predict(X_train), color="blue")
    # predict(X_train) -> Predicted y values ​​calculated by the model for each x
    # plot(x axis, y axis)
    plt.title("Salary vs Experience (Training Set)")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.show()

    plt.scatter(X_test, y_test, color="violet")
    plt.plot(X_train, regressor.predict(X_train), color="blue")
    # We don't want to draw a new line. b0 and b1 already calculated.
    plt.title("Salary vs Experience (Test Set)")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.show()

    return regressor, y_pred


if __name__ == "__main__":
    run_simple_linear_regression()
