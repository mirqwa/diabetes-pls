import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def get_data():
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_and_evaluate_pls_model(X_train, X_test, y_train, y_test):
    n_components = 3
    pls_model = PLSRegression(n_components=n_components)
    pls_model.fit(X_train, y_train)
    y_pred = pls_model.predict(X_test)
    # Evaluate the model performance
    r_squared = pls_model.score(X_test, y_test)
    print(f"R-Squared Error: {r_squared}")
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data()
    train_and_evaluate_pls_model(X_train, X_test, y_train, y_test)
