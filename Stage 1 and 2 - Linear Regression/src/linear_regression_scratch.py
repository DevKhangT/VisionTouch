import numpy as np
import pandas as pd
class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, epochs=1000, calibration_type="IRIS_ONLY"):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.calibration_type = calibration_type
        self.weights = None
        self.bias = None
        self.losses = []
    def fit(self, X, y):
        number_of_samples, number_of_features = X.shape
        number_of_outputs = y.shape[1]
        self.weights = np.zeros((number_of_features, number_of_outputs))
        self.bias = np.zeros((1, number_of_outputs))
        for _ in range(self.epochs):
            predicted = self.predict(X)
            loss = np.mean((predicted - y) ** 2)
            self.losses.append(loss)
            dWL = 2 * X.T @ (predicted - y) / number_of_samples
            dBL = 2 * np.mean(predicted - y, axis=0, keepdims=True)
            self.weights -= self.learning_rate * dWL
            self.bias -= self.learning_rate * dBL
            if(_ % 100 == 0):
                print(f"Epoch {_}: Loss = {loss}")
    def predict(self, X):
        return X @ self.weights + self.bias