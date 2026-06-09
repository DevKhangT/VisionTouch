import numpy as np
class polynomial_regression_scratch:
    def __init__(self, degree = 2, learning_rate = 0.1, epoches = 1000):
        self.degree = degree
        self.learning_rate = learning_rate
        self.epoches = epoches
        self.weights = None
        self.bias = None
    def polynomial_features(self, X):
        # Ex: [1, x , x^2]
        num_of_samples, number_of_features = X.shape
        features = []
        features.append(X)
        for i in range(number_of_features):
            for j in range(i, number_of_features):
                new_feature = (X[:, i] * X[:, j]).reshape(-1, 1)
                features.append(new_feature)
        # [ 1 x x^2]
        # [ x x^2 x^3]
        # [ x^2 x^3 x^4]
        return np.hstack(features)

    def fit(self, X, y):
        X_poly = self.polynomial_features(X)
        number_of_outputs = y.shape[1]
        number_of_samples, number_of_features = X_poly.shape
        self.weights = np.zeros((number_of_features, number_of_outputs))
        self.bias = np.zeros((1, number_of_outputs))
        for _ in range(self.epoches):
            predicted = self.predict(X_poly)
            dWL = 2 * X_poly.T @ (predicted - y) / number_of_samples
            dBL = 2 * np.mean(predicted - y, axis=0, keepdims=True)
            self.weights -= self.learning_rate * dWL
            self.bias -= self.learning_rate * dBL
        pass
    def predict(self, X):
        return X @ self.weights + self.bias