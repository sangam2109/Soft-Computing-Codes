import numpy as np

def delta_rule(X, y, epochs=10, learning_rate=0.1):
    weights = np.random.rand(X.shape[1])

    for epoch in range(epochs):
        for i in range(len(X)):
            output = np.dot(X[i], weights)
            error = y[i] - output
            weights += learning_rate * error * X[i]

    return weights

X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
y = np.array([1, 1, 1, 0])
weights = delta_rule(X, y)
print("Weights:", weights)
