import numpy as np

def hebb_rule(X, y):
    weights = np.zeros(X.shape[1])
    print("Initial Weights:", weights)

    for i in range(len(X)):
        print("Input:", X[i])
        weights += X[i] * y[i]
        print("Weights:", weights)

    return weights

X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
y = np.array([1, 1, 1, -1])
weights = hebb_rule(X, y)
print("Weights:", weights)
