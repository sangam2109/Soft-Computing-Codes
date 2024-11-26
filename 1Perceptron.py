import numpy as np

def perceptron_train(X, y, epochs=10, learning_rate=0.1):
    weights = np.zeros(X.shape[1])
    bias = 0

    for epoch in range(epochs):
        print("---------------------- Epoch:", epoch, "------------------------------------")
        for i in range(len(X)):
            print("Input:", X[i])
            linear_output = np.dot(X[i], weights) + bias
            y_pred = 1 if linear_output >= 0 else 0

            error = y[i] - y_pred
            print("Error:", error)
            weights += learning_rate * error * X[i]
            print("Weights:", weights)
            bias += learning_rate * error
            print("Bias:", bias)

    return weights, bias

X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
y = np.array([1, 1, 1, 0])
weights, bias = perceptron_train(X, y)
print("Weights:", weights)
print("Bias:", bias)
