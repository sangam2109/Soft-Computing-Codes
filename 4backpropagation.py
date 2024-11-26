import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def backpropagation(X, y, epochs=10000, learning_rate=0.1):
    input_neurons, hidden_neurons, output_neurons = X.shape[1], 2, 1

    hidden_weights = np.random.uniform(size=(input_neurons, hidden_neurons))
    hidden_bias = np.random.uniform(size=(1, hidden_neurons))
    output_weights = np.random.uniform(size=(hidden_neurons, output_neurons))
    output_bias = np.random.uniform(size=(1, output_neurons))

    for _ in range(epochs):
        # Forward pass
        hidden_output = sigmoid(np.dot(X, hidden_weights) + hidden_bias)
        predicted_output = sigmoid(np.dot(hidden_output, output_weights) + output_bias)

        # Backward pass
        error = y - predicted_output
        d_output = error * sigmoid_derivative(predicted_output)
        d_hidden = d_output.dot(output_weights.T) * sigmoid_derivative(hidden_output)

        # Update weights and biases
        output_weights += hidden_output.T.dot(d_output) * learning_rate
        output_bias += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        hidden_weights += X.T.dot(d_hidden) * learning_rate
        hidden_bias += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    return predicted_output

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
output = backpropagation(X, y)
print("Predicted Output:\n", output)
