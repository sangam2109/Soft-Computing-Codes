import numpy as np

# Activation function: Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Neural Network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.weights_input_hidden = np.random.rand(input_size, hidden_size) - 0.5
        self.weights_hidden_output = np.random.rand(hidden_size, output_size) - 0.5
        self.bias_hidden = np.random.rand(1, hidden_size) - 0.5
        self.bias_output = np.random.rand(1, output_size) - 0.5

    def feedforward(self, X):
        # Compute hidden layer activations
        self.hidden = sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        # Compute output layer activations
        self.output = sigmoid(np.dot(self.hidden, self.weights_hidden_output) + self.bias_output)

    def backpropagation(self, X, y, learning_rate):
        # Compute output layer error and delta
        output_error = y - self.output
        output_delta = output_error * sigmoid_derivative(self.output)

        # Compute hidden layer error and delta
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden)

        # Update weights and biases
        self.weights_hidden_output += np.dot(self.hidden.T, output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.feedforward(X)
            self.backpropagation(X, y, learning_rate)
            if epoch % 100 == 0:
                loss = np.mean((y - self.output) ** 2)
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Generate synthetic dataset
np.random.seed(42)
X = np.random.rand(100, 3)  # 100 samples, 3 features
y = np.array([[1 if sum(row) > 1.5 else 0] for row in X])  # Simple binary output based on sum of features

# Normalize the input data
X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

# Train-test split
train_size = int(0.7 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Initialize and train the neural network
nn = NeuralNetwork(input_size=3, hidden_size=5, output_size=1)
nn.train(X_train, y_train, epochs=1000, learning_rate=0.1)

# Test the neural network
nn.feedforward(X_test)
predictions = (nn.output > 0.5).astype(int)
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
