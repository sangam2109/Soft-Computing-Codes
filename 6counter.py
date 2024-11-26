import numpy as np

class CounterPropagationNetwork:
    def __init__(self, input_dim, num_hidden, output_dim):
        self.kohonen_weights = np.random.rand(num_hidden, input_dim)  # Kohonen layer weights
        self.grossberg_weights = np.random.rand(output_dim, num_hidden)  # Grossberg layer weights

    def kohonen_winner(self, x):
        # Find the winning Kohonen neuron (closest weight vector)
        distances = np.linalg.norm(self.kohonen_weights - x, axis=1)
        return np.argmin(distances)

    def train(self, X, Y, kohonen_lr=0.1, grossberg_lr=0.1, epochs=10):
        for epoch in range(epochs):
            for x, y in zip(X, Y):
                # Forward pass: Kohonen layer
                winner = self.kohonen_winner(x)

                # Update Kohonen weights (unsupervised learning)
                self.kohonen_weights[winner] += kohonen_lr * (x - self.kohonen_weights[winner])

                # Update Grossberg weights (supervised learning)
                self.grossberg_weights[:, winner] += grossberg_lr * (y - self.grossberg_weights[:, winner])

    def predict(self, X):
        predictions = []
        for x in X:
            winner = self.kohonen_winner(x)  # Find the winning Kohonen neuron
            predictions.append(self.grossberg_weights[:, winner])  # Use its Grossberg weights for output
        return np.array(predictions)

# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input data
Y = np.array([[0], [1], [1], [0]])              # Output data

# Create and train the network
cpn = CounterPropagationNetwork(input_dim=2, num_hidden=2, output_dim=1)
cpn.train(X, Y, epochs=100)

# Predict
predictions = cpn.predict(X)
print("Predictions:\n", predictions)
