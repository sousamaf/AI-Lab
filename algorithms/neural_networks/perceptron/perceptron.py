import numpy as np

# Função de ativação (função degrau)
def step_function(x):
    return np.where(x >= 0, 1, 0)

# Classe do Perceptron
class Perceptron:
    def __init__(self, input_size, learning_rate=1.0, epochs=10):
        self.weights = np.zeros(input_size + 1)  # +1 para o bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict(self, x):
        x_with_bias = np.insert(x, 0, 1)  # Adiciona o bias
        weighted_sum = np.dot(self.weights, x_with_bias)
        return step_function(weighted_sum)

    def fit(self, X, y):
        for epoch in range(self.epochs):
            for xi, yi in zip(X, y):
                xi_with_bias = np.insert(xi, 0, 1)  # Adiciona o bias
                output = self.predict(xi)
                error = yi - output
                self.weights += self.learning_rate * error * xi_with_bias
