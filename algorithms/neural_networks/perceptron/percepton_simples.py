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

# Exemplo de uso
if __name__ == "__main__":
    # Dados de exemplo: função lógica AND
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    y = np.array([0, 0, 0, 1])  # Saídas da função AND

    # Cria o perceptron
    perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=10)

    # Treina o perceptron
    perceptron.fit(X, y)

    # Testa o perceptron
    for xi in X:
        output = perceptron.predict(xi)
        print(f"Entrada: {xi}, Saída Prevista: {output}")

    # Exibe os pesos finais
    print(f"Pesos finais: {perceptron.weights}")
