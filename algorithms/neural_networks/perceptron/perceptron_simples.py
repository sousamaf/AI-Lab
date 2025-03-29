from perceptron import Perceptron, step_function
import numpy as np

# Exemplo de uso
if __name__ == "__main__":
    # Dados de exemplo: função lógica AND
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    # y = np.array([0, 0, 0, 1])  # Saídas da função AND
    # y = np.array([0, 1, 1, 1])  # Saídas da função OR
    y = np.array([0, 1, 1, 0])  # Saídas da função XOR

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
