import numpy as np
import matplotlib.pyplot as plt

# Definindo as funções de ativação
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# Criando uma sequência de valores de entrada para visualizar as funções
x = np.linspace(-10, 10, 100)

# Calculando os valores de saída para cada função de ativação
relu_values = relu(x)
sigmoid_values = sigmoid(x)
tanh_values = tanh(x)

# Plotando as funções de ativação
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.plot(x, relu_values, label='ReLU', color='blue')
plt.title('ReLU Activation Function')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(x, sigmoid_values, label='Sigmoid', color='orange')
plt.title('Sigmoid Activation Function')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(x, tanh_values, label='Tanh', color='green')
plt.title('Tanh Activation Function')
plt.grid(True)

plt.tight_layout()
plt.show()
