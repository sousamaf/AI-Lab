from perceptron import Perceptron, step_function
import numpy as np

# --------- DADOS DE ENTRADA E SAÍDAS ESPERADAS ---------
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

# Saídas desejadas
y_or  = np.array([0, 1, 1, 1])  # OR
y_and = np.array([0, 0, 0, 1])  # AND
y_xor = np.array([0, 1, 1, 0])  # XOR (resultado final desejado)

# --------- TREINAMENTO DOS PERCEPTRONS OR e AND ---------
p_or = Perceptron(input_size=2, learning_rate=0.1, epochs=10)
p_or.fit(X, y_or)

p_and = Perceptron(input_size=2, learning_rate=0.1, epochs=10)
p_and.fit(X, y_and)

# --------- ENTRADAS INTERMEDIÁRIAS PARA XOR ---------
# Cada entrada será [saida_or, not(saida_and)]
intermediate = []
for xi in X:
    or_out = p_or.predict(xi)
    and_out = p_and.predict(xi)
    xor_input = np.array([or_out, 1 - and_out])  # NOT AND
    intermediate.append(xor_input)

intermediate = np.array(intermediate)

# --------- TREINAMENTO DO PERCEPTRON XOR FINAL ---------
p_xor = Perceptron(input_size=2, learning_rate=0.1, epochs=10)
p_xor.fit(intermediate, y_xor)

# --------- TESTE FINAL DA COMPOSIÇÃO ---------
print("Teste do sistema composto de Perceptrons para XOR:\n")
for xi in X:
    or_out = p_or.predict(xi)
    and_out = p_and.predict(xi)
    xor_input = np.array([or_out, 1 - and_out])
    xor_out = p_xor.predict(xor_input)
    print(f"Entrada: {xi}, Saída XOR prevista: {xor_out}")