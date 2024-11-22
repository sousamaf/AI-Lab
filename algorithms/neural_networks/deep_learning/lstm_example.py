import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Texto de exemplo para treinamento
texto = "python eh legal"

# Criação de um conjunto de caracteres únicos no texto
chars = tuple(set(texto))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}

# Função para criar sequências de entrada e saída
def criar_sequencias(texto, seq_length):
    X = []
    Y = []
    for i in range(len(texto) - seq_length):
        seq = texto[i:i+seq_length]
        target = texto[i+seq_length]
        X.append([char2int[ch] for ch in seq])
        Y.append(char2int[target])
    return X, Y

# Parâmetros
seq_length = 4  # Comprimento da sequência de entrada
X, Y = criar_sequencias(texto, seq_length)

# Conversão para tensores PyTorch
X = torch.tensor(X)
Y = torch.tensor(Y)

# Definição do modelo LSTM
class ModeloLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModeloLSTM, self).__init__()
        self.hidden_size = hidden_size
        # Embedding para representar caracteres
        self.embedding = nn.Embedding(input_size, hidden_size)
        # Camada LSTM
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        # Camada linear de saída
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # Passa pela camada de embedding
        x = self.embedding(x)
        # Transposição para correspondência com (seq_len, batch, input_size)
        x = x.view(len(x), 1, -1)
        # Passa pela camada LSTM
        out, hidden = self.lstm(x, hidden)
        # Pega a saída do último passo de tempo
        out = out[-1]
        # Passa pela camada totalmente conectada
        out = self.fc(out)
        return out, hidden

    def init_hidden(self):
        # Inicializa os estados ocultos com zeros
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))

# Instanciação do modelo, definição da perda e do otimizador
n_chars = len(chars)
hidden_size = 12
modelo = ModeloLSTM(n_chars, hidden_size, n_chars)
criterio = nn.CrossEntropyLoss()
otimizador = optim.Adam(modelo.parameters(), lr=0.01)

# Treinamento do modelo
n_epochs = 500
for epoch in range(1, n_epochs + 1):
    loss_total = 0
    modelo.zero_grad()
    # Loop pelas sequências
    for i in range(X.size(0)):
        seq_in = X[i]
        seq_target = Y[i]

        # Inicializa os estados ocultos
        hidden = modelo.init_hidden()

        # Forward
        out, hidden = modelo(seq_in, hidden)
        # Ajusta as dimensões para correspondência com o criterio
        loss = criterio(out.view(1, -1), seq_target.view(1))
        loss_total += loss

    # Backward e otimização
    loss_total.backward()
    otimizador.step()

    # Impressão do progresso a cada 40 épocas
    if epoch % 40 == 0:
        print(f'Época: {epoch}/{n_epochs}... Loss: {loss_total.item()/X.size(0)}')

# Função para prever o próximo caractere com amostragem estocástica
def prever(modelo, char, hidden=None, temperatura=1.0):
    # Conversão do caractere para índice
    char_tensor = torch.tensor([char2int[char]])
    with torch.no_grad():
        out, hidden = modelo(char_tensor, hidden)
    # Remove a dimensão extra
    out = out.squeeze()
    # Aplica a temperatura
    prob = nn.functional.softmax(out / temperatura, dim=0).data
    # Converte para numpy para usar np.random.choice
    prob = prob.cpu().numpy()
    # Amostra um índice com base na distribuição de probabilidades
    char_ind = np.random.choice(len(chars), p=prob)
    return int2char[char_ind], hidden

# Geração de texto a partir do modelo treinado
modelo.eval()
primeiro_char = 'p'
resultado = primeiro_char
hidden = modelo.init_hidden()
char, hidden = prever(modelo, primeiro_char, hidden, temperatura=0.8)
resultado += char

# Gera os próximos caracteres
for _ in range(10):
    char, hidden = prever(modelo, char, hidden, temperatura=0.8)
    resultado += char

print("Texto gerado:", resultado)
