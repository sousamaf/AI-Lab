import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset
digits = load_digits()
X = digits.images  # Imagens de dígitos (8x8 pixels)
y = digits.target  # Rótulos (0 a 9)

# Visualizar algumas imagens
plt.figure(figsize=(8, 4))
for index, (image, label) in enumerate(zip(X[:8], y[:8])):
    plt.subplot(2, 4, index + 1)
    plt.imshow(image, cmap='gray') 
    plt.title(f'Dígito: {label}')
    plt.axis('off')
plt.show()

# Pré-processamento
n_samples = len(X)
X = X.reshape((n_samples, -1))  # Achatar as imagens para vetores de 64 elementos

# Dividir em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=33
)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Converter para tensores do PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Definir o modelo
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(64, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 10)  # 10 classes (dígitos de 0 a 9)
    
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()

# Definir função de perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Listas para armazenar perda e acurácia
train_losses = []
train_accuracies = []

# Loop de treinamento
num_epochs = 150
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    
    # Calcular acurácia no conjunto de treinamento
    with torch.no_grad():
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_train).float().mean().item()
        train_accuracies.append(accuracy)
    
    if (epoch + 1) % 10 == 0:
        print(f'Época [{epoch+1}/{num_epochs}], Perda: {loss.item():.4f}, Acurácia: {accuracy * 100:.2f}%')

# Avaliação no conjunto de teste
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    loss = criterion(outputs, y_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test).float().mean().item()
    print(f'\nAcurácia no conjunto de teste: {accuracy * 100:.2f}%')

# Plotar perda e acurácia
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(12, 5))

# Gráfico da perda
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'b-', label='Perda de Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.title('Perda durante o Treinamento')
plt.legend()

# Gráfico da acurácia
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, 'g-', label='Acurácia de Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.title('Acurácia durante o Treinamento')
plt.legend()

plt.tight_layout()
plt.show()

# Extraindo pesos da primeira camada para visualização
weights_fc1 = model.fc1.weight.detach().numpy()

# Criando o heatmap dos pesos da primeira camada
plt.figure(figsize=(10, 8))
sns.heatmap(weights_fc1, cmap="coolwarm", annot=False)
plt.title("Heatmap dos Pesos - Primeira Camada (fc1)")
plt.xlabel("Neurônios de Entrada")
plt.ylabel("Neurônios de Saída")
plt.show()

# Extraindo pesos da segunda camada para visualização
weights_fc2 = model.fc2.weight.detach().numpy()

# Criando o heatmap dos pesos da segunda camada
plt.figure(figsize=(10, 8))
sns.heatmap(weights_fc2, cmap="coolwarm", annot=False)
plt.title("Heatmap dos Pesos - Segunda Camada (fc2)")
plt.xlabel("Neurônios de Entrada")
plt.ylabel("Neurônios de Saída")
plt.show()

# Extraindo pesos da terceira camada para visualização
weights_fc3 = model.fc3.weight.detach().numpy()

# Criando o heatmap dos pesos da terceira camada
plt.figure(figsize=(10, 8))
sns.heatmap(weights_fc3, cmap="coolwarm", annot=False)
plt.title("Heatmap dos Pesos - Terceira Camada (fc3)")
plt.xlabel("Neurônios de Entrada")
plt.ylabel("Neurônios de Saída")
plt.show()
