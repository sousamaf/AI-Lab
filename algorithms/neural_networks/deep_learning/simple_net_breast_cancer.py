import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  # Importação adicionada

# Carregar o dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Dividir em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Converter para tensores do PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Definir o modelo
class SimpleNet(nn.Module):
    def __init__(self, input_size):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 16)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.relu1(self.layer1(x))
        out = self.relu2(self.layer2(out))
        out = self.sigmoid(self.layer3(out))
        return out

input_size = X_train.shape[1]
model = SimpleNet(input_size)

# Definir função de perda e otimizador
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Listas para armazenar perda e acurácia
train_losses = []
train_accuracies = []

# Loop de treinamento
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward e otimização
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Armazenar perda
    train_losses.append(loss.item())
    
    # Calcular acurácia no conjunto de treinamento
    with torch.no_grad():
        predicted = (outputs >= 0.5).float()
        accuracy = (predicted.eq(y_train).sum() / y_train.shape[0]).item()
        train_accuracies.append(accuracy)
    
    # Imprimir perda e acurácia
    if (epoch+1) % 10 == 0:
        print(f'Época [{epoch+1}/{num_epochs}], Perda: {loss.item():.4f}, Acurácia: {accuracy * 100:.2f}%')

# Avaliação no conjunto de teste
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    predicted = (outputs >= 0.5).float()
    test_accuracy = (predicted.eq(y_test).sum() / y_test.shape[0]).item()
    print(f'\nAcurácia no conjunto de teste: {test_accuracy * 100:.2f}%')

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
