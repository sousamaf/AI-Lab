import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

# Carregamento dos dados
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Pré-processamento
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Divisão em treinamento, validação e teste
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=33)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=33)

# Definição do modelo
class IrisModel(nn.Module):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instancia o modelo
model = IrisModel()

# Função de perda e otimizador com regularização L2
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

# Inicialização das listas para armazenar as perdas e acurácias
train_losses = []
val_losses = []
val_accuracies = []

# Treinamento
epochs = 1000
for epoch in range(epochs):
    model.train()
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass e otimização
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Armazena a perda de treinamento
    train_losses.append(loss.item())

    # Validação
    model.eval()
    with torch.no_grad():
        outputs_val = model(X_val)
        loss_val = criterion(outputs_val, y_val)
        val_losses.append(loss_val.item())

        _, predicted = torch.max(outputs_val, 1)
        accuracy = (predicted == y_val).sum().item() / y_val.size(0)
        val_accuracies.append(accuracy)

    if (epoch+1) % 100 == 0:
        print(f'Época [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Loss Validação: {loss_val.item():.4f}, Acurácia Validação: {accuracy * 100:.2f}%')

# Plotando as perdas
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Perda de Treinamento')
plt.plot(val_losses, label='Perda de Validação')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.show()

# Plotando a acurácia de validação
plt.figure(figsize=(10,5))
plt.plot(val_accuracies, label='Acurácia de Validação')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()

# Teste
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    y_true = y_test.cpu().numpy()
    y_pred = predicted.cpu().numpy()

    # Cálculo das métricas
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f'Precisão: {precision:.2f}, Revocação: {recall:.2f}, F1-Score: {f1:.2f}')

    # Matriz de Confusão
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
    disp.plot()
    plt.show()
