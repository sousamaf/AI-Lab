import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import numpy as np

# Carregamento dos dados
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Pré-processamento
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

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

# Configuração do StratifiedKFold
k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=33)

# Inicialização das listas para armazenar métricas
fold_train_losses = []
fold_val_losses = []
fold_val_accuracies = []
fold_precisions = []
fold_recalls = []
fold_f1_scores = []

# Loop de Validação Cruzada
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f'\nFold {fold+1}/{k_folds}')
    
    # Divisão dos dados
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Instancia um novo modelo
    model = IrisModel()
    
    # Função de perda e otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Listas para armazenar perdas e acurácias deste fold
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
            print(f'Época [{epoch+1}/{epochs}], Loss Treino: {loss.item():.4f}, Loss Validação: {loss_val.item():.4f}, Acurácia Validação: {accuracy * 100:.2f}%')
    
    # Armazena as métricas deste fold
    fold_train_losses.append(train_losses)
    fold_val_losses.append(val_losses)
    fold_val_accuracies.append(val_accuracies)
    
    # Avaliação final neste fold
    with torch.no_grad():
        outputs_val = model(X_val)
        _, predicted = torch.max(outputs_val, 1)
        y_true = y_val.cpu().numpy()
        y_pred = predicted.cpu().numpy()
    
        # Cálculo das métricas
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        print(f'Fold {fold+1} - Precisão: {precision:.2f}, Revocação: {recall:.2f}, F1-Score: {f1:.2f}')
        
        # Armazena as métricas
        fold_precisions.append(precision)
        fold_recalls.append(recall)
        fold_f1_scores.append(f1)

# Resultados médios
print('\nResultados Médios da Validação Cruzada:')
print(f'Precisão Média: {np.mean(fold_precisions):.2f}')
print(f'Revocação Média: {np.mean(fold_recalls):.2f}')
print(f'F1-Score Médio: {np.mean(fold_f1_scores):.2f}')

# Plotando as perdas e acurácias médias
mean_train_losses = np.mean(fold_train_losses, axis=0)
mean_val_losses = np.mean(fold_val_losses, axis=0)
mean_val_accuracies = np.mean(fold_val_accuracies, axis=0)

# Perdas
plt.figure(figsize=(10,5))
plt.plot(mean_train_losses, label='Perda de Treinamento Média')
plt.plot(mean_val_losses, label='Perda de Validação Média')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.show()

# Acurácias
plt.figure(figsize=(10,5))
plt.plot(mean_val_accuracies, label='Acurácia de Validação Média')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()
