import torch
import torch.nn as nn

# Exemplo de valor verdadeiro e previsão do modelo
target = torch.tensor([1.0])  # O valor verdadeiro é 1
prediction = torch.tensor([0.9])  # O modelo prevê 0.9

# Criando a função de perda
loss_fn = nn.BCELoss()

# Calculando a perda
loss = loss_fn(prediction, target)

print(f'Perda: {loss.item()}')
