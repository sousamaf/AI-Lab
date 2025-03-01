# Street View House Numbers (SVHN): Este dataset contém 
# imagens de números de casas obtidas a partir de dados 
# do Google Street View. É um bom exemplo de aplicação 
# prática relacionada ao reconhecimento de caracteres no 
# mundo real. Ele é um pouco mais complexo que o MNIST, 
# mas ainda gerenciável com redes profundas simples e 
# sem convoluções profundas​.

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Configurações de hiperparâmetros
batch_size = 64
num_epochs = 20
use_gpu = True  # Alterar para True para usar GPU (se disponível)

# Configurar dispositivo (CPU, CUDA ou GPU Apple Silicon)
if use_gpu:
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Utiliza CUDA se disponível
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Utiliza GPU MPS para Apple Silicon (M1/M2)
    else:
        device = torch.device("cpu")  # Se nenhuma GPU estiver disponível, utiliza CPU
else:
    device = torch.device("cpu")  # Se o uso de GPU não for ativado, utiliza CPU

# print(f"Usando dispositivo: {device}")


def main():
    # Transformações de pré-processamento para o SVHN
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Corrigido para incluir os desvios padrão
    ])

    # Carregar o dataset SVHN (conjunto de treinamento e teste)
    trainset = torchvision.datasets.SVHN(root='./data', split='train',
                                         download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.SVHN(root='./data', split='test',
                                        download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    # Definir a arquitetura da rede neural
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(32*32*3, 512)  # Adaptado para imagens SVHN
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, 10)  # 10 classes (números de 0 a 9)

        def forward(self, x):
            x = x.view(-1, 32*32*3)  # Achatar a imagem
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)
            return x

    # Instanciar o modelo, função de perda e otimizador
    model = SimpleNet().to(device)  # Mover o modelo para o dispositivo
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Loop de treinamento
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)  # Mover dados para o dispositivo
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Época {epoch+1}/{num_epochs}, Perda: {running_loss/len(trainloader):.4f}')

    # Avaliação no conjunto de teste
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Acurácia no conjunto de teste: {100 * correct / total:.2f}%')
    # Salvar o modelo treinado após o loop de treinamento
    # Incluir a acurracia no nome do arquivo
    # formatar a performance para 2 casas decimais e substituir o ponto por underline
    accuracy = 100 * correct / total
    accuracy_str = f"{accuracy:.2f}".replace(".", "_")
    torch.save(model.state_dict(), f"./models/simple_net_svhn_{accuracy_str}.pth")
    # torch.save(model.state_dict(), "./models/simple_net_svhn.pth")

if __name__ == "__main__":
    # Time the execution of the main function
    import time
    start_time = time.time()
    
    main()

    # end time
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")