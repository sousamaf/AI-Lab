# Redes Neurais Multicamadas (MLP)

Este diretório contém implementações de **Redes Neurais Multicamadas (MLP)**, capazes de resolver problemas mais complexos como a função XOR.

## 📂 Conteúdo

- `mlp_simples.py`: Implementação de uma MLP usando apenas NumPy.
- `mlp_torch_simples.py`: Implementação de uma MLP usando PyTorch.
- `mlp_torch_iris.py`: Implementação de uma MLP usando PyTorch para o dataset Iris.
- `mlp_torch_avaliacao.py`: Implementação de uma MLP usando PyTorch para o dataset Iris com métricas de avaliação.
- `README.md`: Este arquivo com orientações.

## 📖 Descrição

As MLPs são compostas por uma camada de entrada, uma ou mais camadas ocultas e uma camada de saída. Utilizam funções de ativação não lineares e aprendem por meio do algoritmo de retropropagação.

## 🚀 Como Executar

1. **Pré-requisitos**:

   - Python 3.x
   - `numpy` para `mlp_simples.py`
   - `torch` para `mlp_torch_simples.py`

2. **Execução com NumPy**:

   ```bash
   python mlp_simples.py
   ```
3. **Execução com PyTorch**:
   ```bash
   python mlp_torch_simples.py
   ```
4. **Execução com PyTorch - Dataset Iris**:
   ```bash
   python mlp_torch_iris.py
   ```

5. **Execução com PyTorch - Dataset Iris Validação**:
   ```bash
   python mlp_torch_avaliacao.py
   ```

🧪 Exemplos

Os scripts incluem exemplos que treinam a MLP para aprender a função XOR ou realizar a tarefa de classificação do dataset Iris.