# Conjuntos de Dados

Este diretório contém os conjuntos de dados utilizados pelos algoritmos neste repositório.

## Conteúdo

- `iris_dataset.csv`: Conjunto de dados Iris.
- `mnist/`: Diretório para o conjunto de dados MNIST.
- `dataset-buonopreco-registro_de_clientes.csv`: Conjunto de dados simulado de clientes do BuonoPreço.
- `README.md`: Este arquivo com informações sobre os datasets.

## Descrição

- **Iris Dataset**: Usado para problemas de classificação multiclasse.
- **MNIST Dataset**: Conjunto de dados de dígitos manuscritos para tarefas de classificação de imagens.
- **BuonoPreço Dataset**: Conjunto de dados fictício de clientes de supermercado, criado para atividades didáticas.  
  Contém variáveis de comportamento de compra como idade, frequência mensal, gasto médio, uso de promoções, resposta a campanhas de WhatsApp, dia mais frequente de compra e classe de cliente.  
  As classes representam perfis de consumo:  
  - *Fiel Econômico*  
  - *Familiar Mensalista*  
  - *Reativo a Promoções*  
  - *Alto Valor Ticket Médio*  

## Como Utilizar

- Os scripts nos diretórios correspondentes geralmente carregam os conjuntos de dados automaticamente.  
- Para o dataset do BuonoPreço, os notebooks disponíveis no repositório demonstram como realizar pré-processamento, treino de modelos (ex.: SVM) e visualizações.  

## ⚠️ Observação

- Alguns conjuntos de dados podem ser grandes e não estão incluídos no repositório. Nesses casos, os scripts fornecem funções para baixá-los automaticamente.