import os
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from AlgoritmoGenetico import AlgoritmoGenetico as AG

NUM_LIVROS_INICIAL = 1000
LIMIAR_PEQUENO = 3 # Limiar para transporte direto dos últimos livros
arquivo_livros = 'livros.csv'

# Carrega a lista de livros a partir do arquivo CSV ou gera se o arquivo não existir
if os.path.exists(arquivo_livros):
    with open(arquivo_livros, 'r') as f:
        conteudo = f.read().strip()
    if conteudo:
        # Supondo que os livros estejam separados por vírgula em uma única linha
        livros_iniciais = list(map(float, conteudo.split(',')))
    else:
        livros_iniciais = np.random.uniform(0.1, 2.0, NUM_LIVROS_INICIAL).tolist()
        with open(arquivo_livros, 'w') as f:
            f.write(','.join(map(str, livros_iniciais)))
else:
    livros_iniciais = np.random.uniform(0.1, 2.0, NUM_LIVROS_INICIAL).tolist()
    with open(arquivo_livros, 'w') as f:
        f.write(','.join(map(str, livros_iniciais)))

# Parâmetros da simulação
NUM_SIMULACOES = 5

def executa_simulacao(simulacao_id, livros_iniciais):
    """Executa uma simulação e retorna o número de viagens necessárias."""
    livros_origem = livros_iniciais.copy()  # Cópia para preservar a lista original
    livros_destino = []
    num_viagens = 0

    print(f"Simulação {simulacao_id} iniciada.")
    # Enquanto ainda houver livros na origem, faça uma viagem
    while livros_origem:
        # Se restarem poucos livros, transporte-os de uma só vez
        if len(livros_origem) <= LIMIAR_PEQUENO:  # Por exemplo, LIMIAR_PEQUENO = 3
            livros_destino.extend(livros_origem)
            livros_origem.clear()
            num_viagens += 1
            # print(f"Viagem {num_viagens}: Transporte direto dos últimos livros.")
            break

        TAM_GENE = len(livros_origem)
        ag = AG(TAM_POP=50, TAM_GENE=TAM_GENE, numero_geracoes=100)
        ag.livros = livros_origem
        melhor_individuo = ag.operadores_geneticos()

        indices_transportar = [i for i, gene in enumerate(melhor_individuo) if gene == 1]
        if not indices_transportar:
            indices_transportar = [0]

        for i in sorted(indices_transportar, reverse=True):
            livros_destino.append(livros_origem.pop(i))
        
        num_viagens += 1
        # print(f"Viagem {num_viagens}: {len(indices_transportar)} livro(s) transportado(s). Livros restantes: {len(livros_origem)}")
    print(f"Simulação {simulacao_id} concluída em {num_viagens} viagens.")
    return num_viagens

def main():
    viagens_por_simulacao = []

    # Usando ProcessPoolExecutor para executar simulações em paralelo
    with ProcessPoolExecutor() as executor:
        # Submete cada simulação para execução em paralelo
        futures = [executor.submit(executa_simulacao, simulacao, livros_iniciais)
                   for simulacao in range(1, NUM_SIMULACOES + 1)]
        
        # Coleta os resultados à medida que forem finalizados
        for future in as_completed(futures):
            viagens_por_simulacao.append(future.result())

    # Ordena os resultados (opcional, para que o gráfico seja consistente)
    viagens_por_simulacao.sort()

    # Gera o gráfico de barras com o número de viagens de cada simulação
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, NUM_SIMULACOES + 1), viagens_por_simulacao, color='skyblue')
    plt.xlabel('Número da Simulação')
    plt.ylabel('Número de Viagens')
    plt.title('Viagens Necessárias para Transportar Todos os Livros em Cada Simulação')
    plt.xticks(range(1, NUM_SIMULACOES + 1))
    plt.show()

if __name__ == "__main__":
    main()