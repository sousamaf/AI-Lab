# -*- coding: utf-8 -*-
"""
Algoritmo Genético
"""
import numpy as np
import matplotlib.pyplot as plt

class AlgoritmoGenetico:
    def __init__(self, TAM_POP, TAM_GENE, numero_geracoes=100):
        # print("Algoritmo Genético. Executado por Marco.")
        self.TAM_POP = TAM_POP
        self.TAM_GENE = TAM_GENE
        self.POP = []
        self.POP_AUX = []
        self.aptidao = []
        self.numero_geracoes = numero_geracoes
        self.melhor_aptidoes = []
        self.carga_maxima = 6
        self.lambda_books = 0.1
        self.livros = []  # Essa lista deve ser atribuída externamente (app.py)
        self.populacao_inicial()
    
    def populacao_inicial(self):
        # print("Criando população inicial!")
        # Exemplo: baixa probabilidade de selecionar um livro (1) para favorecer viabilidade
        for i in range(self.TAM_POP):
            self.POP.append(np.random.choice([0, 1], size=self.TAM_GENE, p=[0.99, 0.01]))
    
    def seleciona_individuo(self, k=3):
        """
        Seleciona um indivíduo para reprodução.
        Se a soma das aptidões for zero, utiliza seleção por torneio.
        Caso contrário, utiliza a seleção por roleta.
        """
        aptidao_total = sum(self.aptidao)
        if aptidao_total == 0:
            return self.torneio(k)
        else:
            aptidao_perc = [(apt * 100) / aptidao_total for apt in self.aptidao]
            sorteado = np.random.uniform(0.1, 100.1)
            acumulado = 0.0
            for i, p in enumerate(aptidao_perc):
                acumulado += p
                if acumulado > sorteado:
                    return i
            return 0
    
    def torneio(self, k=3):
        """
        Seleção por torneio: seleciona k indivíduos aleatoriamente e retorna
        o índice do indivíduo com a maior aptidão.
        """
        indices = np.random.choice(range(self.TAM_POP), k, replace=False)
        best_index = indices[0]
        best_fitness = self.aptidao[best_index]
        for idx in indices:
            if self.aptidao[idx] > best_fitness:
                best_index = idx
                best_fitness = self.aptidao[idx]
        return best_index
    
    def operadores_geneticos(self):
        tx_cruzamento_simples = 30
        tx_cruzamento_uniforme = 70
        tx_mutacao = 2
        
        for geracao in range(self.numero_geracoes):
            self.POP_AUX = []
            self.avaliacao()
            q, apt = self.pegar_melhor_individuo()
            self.melhor_aptidoes.append(apt)
            self.exibe_melhor_individuo(geracao)
            
            ## Cruzamento simples
            qtd = (self.TAM_POP * tx_cruzamento_simples) / 100
            for i in range(int(qtd)):
                pai1 = self.seleciona_individuo(k=3)
                pai2 = self.seleciona_individuo(k=3)
                while pai1 == pai2:
                    pai2 = self.seleciona_individuo(k=3)
                self.cruzamento_simples(pai1, pai2)
            
            ## Cruzamento uniforme
            qtd = (self.TAM_POP * tx_cruzamento_uniforme) / 100
            for i in range(int(qtd)):
                pai1 = self.seleciona_individuo(k=3)
                pai2 = self.seleciona_individuo(k=3)
                while pai1 == pai2:
                    pai2 = self.seleciona_individuo(k=3)
                self.cruzamento_uniforme(pai1, pai2)
            
            ## Mutação
            qtd = (self.TAM_POP * tx_mutacao) / 100
            for i in range(int(qtd)):
                quem = np.random.randint(0, self.TAM_POP)
                self.mutacao(quem)
            
            self.substituicao()
        
        self.exibe_melhor_individuo(geracao)
        # Retorna o melhor indivíduo encontrado (vetor binário)
        melhor_index, _ = self.pegar_melhor_individuo()
        return self.POP[melhor_index]
    
    def cruzamento_simples(self, pai1, pai2):
        desc1 = np.zeros(self.TAM_GENE, dtype=int)
        desc2 = np.zeros(self.TAM_GENE, dtype=int)
        for i in range(self.TAM_GENE):
            if i < self.TAM_GENE / 2:
                desc1[i] = self.POP[pai1][i]
                desc2[i] = self.POP[pai2][i]
            else:
                desc1[i] = self.POP[pai2][i]
                desc2[i] = self.POP[pai1][i]
        self.POP_AUX.append(desc1)
        self.POP_AUX.append(desc2)
                
    def cruzamento_uniforme(self, pai1, pai2):
        desc1 = np.zeros(self.TAM_GENE, dtype=int)
        desc2 = np.zeros(self.TAM_GENE, dtype=int)
        for i in range(self.TAM_GENE):
            if np.random.randint(0, 2) == 0:
                desc1[i] = self.POP[pai1][i]
                desc2[i] = self.POP[pai2][i]
            else:
                desc1[i] = self.POP[pai2][i]
                desc2[i] = self.POP[pai1][i]
        self.POP_AUX.append(desc1)
        self.POP_AUX.append(desc2)
    
    def mutacao(self, i):
        g = np.random.randint(0, self.TAM_GENE)
        self.POP_AUX[i][g] = 1 - self.POP_AUX[i][g]
    
    def elitismo(self, qtd):
        aptidao_index = []
        for i in range(self.TAM_POP):
            aptidao_index.append([self.aptidao[i], i])
        ord_aptidao = sorted(aptidao_index, key=lambda x: x[0], reverse=True)
        for i in range(int(qtd)):
            eleito = np.zeros(self.TAM_GENE, dtype=int)
            for g in range(self.TAM_GENE):
                eleito[g] = self.POP[ord_aptidao[i][1]][g]
            self.POP_AUX.append(eleito)
    
    def substituicao(self):
        self.POP = self.POP_AUX.copy()
    
    def avaliacao(self):
        # Verifica se a lista de livros está definida e tem o tamanho correto
        if not self.livros or len(self.livros) != self.TAM_GENE:
            raise ValueError("A lista de livros não foi definida ou seu tamanho não coincide com TAM_GENE.")
        
        self.aptidao = []
        
        for i in range(self.TAM_POP):
            total_peso = 0.0
            contagem_livros = 0
            
            # Percorre o cromossomo do indivíduo
            for g in range(self.TAM_GENE):
                if self.POP[i][g] == 1:
                    total_peso += self.livros[g]
                    contagem_livros += 1
            
            epsilon = 1e-6  # valor mínimo para evitar aptidão zero absoluta
            if total_peso > self.carga_maxima:
                fitness = epsilon
            else:
                fitness = total_peso + self.lambda_books * contagem_livros
            
            # print(f"Fitness do indivíduo {i}: {fitness}")
            self.aptidao.append(fitness)
    
    def pegar_melhor_individuo(self):
        apt = max(self.aptidao)
        quem = self.aptidao.index(apt)
        return quem, apt
    
    def exibe_melhor_individuo(self, geracao):
        apt = max(self.aptidao)
        quem = self.aptidao.index(apt)
        # print("Geração: {} | Indivíduo: {} | Aptidão: {}".format(geracao, quem, apt))
    
    def plotar_progresso(self):
        plt.figure(figsize=(10, 5))
        plt.plot(range(self.numero_geracoes), self.melhor_aptidoes, marker='o', linestyle='-', color='b')
        plt.title('Progresso do Melhor Indivíduo ao Longo das Gerações')
        plt.xlabel('Geração')
        plt.ylabel('Aptidão do Melhor Indivíduo')
        plt.grid(True)
        plt.show()