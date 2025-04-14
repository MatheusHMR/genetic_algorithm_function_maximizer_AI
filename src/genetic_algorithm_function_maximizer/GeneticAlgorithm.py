import numpy as np

class GeneticAlgorithm:
    def __init__(self, population_size, chromosome_length, mutation_rate, crossover_rate, elitism_count,
                 selection_method='roulette', tournament_size=None, crossover_type='one_point'):
        """
        Inicializa os parâmetros do algoritmo genético.

        :param population_size: Tamanho da população.
        :param chromosome_length: Comprimento do cromossomo (número de genes).
        :param mutation_rate: Taxa de mutação.
        :param crossover_rate: Taxa de cruzamento.
        :param elitism_count: Número de indivíduos a serem selecionados para a próxima geração.
        :param selection_method: Método de seleção (roulette ou tournament).
        :param tournament_size: Tamanho do torneio (se selection_method for tournament).
        :param crossover_type: Tipo de cruzamento (one_point ou two_point).
        """

        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.crossover_type = crossover_type
        self.decimal_precision = 3

    def real_function(self, x1, x2):
        """
        Função real a ser maximizada.
        """
        return 21.5 + x1 * np.sin(4 * np.pi * x1) + x2 * np.sin(20 * np.pi * x2)

    def initialize_population(self):
        """
        Cria a população inicial de indivíduos.
        """

        x1_min = -3.1
        x1_max = 12.1
        x2_min = 4.1
        x2_max = 5.8

        bounds = [(x1_min, x1_max), (x2_min, x2_max)]  # limites de x1 e x2

        # Inicialização
        population = np.array([
            [np.round(np.random.uniform(low, high), self.decimal_precision) for (low, high) in bounds]
            for _ in range(self.population_size)
        ])

        return population
    
    def decode(self, individual):
        """
        Converte a representação do indivíduo para os valores reais.
        """
        pass

    def fitness(self, individual):
        """
        Calcula a aptidão (fitness) do indivíduo.
        """
        pass

    def selection(self):
        """
        Seleciona os indivíduos para reprodução, com base no método definido.
        """
        pass

    def roulette_selection(self):
        """
        Implementa a seleção por roleta.
        """
        pass

    def tournament_selection(self):
        """
        Implementa a seleção por torneio.
        """
        pass

    def crossover(self, parent1, parent2):
        """
        Realiza o cruzamento entre dois pais (one-point ou two-point).
        """
        pass

    def mutation(self, individual):
        """
        Aplica a mutação no indivíduo.
        """
        pass

    def run(self, generations):
        """
        Executa o algoritmo genético por um número definido de gerações.
        """
        pass
