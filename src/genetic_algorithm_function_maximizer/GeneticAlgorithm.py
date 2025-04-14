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
        self.population = self.initialize_population()
        self.current_population = self.population
        self.x1_min = -3.1
        self.x1_max = 12.1
        self.x2_min = 4.1
        self.x2_max = 5.8

    def real_function(self, population):
        """
        Função real a ser maximizada.
        """
        return 21.5 + population[:, 0] * np.sin(4 * np.pi * population[:, 0]) + population[:, 1] * np.sin(20 * np.pi * population[:, 1])

    def initialize_population(self):
        """
        Cria a população inicial de indivíduos.
        """
        bounds = [(self.x1_min, self.x1_max), (self.x2_min, self.x2_max)]  # limites de x1 e x2

        # Inicialização
        population = np.array([
            [np.round(np.random.uniform(low, high), self.decimal_precision) for (low, high) in bounds]
            for _ in range(self.population_size)
        ])

        return population
    
    def fitness(self, individuals_array):
        """
        Calcula a aptidão (fitness) do indivíduo.
        """
        return self.real_function(individuals_array)

    def selection(self):
        """
        Seleciona os indivíduos para reprodução, com base no método definido.
        """
        # Seleciona o método de seleção
        if self.selection_method == 'roulette':
            # Seleciona os indivíduos para reprodução
            return self.roulette_selection()
        elif self.selection_method == 'tournament':
            # Seleciona os indivíduos para reprodução
            return self.tournament_selection()

    def roulette_selection(self):
        """
        Implementa a seleção por roleta.
        """
        # Calcula a aptidão de cada indivíduo
        fitness_values = self.fitness(self.current_population)
        # Calcula a probabilidade de cada indivíduo
        probabilities = fitness_values / np.sum(fitness_values)
        # Seleciona os indivíduos para reprodução
        selected_individuals = np.random.choice(len(self.current_population), size=self.population_size, p=probabilities)
        # Retorna os indivíduos selecionados
        return self.current_population[selected_individuals]

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
