import numpy as np
import math
import random

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, crossover_rate, elitism_count = None, min_known_value = None,
                 selection_method='roulette', tournament_size=None, crossover_type='radcliff', decimal_precision=1):
        """
        Inicializa os parâmetros do algoritmo genético.

        :param population_size: Tamanho da população.
        :param mutation_rate: Taxa de mutação.
        :param crossover_rate: Taxa de cruzamento.
        :param elitism_count: Número de indivíduos a serem selecionados para a próxima geração.
        :param selection_method: Método de seleção (roulette ou tournament).
        :param tournament_size: Tamanho do torneio (se selection_method for tournament).
        :param crossover_type: Tipo de cruzamento (single_point ou double_point).
        :param max_known_value: Valor máximo conhecido da função, nem sempre é conhecido.
        """

        self.population_size = population_size
        self.bounds = [(-3.1, 12.1), (4.1, 5.8)]
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.crossover_type = crossover_type
        self.min_known_value = min_known_value
        self.best_individual = None
        self.best_fitness = None
        self.current_error = np.inf if min_known_value is not None else None
        self.mean_error = np.inf if min_known_value is not None else None
        self.decimal_precision = decimal_precision
        self.current_population = None
        self.stop = None # Callback para parar o algoritmo

    def real_function(self, population=None):
        """
        Função real a ser maximizada.
        """
        if population is None:
            population = self.current_population
        
        return 21.5 + population[:, 0] * np.sin(4 * np.pi * population[:, 0]) + population[:, 1] * np.sin(20 * np.pi * population[:, 1])

    def initialize_population_with_random_values(self):
        """
        Inicializa a população com valores aleatórios usando a fórmula x = a + c*(b - a),
        onde c é um valor aleatório entre 0 e 1.
        """
        return np.array([
            [low + np.random.random() * (high - low) for (low, high) in self.bounds]
            for _ in range(self.population_size)
        ])

    def fitness(self):
        """
        Calcula a aptidão (fitness) da população.
        Atualiza o melhor indivíduo, o erro do melhor e o erro médio da população.
        """
        fitness_values = -self.real_function()
        # Índice do melhor indivíduo
        best_idx = np.argmax(fitness_values)
        self.best_individual = self.current_population[best_idx]
        self.best_fitness = fitness_values[best_idx]

        if self.min_known_value is not None:
            # Erro do melhor indivíduo
            self.current_error = np.abs(self.min_known_value + self.best_fitness)
            # print(f"Erro do melhor indivíduo: {self.current_error}")
            # Erro médio da população
            self.mean_error = np.mean(np.abs(self.min_known_value + fitness_values))
            # print(f"Erro médio da população: {self.mean_error}")
        else:
            self.current_error = None
            self.mean_error = None

        return fitness_values
    

    def radcliff_crossover(self, parent1, parent2):
        """
        Realiza o cruzamento de Radcliff entre dois indivíduos.
        Usa uma combinação convexa com beta dinâmico para cada variável.
        """
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)
        
        for i in range(parent1.shape[0]):
            # Beta diferente para cada variável
            beta = random.random()
            # Combinação convexa
            child1[i] = beta * parent1[i] + (1 - beta) * parent2[i]
            child2[i] = (1 - beta) * parent1[i] + beta * parent2[i]


        return [child1, child2] 

    def wright_crossover(self, parent1, parent2):
        """
        Realiza o cruzamento de Wright entre dois indivíduos.
        Gera 3 filhos e retorna os 2 melhores, ou mantém os pais se necessário.
        Sempre retorna exatamente 2 indivíduos.
        """
        
        # Gera três filhos diferentes
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent1)
        child3 = np.zeros_like(parent1)
        
        # Primeiro filho: média dos pais
        child1 = 0.5 * (parent1 + parent2)
        
        # Segundo filho: combinação com beta aleatório
        beta = random.random()
        child2 = beta * parent1 + (1 - beta) * parent2
        
        # Terceiro filho: combinação com beta diferente para cada variável
        for i in range(parent1.shape[0]):
            beta = random.random()
            child3[i] = beta * parent1[i] + (1 - beta) * parent2[i]
        
        # Lista para armazenar os filhos válidos
        valid_children = []
        
        # Verifica cada filho e adiciona à lista se estiver dentro dos limites
        if (self.bounds[0][0] <= child1[0] <= self.bounds[0][1] and 
            self.bounds[1][0] <= child1[1] <= self.bounds[1][1]):
            valid_children.append(child1)
            
        if (self.bounds[0][0] <= child2[0] <= self.bounds[0][1] and 
            self.bounds[1][0] <= child2[1] <= self.bounds[1][1]):
            valid_children.append(child2)
            
        if (self.bounds[0][0] <= child3[0] <= self.bounds[0][1] and 
            self.bounds[1][0] <= child3[1] <= self.bounds[1][1]):
            valid_children.append(child3)
        
        # Se não houver filhos válidos, retorna os pais
        if len(valid_children) == 0:
            return np.array([parent1, parent2])
        # Se houver apenas um filho válido, retorna ele e o melhor pai
        elif len(valid_children) == 1:
            parent_fitness = self.real_function(np.array([parent1, parent2]))
            if parent_fitness[0] > parent_fitness[1]:
                return np.array([parent1, valid_children[0]])
            else:
                return np.array([valid_children[0], parent2])
        # Se houver dois filhos válidos, retorna eles
        elif len(valid_children) == 2:
            return np.array(valid_children)
        # Se houver três filhos válidos, retorna os dois melhores
        else:
            valid_children = np.array(valid_children)
            children_fitness = self.real_function(valid_children)
            best_indices = np.argsort(children_fitness)[-2:]
            return valid_children[best_indices]

    def selection(self, fitness_values):
        """
        Seleciona os indivíduos para reprodução, com base no método definido.
        """

        # Seleciona o método de seleção
        if self.selection_method == 'roulette':
            # Seleciona os indivíduos para reprodução
            self.current_population = self.roulette_selection(fitness_values)
        elif self.selection_method == 'tournament':
            # Seleciona os indivíduos para reprodução
            self.current_population = self.tournament_selection(fitness_values)

    def roulette_selection(self, fitness_values):
        """
        Implementa a seleção por roleta.
        """

        # Calcula a probabilidade de cada indivíduo
        probabilities = fitness_values / np.sum(fitness_values)
        # Seleciona os indivíduos para reprodução
        selected_individuals = np.random.choice(len(self.current_population), size=self.population_size, p=probabilities)
        # Retorna os indivíduos selecionados
        return self.current_population[selected_individuals]

    def tournament_selection(self, fitness_values):
        """
        Implementa a seleção por torneio.
        """
        selected = []

        for _ in range(self.population_size):
            # Sorteia os indivíduos aleatórios da população de acordo com o tamanho do torneio
            participants_indices = np.random.choice(len(self.current_population), self.tournament_size, replace=False)
            participants_fitness = fitness_values[participants_indices]
            
            # Seleciona o melhor
            winner_indice = participants_indices[np.argmax(participants_fitness)]
            selected.append(self.current_population[winner_indice])

        return np.array(selected)

    def crossover(self):
        """
        Realiza o cruzamento entre dois pais (one-point ou two-point).
        """
        np.random.shuffle(self.current_population)
        children = []
        n = len(self.current_population)
        i = 0
        while i < n - 1: # Parar antes do último se for ímpar
            parent1 = self.current_population[i]
            parent2 = self.current_population[i+1]
            if random.random() < self.crossover_rate:
                if self.crossover_type == 'radcliff':
                    children.extend(self.radcliff_crossover(parent1, parent2))
                elif self.crossover_type == 'wright':
                    children.extend(self.wright_crossover(parent1, parent2))
                else:
                    raise ValueError("Invalid crossover type")
            else:
                children.extend([parent1, parent2])
            i += 2

        if n % 2 == 1:
            children.append(self.current_population[-1])
        self.current_population = np.array(children)

    def mutation(self):
        """
        Aplica a mutação no indivíduo.
        """

        for idx, individual in enumerate(self.current_population):
            if random.random() < self.mutation_rate:
                i  = random.randint(0, len(individual) - 1)
                individual[i] = self.bounds[i][0] + np.random.random() * (self.bounds[i][1] - self.bounds[i][0])
                self.current_population[idx] = individual
        

    def run(self, generations, update_callback=None):
        """
        Executa o algoritmo genético por um número definido de gerações.
        
        :param generations: Número de gerações a serem executadas.
        :return: O melhor indivíduo encontrado.
        """
        elite_individuals = None
        self.current_population = self.initialize_population_with_random_values()
        #print(f"População inicial: {self.current_population}")
        #print(f"Aptidão da população inicial: {self.fitness()}")
        for _ in range(generations):

            if self.stop and self.stop():
                break

            #print(f"Geração {_ + 1}")
            
            # Calcula a aptidão de cada indivíduo, 
            fitness_values = self.fitness()
            #print(f"Aptidão da população: {fitness_values}")

            # Elitismo: mantém os melhores indivíduos da geração anterior
            if self.elitism_count and self.elitism_count > 0:
                elite_indices = np.argsort(fitness_values)[-self.elitism_count:]
                elite_individuals = self.current_population[elite_indices]

            # Faz a seleção, crossover e mutação
            self.selection(fitness_values)
            self.crossover() 
            self.mutation()

            if elite_individuals is not None:
                new_fitness_values = self.real_function()
                worst_indices = np.argsort(new_fitness_values)[:self.elitism_count]
                for i, idx in enumerate(worst_indices):
                    self.current_population[idx] = elite_individuals[i]

            # Atualiza a população com os melhores indivíduos

            #print(f"População após a mutação e elitismo: {self.current_population}")

            self.fitness()

            if update_callback:
                update_callback(
                    generation=_ + 1,
                    best_individual=self.best_individual,
                    best_fitness=self.best_fitness,
                    error = self.current_error if self.current_error is not None else 0
                )


            if self.min_known_value is not None and self.current_error < 1e-6:
                #print(f"Encerrando o algoritmo, pois o erro é menor que 1e-6")
                break     

        return self.best_individual, self.best_fitness
