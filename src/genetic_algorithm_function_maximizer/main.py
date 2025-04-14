from GeneticAlgorithm import GeneticAlgorithm

if __name__ == "__main__":
    ga = GeneticAlgorithm(population_size=100, chromosome_length=2, mutation_rate=0.01, crossover_rate=0.8, elitism_count=10)
    print(ga.roulette_selection())
