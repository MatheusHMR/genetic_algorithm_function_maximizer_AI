from GeneticAlgorithm import GeneticAlgorithm

if __name__ == "__main__":

    bounds = [(-3.1, 12.1), (4.1, 5.8)]
    precision = 2

    max_known_value = 38.85

    ga = GeneticAlgorithm(
        population_size=100, 
        bounds=bounds, 
        mutation_rate=0.5, 
        crossover_rate=0.75, 
        elitism_count=10,
        max_known_value=max_known_value
    )

    best_individual, best_fitness = ga.run(generations=1000)
    print("best_individual: ", best_individual)
    print("best_fitness: ", best_fitness)

