from GeneticAlgorithm import GeneticAlgorithm

if __name__ == "__main__":

    bounds = [(-3.1, 12.1), (4.1, 5.8)]
    precision = 2

    max_known_value = 38.85

    ga = GeneticAlgorithm(
        population_size=30, 
        bounds=bounds, 
        mutation_rate=0.9, 
        crossover_rate=0.8,
        crossover_type="single_point",
        max_known_value=max_known_value
    )

    best_individual, best_fitness = ga.run(generations=3)
    print("best_individual: ", best_individual)
    print("best_fitness: ", best_fitness)

