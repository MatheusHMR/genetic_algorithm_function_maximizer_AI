from GeneticAlgorithm import GeneticAlgorithm

if __name__ == "__main__":

    bounds = [(-3.1, 12.1), (4.1, 5.8)]
    precision = 4

    max_known_value = 38.85

    ga = GeneticAlgorithm(
        population_size=100, 
        mutation_rate=0.2, 
        crossover_rate=0.85,
        crossover_type="single_point",
        max_known_value=max_known_value,
        decimal_precision=precision,
        elitism_count= 2,
        selection_method="tournament",
        tournament_size=5,
    )

    best_individual, best_fitness = ga.run(generations=10)
    print("best_individual: ", best_individual)
    print("best_fitness: ", best_fitness)

