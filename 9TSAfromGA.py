import numpy as np
import random

# Create the distance matrix
def create_distance_matrix(cities):
    num_cities = len(cities)
    distances = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            distances[i, j] = np.linalg.norm(cities[i] - cities[j])
    return distances

# Fitness function: inverse of total distance
def calculate_fitness(route, distance_matrix):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i], route[i + 1]]
    total_distance += distance_matrix[route[-1], route[0]]  # Return to start
    # If total distance is zero (which shouldn't happen in a real case), return a very small fitness value
    if total_distance == 0:
        return float('inf')
    return 1 / total_distance

# Mutation: Swap two cities
def mutate(route):
    i, j = random.sample(range(len(route)), 2)
    route[i], route[j] = route[j], route[i]

# Crossover: Partially Mapped Crossover (PMX)
def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[start:end] = parent1[start:end]
    
    for i in range(start, end):
        if parent2[i] not in child:
            for j in range(size):
                if child[j] == -1 and parent2[j] not in parent1[start:end]:
                    child[j] = parent2[i]
                    break
    for i in range(size):
        if child[i] == -1:
            child[i] = parent2[i]
    return child

# Genetic Algorithm
def genetic_algorithm(cities, population_size=50, generations=100, mutation_rate=0.2):
    distance_matrix = create_distance_matrix(cities)
    num_cities = len(cities)

    # Initialize random population
    population = [random.sample(range(num_cities), num_cities) for _ in range(population_size)]

    for generation in range(generations):
        # Calculate fitness for each individual
        fitness = [calculate_fitness(route, distance_matrix) for route in population]

        # Handle infinite or invalid fitness values
        if any(f == float('inf') for f in fitness):
            print("Warning: Infinite fitness detected in generation", generation)
        
        # Normalize fitness values to probabilities (avoid NaN or inf)
        total_fitness = sum(f for f in fitness if f != float('inf'))
        if total_fitness == 0:
            probabilities = [0] * population_size
        else:
            probabilities = [f / total_fitness if f != float('inf') else 0 for f in fitness]

        # Create new population using selection, crossover, and mutation
        new_population = []
        for _ in range(population_size):
            parent1, parent2 = random.choices(population, weights=probabilities, k=2)
            child = crossover(parent1, parent2)
            if random.random() < mutation_rate:
                mutate(child)
            new_population.append(child)

        population = new_population

    # Return the best route
    best_route_index = np.argmax([calculate_fitness(route, distance_matrix) for route in population])
    best_route = population[best_route_index]
    return best_route, 1 / calculate_fitness(best_route, distance_matrix)

# Example usage
cities = np.array([[0, 0], [2, 3], [5, 2], [7, 5], [6, 7]])  # Coordinates of cities
best_route, best_distance = genetic_algorithm(cities)

print("Best Route:", best_route)
print("Best Distance:", best_distance)
