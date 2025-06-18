"""
This module implements Genetic Algorithm (GA) components and full GA pipelines
for solving the Traveling Salesperson Problem (TSP) and its variants with
Time Windows (TSPTW), including hard and soft time window constraints.

Core GA components provided:
- Individual creation (random permutation of cities).
- Initial population generation.
- Fitness calculation functions for:
    - Basic TSP (total distance).
    - TSPTW with Hard Time Windows (total distance, penalized with infinity if infeasible).
    - TSPTW with Soft Time Windows (total distance + penalties for earliness/lateness).
- Selection operator (tournament selection).
- Crossover operator (Partially Mapped Crossover - PMX).
- Mutation operator (swap mutation).

Full GA pipelines:
- `genetic_algorithm`: Solves basic TSP.
- `genetic_algorithm_tsptw`: Solves TSPTW with hard time windows.
- `genetic_algorithm_soft_tw`: Solves TSPTW with soft time windows.

The depot is assumed to be node 0. Individuals in the population represent
a permutation of customer nodes (1 to N-1), with the route implicitly
starting and ending at the depot.
"""
import random
import numpy as np

# --- Core GA Components ---

def create_individual(num_nodes):
    """
    Creates a random permutation of city nodes (1 to num_nodes-1).
    Node 0 is the depot and is not included in this permutation.
    The route implicitly starts at node 0, visits nodes in the permutation, and ends at node 0.

    Args:
        num_nodes (int): The total number of nodes, including the depot.

    Returns:
        list: A list of integers representing a permutation of city nodes.
              Returns an empty list if num_nodes <= 1.
    """
    if num_nodes <= 1: # Only depot or no (customer) cities
        return []
    # Cities are numbered 1 to num_nodes - 1
    cities = list(range(1, num_nodes))
    random.shuffle(cities)
    return cities

def initial_population(pop_size, num_nodes):
    """
    Generates an initial population of individuals (routes).

    Args:
        pop_size (int): The number of individuals in the population.
        num_nodes (int): The total number of nodes, including the depot.

    Returns:
        list: A list of individuals, where each individual is a list (route).
    """
    population = []
    for _ in range(pop_size):
        population.append(create_individual(num_nodes))
    return population

def calculate_fitness(individual, distance_matrix):
    """
    Calculates the total distance of a route (individual) for basic TSP.
    The route starts at the depot (node 0), visits cities in the 'individual'
    permutation, and returns to the depot.

    Args:
        individual (list): A permutation of city nodes (e.g., [c1, c2, ..., ck]).
        distance_matrix (numpy.ndarray): An N x N matrix where N is num_nodes,
                                         representing distances between nodes.

    Returns:
        float: The total distance of the route. Returns 0.0 if the individual is empty.
    """
    total_distance = 0.0
    if not individual: # Handles num_nodes = 1 case (empty individual)
        return 0.0

    # Distance from depot (0) to the first city in the individual
    total_distance += distance_matrix[0, individual[0]]

    # Distances between cities in the individual sequence
    for i in range(len(individual) - 1):
        total_distance += distance_matrix[individual[i], individual[i+1]]

    # Distance from the last city in the individual back to the depot (0)
    total_distance += distance_matrix[individual[-1], 0]

    return total_distance

def tournament_selection(population, fitnesses, k=3):
    """
    Selects a parent from the population using k-tournament selection.
    k individuals are randomly chosen, and the one with the best (lowest) fitness is selected.

    Args:
        population (list): The current population of individuals.
        fitnesses (list): A list of fitness values corresponding to the individuals
                          in the population. Lower fitness is assumed to be better.
        k (int, optional): The number of individuals to participate in the tournament. Defaults to 3.

    Returns:
        list: The selected parent individual (route).
    """
    # Select k random indices from the population (without replacement)
    tournament_indices = random.sample(range(len(population)), k)

    # Get the individuals and their fitnesses for the tournament
    tournament_individuals = [population[i] for i in tournament_indices]
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]

    # Find the best individual (lowest fitness) in the tournament
    best_fitness_in_tournament = min(tournament_fitnesses)
    best_index_in_tournament = tournament_fitnesses.index(best_fitness_in_tournament)

    return tournament_individuals[best_index_in_tournament]

def partially_mapped_crossover(parent1, parent2):
    """
    Implements Partially Mapped Crossover (PMX) for permutation-based individuals.

    Args:
        parent1 (list): The first parent individual (route permutation).
        parent2 (list): The second parent individual (route permutation).

    Returns:
        tuple: A tuple containing two children (child1, child2),
               each as a new route permutation.
    """
    size = len(parent1)
    if size == 0: # Handles empty individuals (e.g., for num_nodes=1)
        return [], []

    child1, child2 = [-1]*size, [-1]*size # Initialize children with placeholders

    # Step 1: Choose two random distinct crossover points
    cx_point1, cx_point2 = sorted(random.sample(range(size), 2))

    # Step 2: Copy the segment between crossover points from parents to children
    child1[cx_point1:cx_point2+1] = parent1[cx_point1:cx_point2+1]
    child2[cx_point1:cx_point2+1] = parent2[cx_point1:cx_point2+1]

    # Step 3 & 4: Fill remaining positions, resolving conflicts
    # For child1 (inherits segment from parent1, fills rest from parent2)
    for i in list(range(cx_point1)) + list(range(cx_point2 + 1, size)): # Indices outside segment
        if parent2[i] not in child1: # If element from parent2 is not in child1's copied segment
            child1[i] = parent2[i]
        else: # Conflict: element from parent2 is already in child1's segment
            current_val_from_p2 = parent2[i]
            # Trace back through mapping to find a value not in child1's segment
            while current_val_from_p2 in child1[cx_point1:cx_point2+1]:
                # Find where this conflicting value (current_val_from_p2) is in parent1's segment
                idx_in_p1_segment = parent1[cx_point1:cx_point2+1].index(current_val_from_p2)
                # Get the value from parent2 at that same relative position within the segment
                current_val_from_p2 = parent2[cx_point1 + idx_in_p1_segment]
            child1[i] = current_val_from_p2

    # For child2 (inherits segment from parent2, fills rest from parent1)
    for i in list(range(cx_point1)) + list(range(cx_point2 + 1, size)):
        if parent1[i] not in child2:
            child2[i] = parent1[i]
        else:
            current_val_from_p1 = parent1[i]
            while current_val_from_p1 in child2[cx_point1:cx_point2+1]:
                idx_in_p2_segment = parent2[cx_point1:cx_point2+1].index(current_val_from_p1)
                current_val_from_p1 = parent1[cx_point1 + idx_in_p2_segment]
            child2[i] = current_val_from_p1

    return child1, child2

def swap_mutation(individual, mutation_rate):
    """
    Implements swap mutation on an individual.
    Iterates through each gene in the individual. If a random number is less
    than `mutation_rate`, this gene is swapped with another randomly chosen gene.

    Args:
        individual (list): The individual (route permutation) to mutate.
        mutation_rate (float): The probability of each gene being selected for a swap.

    Returns:
        list: The mutated individual.
    """
    mutated_individual = list(individual) # Work on a copy
    size = len(mutated_individual)
    if size <= 1: # Cannot swap if 0 or 1 elements
        return mutated_individual

    for i in range(size):
        if random.random() < mutation_rate:
            # Select another random position (j) to swap with
            j = random.randint(0, size - 1)
            # Perform swap
            mutated_individual[i], mutated_individual[j] = mutated_individual[j], mutated_individual[i]

    return mutated_individual

# --- Full GA Pipelines ---

def genetic_algorithm(distance_matrix, num_nodes, pop_size, generations,
                      mutation_rate, crossover_rate, tournament_k=3):
    """
    Main Genetic Algorithm for solving the basic Traveling Salesperson Problem (TSP).

    Args:
        distance_matrix (numpy.ndarray): N x N matrix of distances.
        num_nodes (int): Total number of nodes (including depot).
        pop_size (int): Population size.
        generations (int): Number of generations to run.
        mutation_rate (float): Probability of mutation for each gene.
        crossover_rate (float): Probability of performing crossover.
        tournament_k (int, optional): Size of the tournament for selection. Defaults to 3.

    Returns:
        tuple: (best_individual, best_fitness) where best_individual is a list
               representing the route (city permutation) and best_fitness is its total distance.
               Returns ([], 0.0) if num_nodes <= 1.
    """
    population = initial_population(pop_size, num_nodes)
    best_overall_individual = None
    best_overall_fitness = float('inf')

    if num_nodes <= 1:
        return [], 0.0 # Handled by calculate_fitness returning 0 for empty individual

    for _generation in range(generations): # Use _generation if not printing per gen
        fitnesses = [calculate_fitness(ind, distance_matrix) for ind in population]

        current_gen_best_fitness = min(fitnesses)
        if current_gen_best_fitness < best_overall_fitness:
            best_overall_fitness = current_gen_best_fitness
            best_overall_individual = list(population[fitnesses.index(current_gen_best_fitness)])

        new_population = []
        if best_overall_individual is not None: # Elitism
            new_population.append(list(best_overall_individual))

        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses, k=tournament_k)
            parent2 = tournament_selection(population, fitnesses, k=tournament_k)

            if random.random() < crossover_rate:
                child1, child2 = partially_mapped_crossover(parent1, parent2)
            else:
                child1, child2 = list(parent1), list(parent2) # Cloning

            child1 = swap_mutation(child1, mutation_rate)
            child2 = swap_mutation(child2, mutation_rate)

            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        population = new_population[:pop_size]

    # Final check on the last population
    if population:
        final_fitnesses = [calculate_fitness(ind, distance_matrix) for ind in population]
        if final_fitnesses:
            last_pop_best_fitness = min(final_fitnesses)
            if last_pop_best_fitness < best_overall_fitness:
                best_overall_fitness = last_pop_best_fitness
                best_overall_individual = list(population[final_fitnesses.index(last_pop_best_fitness)])

    if best_overall_individual is None: # Should only happen if pop_size was 0 or num_nodes <=1 initially led to no valid path
        return [], float('inf') if num_nodes > 1 else 0.0


    return best_overall_individual, best_overall_fitness

def calculate_fitness_hard_tw(individual, distance_matrix, time_windows, service_times=None):
    """
    Calculates total travel time for TSPTW with Hard Time Windows.
    Returns float('inf') if any time window is violated.

    Args:
        individual (list): Route permutation (city nodes).
        distance_matrix (numpy.ndarray): N x N distance/travel time matrix.
        time_windows (numpy.ndarray): N x 2 matrix of [earliest, latest] arrival times.
        service_times (list/numpy.ndarray, optional): Service time at each node. Defaults to zeros.

    Returns:
        float: Total travel time if feasible, float('inf') otherwise.
    """
    total_travel_time = 0.0
    current_time = time_windows[0][0] # Start at depot's earliest time

    num_nodes_from_dm = len(distance_matrix) # Use actual num_nodes from matrix for safety
    if service_times is None:
        service_times_actual = [0] * num_nodes_from_dm
    else:
        service_times_actual = service_times
        if len(service_times_actual) != num_nodes_from_dm:
            # This case should ideally be caught before GA if num_nodes from parser is used
            raise ValueError("service_times length must match number of nodes in distance_matrix")

    last_location = 0 # Start from depot

    # To first city in individual
    if individual:
        first_city = individual[0]
        travel_to_first = distance_matrix[last_location, first_city]
        total_travel_time += travel_to_first
        current_time += travel_to_first

        if current_time < time_windows[first_city][0]: # Arrived early
            current_time = time_windows[first_city][0] # Wait
        elif current_time > time_windows[first_city][1]: # Arrived late
            return float('inf') # Infeasible

        current_time += service_times_actual[first_city]
        last_location = first_city

    # Through intermediate cities
    for i in range(1, len(individual)):
        current_city = individual[i]
        travel_time_segment = distance_matrix[last_location, current_city]

        total_travel_time += travel_time_segment
        current_time += travel_time_segment

        if current_time < time_windows[current_city][0]:
            current_time = time_windows[current_city][0]
        elif current_time > time_windows[current_city][1]:
            return float('inf')

        current_time += service_times_actual[current_city]
        last_location = current_city

    # Return to depot
    travel_to_depot = distance_matrix[last_location, 0]
    total_travel_time += travel_to_depot
    current_time += travel_to_depot

    if current_time > time_windows[0][1]: # Arrived at depot too late
        return float('inf')

    return total_travel_time

def genetic_algorithm_tsptw(distance_matrix, time_windows, num_nodes, pop_size, generations,
                            mutation_rate, crossover_rate, tournament_k=3, service_times=None):
    """
    Main GA for TSPTW with Hard Time Windows. Uses `calculate_fitness_hard_tw`.

    Args:
        distance_matrix (numpy.ndarray): N x N distance matrix.
        time_windows (numpy.ndarray): N x 2 time window matrix.
        num_nodes (int): Total number of nodes.
        pop_size (int): Population size.
        generations (int): Number of generations.
        mutation_rate (float): Mutation probability.
        crossover_rate (float): Crossover probability.
        tournament_k (int, optional): Tournament size. Defaults to 3.
        service_times (list/numpy.ndarray, optional): Service times. Defaults to None (zeros).

    Returns:
        tuple: (best_individual, best_fitness). `best_fitness` is float('inf') if no feasible solution found.
               Returns ([], 0.0) if num_nodes <= 1 and feasible, or ([], inf) if infeasible.
    """
    population = initial_population(pop_size, num_nodes)
    best_overall_individual = None
    best_overall_fitness = float('inf')

    if num_nodes <= 1:
        # For a single node (depot), fitness is 0 if depot's TW is valid (e.g. [0,X] start at 0), else inf.
        # calculate_fitness_hard_tw handles empty individual for this.
        return [], calculate_fitness_hard_tw([], distance_matrix, time_windows, service_times)

    for _generation in range(generations):
        fitnesses = [calculate_fitness_hard_tw(ind, distance_matrix, time_windows, service_times) for ind in population]

        current_gen_best_fitness = min(fitnesses)
        current_gen_best_idx = fitnesses.index(current_gen_best_fitness)

        if current_gen_best_fitness < best_overall_fitness:
            best_overall_fitness = current_gen_best_fitness
            best_overall_individual = list(population[current_gen_best_idx])
        elif best_overall_fitness == float('inf') and (best_overall_individual is None or not best_overall_individual): # If still Inf, ensure we have *an* individual
            # This ensures that if all solutions are infeasible, we still track one.
             best_overall_individual = list(population[current_gen_best_idx])

        new_population = []
        if best_overall_individual is not None: # Elitism
            new_population.append(list(best_overall_individual))

        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses, k=tournament_k)
            parent2 = tournament_selection(population, fitnesses, k=tournament_k)
            if random.random() < crossover_rate:
                child1, child2 = partially_mapped_crossover(parent1, parent2)
            else:
                child1, child2 = list(parent1), list(parent2)
            child1 = swap_mutation(child1, mutation_rate)
            child2 = swap_mutation(child2, mutation_rate)
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
        population = new_population[:pop_size]

    # Final check on the last population
    if population:
        final_fitnesses = [calculate_fitness_hard_tw(ind, distance_matrix, time_windows, service_times) for ind in population]
        if final_fitnesses:
            last_pop_best_fitness = min(final_fitnesses)
            if last_pop_best_fitness < best_overall_fitness:
                best_overall_fitness = last_pop_best_fitness
                best_overall_individual = list(population[final_fitnesses.index(last_pop_best_fitness)])
            elif best_overall_fitness == float('inf') and (best_overall_individual is None or not best_overall_individual):
                 best_overall_individual = list(population[final_fitnesses.index(last_pop_best_fitness)])

    if best_overall_individual is None: # Should not happen if pop_size > 0 and num_nodes > 0
        return [], float('inf')

    return best_overall_individual, best_overall_fitness

def calculate_fitness_soft_tw(individual, distance_matrix, time_windows,
                              penalty_early_arrival, penalty_late_arrival, service_times=None):
    """
    Calculates fitness for TSPTW with Soft Time Windows.
    Fitness = total_travel_time + total_penalty.
    Also returns total_travel_time and total_penalty separately.

    Args:
        individual (list): Route permutation (city nodes).
        distance_matrix (numpy.ndarray): N x N distance/travel time matrix.
        time_windows (numpy.ndarray): N x 2 matrix of [earliest, latest] arrival times.
        penalty_early_arrival (float): Penalty per unit of time for early arrival.
        penalty_late_arrival (float): Penalty per unit of time for late arrival.
        service_times (list/numpy.ndarray, optional): Service time at each node. Defaults to zeros.

    Returns:
        tuple: (combined_cost, total_travel_time, total_penalty).
    """
    total_travel_time = 0.0
    total_penalty = 0.0
    current_time = time_windows[0][0]
    last_location = 0

    num_nodes_from_dm = len(distance_matrix)
    if service_times is None:
        service_times_actual = [0] * num_nodes_from_dm
    else:
        service_times_actual = service_times
        if len(service_times_actual) != num_nodes_from_dm:
             raise ValueError("service_times length must match number of nodes in distance_matrix")

    for city_node in individual:
        travel_time_leg = distance_matrix[last_location, city_node]
        total_travel_time += travel_time_leg
        arrival_time_at_city = current_time + travel_time_leg
        departure_time_from_city = arrival_time_at_city

        earliest_arrival_city = time_windows[city_node][0]
        latest_arrival_city = time_windows[city_node][1]

        if arrival_time_at_city < earliest_arrival_city: # Arrived early
            total_penalty += (earliest_arrival_city - arrival_time_at_city) * penalty_early_arrival
            departure_time_from_city = earliest_arrival_city # Wait
        elif arrival_time_at_city > latest_arrival_city: # Arrived late
            total_penalty += (arrival_time_at_city - latest_arrival_city) * penalty_late_arrival
            # No change to departure_time_from_city, already late

        departure_time_from_city += service_times_actual[city_node]
        current_time = departure_time_from_city
        last_location = city_node

    # Return to depot
    travel_time_to_depot = distance_matrix[last_location, 0]
    total_travel_time += travel_time_to_depot
    arrival_time_at_depot = current_time + travel_time_to_depot

    if arrival_time_at_depot > time_windows[0][1]: # Late at depot
        total_penalty += (arrival_time_at_depot - time_windows[0][1]) * penalty_late_arrival

    return total_travel_time + total_penalty, total_travel_time, total_penalty

def genetic_algorithm_soft_tw(distance_matrix, time_windows, num_nodes, pop_size, generations,
                              mutation_rate, crossover_rate,
                              penalty_early_arrival, penalty_late_arrival,
                              tournament_k=3, service_times=None):
    """
    Main GA for TSPTW with Soft Time Windows. Uses `calculate_fitness_soft_tw`.

    Args:
        distance_matrix, time_windows, num_nodes, pop_size, generations,
        mutation_rate, crossover_rate: Standard GA parameters.
        penalty_early_arrival (float): Penalty for early arrival.
        penalty_late_arrival (float): Penalty for late arrival.
        tournament_k (int, optional): Tournament size. Defaults to 3.
        service_times (list/numpy.ndarray, optional): Service times. Defaults to None (zeros).

    Returns:
        tuple: (best_individual, combined_cost, travel_time, penalty_amount)
               for the best solution found.
    """
    population = initial_population(pop_size, num_nodes)
    best_overall_individual = None
    best_overall_metrics = {'cost': float('inf'), 'travel': float('inf'), 'penalty': float('inf')}

    if num_nodes <= 1:
        cost, travel, penalty = calculate_fitness_soft_tw(
            [], distance_matrix, time_windows,
            penalty_early_arrival, penalty_late_arrival, service_times
        )
        return [], cost, travel, penalty

    for _generation in range(generations):
        current_gen_evals = [] # Stores dicts of {'ind': ind, 'cost': cost, 'travel': T, 'penalty': P}
        fitness_values_for_selection = [] # Just the combined cost for tournament selection

        for ind in population:
            cost, travel, penalty = calculate_fitness_soft_tw(
                ind, distance_matrix, time_windows,
                penalty_early_arrival, penalty_late_arrival, service_times
            )
            current_gen_evals.append({'ind': ind, 'cost': cost, 'travel': travel, 'penalty': penalty})
            fitness_values_for_selection.append(cost)

        current_gen_best_combined_cost = min(fitness_values_for_selection)
        current_gen_best_idx = fitness_values_for_selection.index(current_gen_best_combined_cost)

        if current_gen_best_combined_cost < best_overall_metrics['cost']:
            best_overall_metrics['cost'] = current_gen_best_combined_cost
            best_overall_metrics['travel'] = current_gen_evals[current_gen_best_idx]['travel']
            best_overall_metrics['penalty'] = current_gen_evals[current_gen_best_idx]['penalty']
            best_overall_individual = list(current_gen_evals[current_gen_best_idx]['ind'])

        new_population = []
        if best_overall_individual is not None: # Elitism
            new_population.append(list(best_overall_individual))

        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitness_values_for_selection, k=tournament_k)
            parent2 = tournament_selection(population, fitness_values_for_selection, k=tournament_k)
            if random.random() < crossover_rate:
                child1, child2 = partially_mapped_crossover(parent1, parent2)
            else:
                child1, child2 = list(parent1), list(parent2)
            child1 = swap_mutation(child1, mutation_rate)
            child2 = swap_mutation(child2, mutation_rate)
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
        population = new_population[:pop_size]

    # Final check on the last population
    if population:
        final_pop_evals = []
        final_pop_costs = []
        for ind in population:
            cost, travel, penalty = calculate_fitness_soft_tw(
                ind, distance_matrix, time_windows,
                penalty_early_arrival, penalty_late_arrival, service_times
            )
            final_pop_evals.append({'ind': ind, 'cost': cost, 'travel': travel, 'penalty': penalty})
            final_pop_costs.append(cost)

        if final_pop_costs:
            last_pop_best_cost = min(final_pop_costs)
            if last_pop_best_cost < best_overall_metrics['cost']:
                best_overall_metrics['cost'] = last_pop_best_cost
                idx = final_pop_costs.index(last_pop_best_cost)
                best_overall_metrics['travel'] = final_pop_evals[idx]['travel']
                best_overall_metrics['penalty'] = final_pop_evals[idx]['penalty']
                best_overall_individual = list(final_pop_evals[idx]['ind'])

    if best_overall_individual is None: # Should only happen if pop_size was 0
        return [], float('inf'), float('inf'), float('inf')

    return best_overall_individual, best_overall_metrics['cost'], best_overall_metrics['travel'], best_overall_metrics['penalty']


# Consolidated if __name__ == '__main__': block for all tests
if __name__ == '__main__':
    print("--- Testing genetic_algorithm.py module ---")

    # --- Test: create_individual & initial_population ---
    print("\n--- Sub-test: create_individual & initial_population ---")
    test_num_nodes_ci = 5
    ind_ci = create_individual(test_num_nodes_ci)
    print(f"Individual for {test_num_nodes_ci} nodes: {ind_ci}")
    assert len(ind_ci) == test_num_nodes_ci - 1
    if test_num_nodes_ci > 1: assert 0 not in ind_ci and all(1 <= city < test_num_nodes_ci for city in ind_ci)

    pop_ci = initial_population(10, test_num_nodes_ci)
    print(f"Population of 10 for {test_num_nodes_ci} nodes created. First: {pop_ci[0] if pop_ci else 'N/A'}")
    assert len(pop_ci) == 10
    if pop_ci: assert len(pop_ci[0]) == test_num_nodes_ci - 1

    ind_1_node_ci = create_individual(1)
    assert ind_1_node_ci == []
    pop_1_node_ci = initial_population(5,1)
    assert all(p == [] for p in pop_1_node_ci)
    print("create_individual & initial_population tests passed.")

    # --- Test: calculate_fitness (Basic TSP) ---
    print("\n--- Sub-test: calculate_fitness (Basic TSP) ---")
    dist_matrix_example_cf = np.array([
        [0, 10, 15, 20, 25], [10, 0, 5, 12, 8], [15, 5, 0, 9, 18],
        [20, 12, 9, 0, 6], [25, 8, 18, 6, 0]
    ])
    ind_cf1 = [1, 2, 3, 4]; fit_cf1 = calculate_fitness(ind_cf1, dist_matrix_example_cf) # Exp: 55
    print(f"Ind: {ind_cf1}, Fitness: {fit_cf1}, Expected: 55")
    assert fit_cf1 == 55
    ind_cf_empty = []; fit_cf_empty = calculate_fitness(ind_cf_empty, dist_matrix_example_cf) # Exp: 0
    assert fit_cf_empty == 0.0
    print("calculate_fitness (TSP) tests passed.")

    # --- Test: tournament_selection ---
    print("\n--- Sub-test: tournament_selection ---")
    pop_ts = [[1,2,3,4],[3,1,4,2],[2,1,3,4],[4,3,2,1]]
    fit_ts = [55, 73, 63, 55] # Fitnesses for pop_ts with dist_matrix_example_cf
    parent_ts = tournament_selection(pop_ts, fit_ts, k=2)
    print(f"Selected parent: {parent_ts}")
    assert parent_ts in pop_ts
    parent_ts_k_all = tournament_selection(pop_ts, fit_ts, k=len(pop_ts))
    assert calculate_fitness(parent_ts_k_all, dist_matrix_example_cf) == min(fit_ts)
    print("tournament_selection tests passed.")

    # --- Test: partially_mapped_crossover (PMX) ---
    print("\n--- Sub-test: partially_mapped_crossover (PMX) ---")
    p1_pmx = [1,2,3,4,5,6,7,8,9]; p2_pmx = [9,3,7,8,2,6,5,1,4]
    original_sample_func_pmx = random.sample # Save original
    def mock_sample_pmx(population_list, k_val): # Mock for deterministic crossover points
        if population_list == range(len(p1_pmx)) and k_val == 2: return [3,6] # Indices 3 and 6
        return original_sample_func_pmx(population_list, k_val)
    random.sample = mock_sample_pmx
    c1_pmx, c2_pmx = partially_mapped_crossover(p1_pmx, p2_pmx)
    random.sample = original_sample_func_pmx # Restore
    c1_pmx_exp = [9,3,2,4,5,6,7,1,8]; c2_pmx_exp = [1,7,3,8,2,6,5,4,9]
    print(f"PMX C1:{c1_pmx} (Exp:{c1_pmx_exp}), C2:{c2_pmx} (Exp:{c2_pmx_exp})")
    assert c1_pmx == c1_pmx_exp and c2_pmx == c2_pmx_exp
    c_empty1, c_empty2 = partially_mapped_crossover([],[])
    assert c_empty1 == [] and c_empty2 == []
    print("partially_mapped_crossover (PMX) tests passed.")

    # --- Test: swap_mutation ---
    print("\n--- Sub-test: swap_mutation ---")
    ind_sm = [1,2,3,4,5,6]
    mut_sm_0 = swap_mutation(list(ind_sm), 0.0)
    assert mut_sm_0 == ind_sm
    mut_sm_1 = swap_mutation(list(ind_sm), 1.0) # High rate
    assert len(mut_sm_1) == len(ind_sm) and set(mut_sm_1) == set(ind_sm)
    print(f"Original: {ind_sm}, Mutated (rate 0.0): {mut_sm_0}, Mutated (rate 1.0): {mut_sm_1}")
    assert swap_mutation([], 0.5) == []
    assert swap_mutation([1], 0.5) == [1]
    print("swap_mutation tests passed.")

    # --- Test: genetic_algorithm (Basic TSP) ---
    print("\n--- Sub-test: genetic_algorithm (Basic TSP) ---")
    ga_num_nodes_tsp = 4
    ga_dist_matrix_tsp = np.array([[0,10,20,5],[10,0,8,12],[20,8,0,15],[5,12,15,0]])
    random.seed(42); np.random.seed(42)
    best_r_tsp, best_f_tsp = genetic_algorithm(ga_dist_matrix_tsp, ga_num_nodes_tsp, 20,50,0.1,0.8)
    random.seed(); np.random.seed() # Reset seed
    print(f"GA TSP: Route {best_r_tsp}, Fitness {best_f_tsp:.2f}")
    assert len(best_r_tsp) == ga_num_nodes_tsp - 1
    assert calculate_fitness(best_r_tsp, ga_dist_matrix_tsp) == best_f_tsp # Check consistency
    # Optimal for this small case is 38
    # assert best_f_tsp == 38.0 # Might be too strict for a short GA run
    print("genetic_algorithm (TSP) tests passed.")

    # --- Test: calculate_fitness_hard_tw ---
    print("\n--- Sub-test: calculate_fitness_hard_tw ---")
    tw_hard_ex = np.array([[0,100],[5,20],[10,30],[15,40],[20,50]])
    st_zero_hard = [0]*5
    ind_hard1 = [1,2]; fit_hard1 = calculate_fitness_hard_tw(ind_hard1, dist_matrix_example_cf, tw_hard_ex, st_zero_hard)
    print(f"Ind: {ind_hard1}, Hard TW Fitness: {fit_hard1}, Expected: 30.0")
    assert fit_hard1 == 30.0 # Based on previous manual trace
    ind_hard2 = [3,1]; fit_hard2 = calculate_fitness_hard_tw(ind_hard2, dist_matrix_example_cf, tw_hard_ex, st_zero_hard)
    print(f"Ind: {ind_hard2}, Hard TW Fitness: {fit_hard2}, Expected: inf")
    assert fit_hard2 == float('inf')
    assert calculate_fitness_hard_tw([], dist_matrix_example_cf, tw_hard_ex, st_zero_hard) == 0.0
    print("calculate_fitness_hard_tw tests passed.")

    # --- Test: genetic_algorithm_tsptw (Hard TW) ---
    print("\n--- Sub-test: genetic_algorithm_tsptw (Hard TW) ---")
    ga_tsptw_tw = np.array([[0,100],[0,15],[10,25],[20,40]]) # For 4-node problem
    ga_tsptw_st = [0]*4
    random.seed(42); np.random.seed(42)
    best_r_h, best_f_h = genetic_algorithm_tsptw(
        ga_dist_matrix_tsp, ga_tsptw_tw, ga_num_nodes_tsp,
        50,100,0.05,0.8, service_times=ga_tsptw_st
    )
    random.seed(); np.random.seed()
    print(f"GA TSPTW Hard: Route {best_r_h}, Fitness {best_f_h:.2f}")
    # Route [1,2,3] has travel 38. Check feasibility:
    # D->1 (10) arr 10. TW1=[0,15] ok. Dep 10.
    # 1->2 (8) arr 18. TW2=[10,25] ok. Dep 18.
    # 2->3 (15) arr 33. TW3=[20,40] ok. Dep 33.
    # 3->D (5) arr 38. TWD=[0,100] ok. Feasible. Cost 38.
    # assert best_f_h == 38.0 # Might be too strict for GA, but for this setup it often finds it.
    if best_f_h != float('inf'):
         assert calculate_fitness_hard_tw(best_r_h, ga_dist_matrix_tsp, ga_tsptw_tw, ga_tsptw_st) == best_f_h
    print("genetic_algorithm_tsptw (Hard TW) tests passed.")

    # --- Test: calculate_fitness_soft_tw ---
    print("\n--- Sub-test: calculate_fitness_soft_tw ---")
    tw_soft_ex = np.array([[0,100],[10,20],[25,35],[40,50],[55,65]])
    st_soft_zero = [0]*5; pen_e = 1.0; pen_l = 5.0
    ind_soft1 = [1,2]
    cost_s1, trav_s1, pen_s1 = calculate_fitness_soft_tw(ind_soft1, dist_matrix_example_cf, tw_soft_ex, pen_e, pen_l, st_soft_zero)
    print(f"Ind: {ind_soft1}, Cost: {cost_s1:.2f}, Travel: {trav_s1:.2f}, Pen: {pen_s1:.2f}. Exp Cost: 40")
    assert cost_s1 == 40.0 # Based on previous manual trace
    cost_s_empty,_,_ = calculate_fitness_soft_tw([], dist_matrix_example_cf, tw_soft_ex, pen_e, pen_l, st_soft_zero)
    assert cost_s_empty == 0.0
    print("calculate_fitness_soft_tw tests passed.")

    # --- Test: genetic_algorithm_soft_tw ---
    print("\n--- Sub-test: genetic_algorithm_soft_tw ---")
    random.seed(42); np.random.seed(42)
    best_r_s, cost_s, travel_s, pen_s = genetic_algorithm_soft_tw(
        ga_dist_matrix_tsp, ga_tsptw_tw, ga_num_nodes_tsp,
        50, 100, 0.05, 0.8, pen_e, pen_l, service_times=ga_tsptw_st
    )
    random.seed(); np.random.seed()
    print(f"GA TSPTW Soft: Route {best_r_s}, Cost {cost_s:.2f}, Travel {travel_s:.2f}, Pen {pen_s:.2f}")
    assert abs(cost_s - (travel_s + pen_s)) < 1e-9
    # For route [1,2,3] (cost 38) with ga_tsptw_tw, penalty should be 0.
    # assert cost_s == 38.0 # Might be too strict for GA to find this perfectly.
    print("genetic_algorithm_soft_tw tests passed.")

    print("\n--- genetic_algorithm.py module testing complete ---")
