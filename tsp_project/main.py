import os
import glob
import time
import random
import numpy as np
import pandas as pd # Ensure pandas is imported for saving results
from src.data_parser import parse_tsptw_file
from src.genetic_algorithm import genetic_algorithm, genetic_algorithm_tsptw, genetic_algorithm_soft_tw
from src.tsp_visualizer import plot_tsp_route

# --- Configuration Section ---
# GA Parameters - can be tuned here
GA_PARAMS = {
    'common': {
        'mutation_rate': 0.02,
        'crossover_rate': 0.80,
        'tournament_k': 3
    },
    'basic_tsp': {
        'pop_size': 50,
        'generations': 100
    },
    'tsptw_hard': {
        'pop_size': 100,
        'generations': 200 # Reduced from 500 in prev. test for quicker full runs
    },
    'tsptw_soft': {
        'pop_size': 100,
        'generations': 200 # Reduced from 500 for quicker full runs
    }
}

# Penalty weights for Soft Time Windows
SOFT_TW_PENALTIES = {
    'penalty_early': 1.0,
    'penalty_late': 5.0
}

# Seed for reproducibility
RANDOM_SEED = 42
# --- End Configuration Section ---

def get_data_files(base_data_path):
    """
    Scans for .txt data files in specified subdirectories of base_data_path.
    Returns a list of full file paths.
    """
    data_files = []
    dataset_types = ["DataSet1-asymmetric", "DataSet2-symmetric"]
    if not os.path.isabs(base_data_path): # Ensure path is absolute
        base_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), base_data_path)

    for ds_type in dataset_types:
        search_path = os.path.join(base_data_path, ds_type, "*.txt")
        found_files = glob.glob(search_path)
        print(f"Searching in: {search_path}, Found: {len(found_files)} files")
        data_files.extend(found_files)
    return data_files

def main():
    """
    Main loop to process all data files with three implemented GAs,
    save results to CSV, and generate route visualizations.
    """
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_dir = os.path.join(current_script_dir, 'data', 'TSP-Data')

    data_files_to_process = get_data_files(base_data_dir)

    if not data_files_to_process:
        print(f"No .txt data files found in subdirectories of {base_data_dir}.")
        example_dir = os.path.join(base_data_dir, "DataSet1-asymmetric")
        example_file_path = os.path.join(example_dir, "example_instance.txt")
        if not os.path.exists(example_file_path):
            print(f"Attempting to create a dummy {example_file_path} for test run as it was not found.")
            try:
                os.makedirs(example_dir, exist_ok=True)
                with open(example_file_path, 'w') as f: # Minimal valid structure
                    f.write("3\n0 10 20\n10 0 5\n20 5 0\n0 100\n0 0\n0 0\n")
                data_files_to_process = [example_file_path]
                print(f"Dummy file created. Processing: {data_files_to_process}")
            except Exception as e:
                print(f"Error creating dummy file: {e}"); return
        else:
             print(f"Warning: get_data_files found no files, but {example_file_path} exists. Using it for processing.")
             data_files_to_process = [example_file_path]
        if not data_files_to_process:
            print("No data files available. Exiting."); return

    all_results = []
    viz_dir = os.path.join(current_script_dir, 'results', 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    # Use common GA parameters
    common_params = GA_PARAMS['common']
    mutation_rate = common_params['mutation_rate']
    crossover_rate = common_params['crossover_rate']
    tournament_k = common_params['tournament_k']

    for file_path in data_files_to_process:
        print(f"\n--- Processing File: {os.path.basename(file_path)} ---")

        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        parsed_data = parse_tsptw_file(file_path)
        if not parsed_data:
            print(f"Could not parse {file_path}. Skipping.")
            all_results.append({'instance_name': os.path.basename(file_path), 'problem_type': 'Unknown',
                                'ga_type': 'N/A', 'error': 'Parsing failed'})
            continue

        instance_name = parsed_data['file_name']
        problem_type = parsed_data['problem_type']
        distance_matrix = parsed_data['distance_matrix']
        time_windows = parsed_data['time_windows']
        num_nodes = parsed_data['num_nodes']

        print(f"Instance: {instance_name}, Type: {problem_type}, Nodes: {num_nodes}")
        if num_nodes <= 0: print("Invalid number of nodes. Skipping GAs."); continue

        # --- Run Basic TSP GA ---
        params_tsp = GA_PARAMS['basic_tsp']
        print("\nRunning Basic TSP GA...")
        start_time = time.time()
        best_route_tsp, best_fitness_tsp = genetic_algorithm(
            distance_matrix, num_nodes, params_tsp['pop_size'], params_tsp['generations'],
            mutation_rate, crossover_rate, tournament_k
        )
        comp_time_tsp = time.time() - start_time
        route_tsp_fmt = [0] + best_route_tsp + [0] if best_route_tsp else ([0] if num_nodes == 1 else [])
        all_results.append({
            'instance_name': instance_name, 'problem_type': problem_type, 'ga_type': 'Basic TSP',
            'route': str(route_tsp_fmt), # Store route as string for CSV
            'total_distance_or_cost': best_fitness_tsp, 'computation_time_sec': comp_time_tsp
        })
        print(f"Basic TSP: Route: {route_tsp_fmt}, Distance: {best_fitness_tsp:.2f}, Time: {comp_time_tsp:.2f}s")
        if route_tsp_fmt:
            plot_filename_tsp = os.path.join(viz_dir, f"{instance_name.replace('.txt','_')}_Basic_TSP.png")
            plot_tsp_route(route_tsp_fmt, num_nodes, instance_name, "Basic TSP", plot_filename_tsp)

        # --- Run TSPTW (Hard Time Windows) GA ---
        params_hard = GA_PARAMS['tsptw_hard']
        print("\nRunning TSPTW (Hard Time Windows) GA...")
        start_time = time.time()
        best_route_hard, best_fitness_hard = genetic_algorithm_tsptw(
            distance_matrix, time_windows, num_nodes, params_hard['pop_size'], params_hard['generations'],
            mutation_rate, crossover_rate, tournament_k, service_times=None
        )
        comp_time_hard = time.time() - start_time
        feasible_hard = best_fitness_hard != float('inf')
        route_hard_fmt = [0] + best_route_hard + [0] if best_route_hard else ([0] if num_nodes == 1 and feasible_hard else [])
        all_results.append({
            'instance_name': instance_name, 'problem_type': problem_type, 'ga_type': 'TSPTW Hard',
            'route': str(route_hard_fmt), 'total_distance_or_cost': best_fitness_hard,
            'is_feasible': feasible_hard, 'computation_time_sec': comp_time_hard
        })
        print(f"TSPTW Hard: Route: {route_hard_fmt}, Cost: {best_fitness_hard:.2f}, Feasible: {feasible_hard}, Time: {comp_time_hard:.2f}s")
        plot_title_hard = "TSPTW Hard" + (" (Infeasible)" if not feasible_hard else "")
        plot_filename_hard = os.path.join(viz_dir, f"{instance_name.replace('.txt','')}_{plot_title_hard.replace(' ','_').replace('(','').replace(')','')}.png")
        plot_tsp_route(route_hard_fmt if feasible_hard else [], num_nodes, instance_name, plot_title_hard, plot_filename_hard)


        # --- Run TSPTW (Soft Time Windows) GA ---
        params_soft = GA_PARAMS['tsptw_soft']
        print("\nRunning TSPTW (Soft Time Windows) GA...")
        start_time = time.time()
        best_route_soft, cost_soft, travel_soft, penalty_soft = genetic_algorithm_soft_tw(
            distance_matrix, time_windows, num_nodes, params_soft['pop_size'], params_soft['generations'],
            mutation_rate, crossover_rate,
            SOFT_TW_PENALTIES['penalty_early'], SOFT_TW_PENALTIES['penalty_late'],
            tournament_k, service_times=None
        )
        comp_time_soft = time.time() - start_time
        route_soft_fmt = [0] + best_route_soft + [0] if best_route_soft else ([0] if num_nodes == 1 else [])
        all_results.append({
            'instance_name': instance_name, 'problem_type': problem_type, 'ga_type': 'TSPTW Soft',
            'route': str(route_soft_fmt), 'total_distance_or_cost': cost_soft,
            'travel_time': travel_soft, 'penalty_time': penalty_soft,
            'computation_time_sec': comp_time_soft
        })
        print(f"TSPTW Soft: Route: {route_soft_fmt}, Cost: {cost_soft:.2f}, Travel: {travel_soft:.2f}, Penalty: {penalty_soft:.2f}, Time: {comp_time_soft:.2f}s")
        if route_soft_fmt:
            plot_filename_soft = os.path.join(viz_dir, f"{instance_name.replace('.txt','_')}_TSPTW_Soft.png")
            plot_tsp_route(route_soft_fmt, num_nodes, instance_name, "TSPTW Soft", plot_filename_soft)

    print("\n\n--- All Collected Results ---")
    if not all_results:
        print("No results were collected.")
    else:
        # Save to CSV
        try:
            results_df = pd.DataFrame(all_results)
            column_order = [
                'instance_name', 'problem_type', 'ga_type', 'route',
                'total_distance_or_cost', 'travel_time', 'penalty_time',
                'is_feasible', 'computation_time_sec', 'error'
            ]
            results_df = results_df.reindex(columns=column_order) # Ensure consistent column order

            results_table_dir = os.path.join(current_script_dir, 'results', 'tables')
            os.makedirs(results_table_dir, exist_ok=True)
            csv_file_path = os.path.join(results_table_dir, 'tsp_results_summary.csv')
            results_df.to_csv(csv_file_path, index=False, float_format='%.4f')
            print(f"\nResults successfully saved to: {csv_file_path}")
        except Exception as e:
            print(f"\nAn error occurred while saving results to CSV: {e}")

        # Print to console (optional, can be removed if CSV is primary output)
        # for i, result in enumerate(all_results):
        #     print(f"\nResult {i+1}:")
        #     for key, value in result.items():
        #         if isinstance(value, float): print(f"  {key}: {value:.4f}")
        #         else: print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
