"""
This module handles the parsing of TSP and TSPTW data files.
It expects a specific format for these files, as described in project documentation.
"""
import numpy as np
import os

def parse_tsptw_file(file_path):
    """
    Parses a TSPTW (Traveling Salesperson Problem with Time Windows) data file.

    The expected file format is:
    - Line 1: Number of nodes (N), including the depot (usually node 0).
    - Lines 2 to N+1: The N x N distance (or travel time) matrix. Each line contains N space-separated values.
    - Lines N+2 to 2N+1: The N x 2 time window matrix. Each line contains two space-separated values
                         (earliest_arrival, latest_arrival) for each node. Node 0's time window
                         represents the depot's operating window.

    Args:
        file_path (str): The full path to the data file.

    Returns:
        dict: A dictionary containing the parsed data:
            - 'num_nodes' (int): The number of nodes.
            - 'distance_matrix' (numpy.ndarray): The N x N distance matrix.
            - 'time_windows' (numpy.ndarray): The N x 2 time window matrix.
            - 'file_name' (str): The base name of the parsed file.
            - 'problem_type' (str): 'asymmetric' or 'symmetric', derived from the file path.
                                    Defaults to 'unknown' if not discernible.
        None: If an error occurs during parsing (e.g., FileNotFoundError, ValueError).
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        lines = [line.strip() for line in lines if line.strip()] # Remove empty lines and strip whitespace

        if not lines:
            print(f"Error: File {file_path} is empty or contains only whitespace.")
            return None

        # First line is the number of nodes
        num_nodes = int(lines[0])

        # Read distance matrix (N x N)
        # The distance matrix starts from the second line and has num_nodes rows
        distance_matrix_data = []
        # Ensure we have enough lines for distance matrix and time windows
        if len(lines) < (1 + num_nodes + num_nodes) :
             print(f"Error: File {file_path} has insufficient lines for the declared number of nodes.")
             return None

        for i in range(1, num_nodes + 1):
            distance_matrix_data.append(list(map(float, lines[i].split())))

        distance_matrix = np.array(distance_matrix_data)
        if distance_matrix.shape != (num_nodes, num_nodes):
            print(f"Error: Distance matrix shape mismatch in {file_path}. Expected ({num_nodes},{num_nodes}), got {distance_matrix.shape}")
            return None

        # Read time windows (N x 2)
        # Time windows start after the distance matrix and have num_nodes rows
        time_windows_data = []
        for i in range(num_nodes + 1, num_nodes * 2 + 1):
            time_windows_data.append(list(map(int, lines[i].split()))) # Usually integers

        time_windows = np.array(time_windows_data)
        if time_windows.shape != (num_nodes, 2):
            print(f"Error: Time windows matrix shape mismatch in {file_path}. Expected ({num_nodes},2), got {time_windows.shape}")
            return None

        # Determine problem type from file_path
        problem_type = 'unknown'
        path_lower = file_path.lower()
        if 'asymmetric' in path_lower:
            problem_type = 'asymmetric'
        elif 'symmetric' in path_lower:
            problem_type = 'symmetric'

        file_name = os.path.basename(file_path)

        return {
            'num_nodes': num_nodes,
            'distance_matrix': distance_matrix,
            'time_windows': time_windows,
            'file_name': file_name,
            'problem_type': problem_type
        }

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except ValueError as ve:
        print(f"ValueError during parsing of {file_path}: {ve}. Check data format (e.g., non-numeric values).")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while parsing {file_path}: {e}")
        return None

if __name__ == '__main__':
    # This section is for direct testing of the parser.
    # It assumes that there's an example file relative to this script's location if run directly.
    print("--- Testing data_parser.py ---")

    # Construct a more robust path to the example file for testing
    # Assuming this script is in tsp_project/src and data is in tsp_project/data
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_base_dir = os.path.dirname(current_script_dir) # Goes up to tsp_project
    example_file_relative_path = os.path.join('data', 'TSP-Data', 'DataSet1-asymmetric', 'example_instance.txt')
    example_file_full_path = os.path.join(project_base_dir, example_file_relative_path)

    print(f"Attempting to parse example file: {example_file_full_path}")

    # Create a dummy example file if it doesn't exist for the test
    if not os.path.exists(example_file_full_path):
        print(f"Example file not found. Creating a dummy file for testing: {example_file_full_path}")
        os.makedirs(os.path.dirname(example_file_full_path), exist_ok=True)
        with open(example_file_full_path, 'w') as f:
            f.write("3\n") # num_nodes = 3 (depot + 2 cities)
            f.write("0 10 20\n")
            f.write("10 0 5\n")
            f.write("20 5 0\n")
            f.write("0 100\n") # Time window for node 0
            f.write("10 20\n")# Time window for node 1
            f.write("15 25\n") # Time window for node 2
        created_dummy = True
    else:
        created_dummy = False

    parsed_data = parse_tsptw_file(example_file_full_path)

    if parsed_data:
        print(f"\nSuccessfully parsed '{parsed_data['file_name']}':")
        print(f"  Number of nodes: {parsed_data['num_nodes']}")
        print(f"  Problem type: {parsed_data['problem_type']}")
        print(f"  Distance Matrix shape: {parsed_data['distance_matrix'].shape}")
        print(f"  Distance Matrix (first 3x3 snippet):\n{parsed_data['distance_matrix'][:3,:3]}")
        print(f"  Time Windows shape: {parsed_data['time_windows'].shape}")
        print(f"  Time Windows (first 3 lines snippet):\n{parsed_data['time_windows'][:3]}")
    else:
        print(f"Failed to parse {example_file_full_path}")

    if created_dummy:
        print(f"\nRemoving dummy test file: {example_file_full_path}")
        os.remove(example_file_full_path)
        # Try to remove directory if empty, not critical if it fails
        try:
            os.rmdir(os.path.dirname(example_file_full_path))
            # os.rmdir(os.path.join(project_base_dir, 'data', 'TSP-Data')) # Careful with this
        except OSError:
            pass

    print("\n--- data_parser.py testing complete ---")
