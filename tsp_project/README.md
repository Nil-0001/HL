# TSP and TSPTW Solver using Genetic Algorithms

## Description

This project implements Genetic Algorithms (GAs) to solve the Traveling Salesperson Problem (TSP) and its variants with Time Windows (TSPTW). It includes implementations for:
1.  **Basic TSP:** Minimizes total travel distance.
2.  **TSPTW with Hard Time Windows:** Minimizes total travel distance while strictly adhering to earliest and latest arrival times at each location. Solutions violating time windows are considered infeasible.
3.  **TSPTW with Soft Time Windows:** Minimizes a combined cost of total travel distance and penalties incurred for arriving earlier or later than the specified time windows.

The project is structured to parse common TSP/TSPTW data file formats, run the selected GAs, and output results including solution routes, costs, and visualizations.

## Data Format

Input data files are expected to be text files (`.txt`) with the following structure:

1.  **Line 1:** A single integer `N` representing the total number of nodes (including the depot, typically node 0).
2.  **Lines 2 to N+1:** An `N x N` distance (or travel time) matrix. Each line should contain `N` space-separated numerical values representing the cost of travel between nodes. `matrix[i][j]` is the cost from node `i` to node `j`.
3.  **Lines N+2 to 2N+1:** An `N x 2` time window matrix. Each line should contain two space-separated numerical values: `earliest_arrival latest_arrival` for each node `i`.
    -   Node 0's time window typically represents the depot's operating window (e.g., earliest departure time from depot, latest arrival time back at depot).

**Example Data Location:**
Sample data files should be placed in subdirectories within `tsp_project/data/TSP-Data/`. The script is configured to look for files in:
-   `tsp_project/data/TSP-Data/DataSet1-asymmetric/`
-   `tsp_project/data/TSP-Data/DataSet2-symmetric/`

An example file (`example_instance.txt`) is provided in `DataSet1-asymmetric`.

## How to Run

1.  **Prerequisites:** Ensure Python 3 is installed.
2.  **Install Dependencies:** Open a terminal and navigate to the `tsp_project` directory (or its parent). Install the required Python libraries:
    ```bash
    pip install numpy pandas matplotlib
    ```
3.  **Execute the Main Script:** Run the main processing script from the directory containing the `tsp_project` folder:
    ```bash
    python tsp_project/main.py
    ```
    Or, if you are already inside the `tsp_project` directory:
    ```bash
    python main.py
    ```

## Output

The script generates the following outputs:

1.  **CSV Results Table:**
    -   A summary of all runs for all processed data files is saved to:
        `tsp_project/results/tables/tsp_results_summary.csv`
    -   Columns include: instance name, problem type, algorithm type, best route found, total distance/cost, travel time (for soft TW), penalty time (for soft TW), feasibility (for hard TW), and computation time.

2.  **Route Visualizations:**
    -   Graphical plots of the best routes found are saved as PNG images in:
        `tsp_project/results/visualizations/`
    -   Filenames are typically in the format: `{instance_name}_{algorithm_type_sanitized}.png`.
    -   For TSPTW with Hard Time Windows, if no feasible solution is found, a placeholder image indicating this will be generated.

3.  **Console Output:**
    -   Progress messages indicating which file and algorithm are currently being processed.
    -   A summary of the results for each algorithm run per instance.
    -   Confirmation of where the CSV results and plot images are saved.

## Implemented Algorithms

The following Genetic Algorithm variants are implemented:

1.  **Basic Genetic Algorithm for TSP:**
    -   Fitness function: Total tour distance.
2.  **Genetic Algorithm for TSPTW (Hard Time Windows):**
    -   Fitness function: Total tour distance.
    -   Infeasible solutions (violating any time window) are assigned an infinitely high fitness value.
3.  **Genetic Algorithm for TSPTW (Soft Time Windows):**
    -   Fitness function: Combined cost of total tour distance and penalties for early or late arrivals at each node (including the depot for late return).
    -   Penalty weights for earliness and lateness can be configured in `main.py`.

All GAs use common components:
-   **Selection:** Tournament Selection
-   **Crossover:** Partially Mapped Crossover (PMX)
-   **Mutation:** Swap Mutation
-   **Elitism:** The best individual from the current generation is carried over to the next.
