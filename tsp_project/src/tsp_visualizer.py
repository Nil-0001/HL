"""
This module provides functions for visualizing TSP routes.
It uses Matplotlib to generate plots of routes, highlighting nodes and the depot.
"""
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting, place before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_tsp_route(route, num_nodes, instance_name, algorithm_type, output_filename):
    """
    Plots a TSP route and saves it to a file.

    Nodes are arranged in a circular layout. The depot (node 0) is highlighted.
    If the route is considered invalid or not plottable (e.g., too short for a multi-node problem,
    or representing an infeasible solution), a placeholder image with a message is generated.

    Args:
        route (list): The route sequence of node indices, e.g., [0, 1, 2, 3, 0].
                      It must start and end at the depot (node 0).
        num_nodes (int): Total number of nodes including the depot.
        instance_name (str): Name of the instance (e.g., 'example_instance.txt'). Used in the plot title.
        algorithm_type (str): Type of algorithm (e.g., 'Basic TSP', 'TSPTW Hard'). Used in the plot title.
        output_filename (str): Full path (including directory and .png extension) where the plot image will be saved.
    """

    # Validate route for plotting viability
    # A valid plottable route for num_nodes > 1 should have at least 3 elements e.g. [0, 1, 0]
    # For num_nodes = 1, a valid route is typically [0,0] or just [0] (representing staying at depot)
    is_valid_for_plotting = True
    if num_nodes > 1 and (not route or len(route) < 3):
        is_valid_for_plotting = False
        print(f"Info: Route {route} for {instance_name} - {algorithm_type} is too short for a multi-node plot.")
    elif num_nodes == 1 and route not in ([0], [0,0]): # Only depot
         is_valid_for_plotting = False
         print(f"Info: Invalid route {route} for single depot problem {instance_name} - {algorithm_type}.")
    elif not route and num_nodes > 0 : # Empty route for multi-node problem
        is_valid_for_plotting = False
        print(f"Info: Empty route for {instance_name} - {algorithm_type}.")


    fig, ax = plt.subplots(figsize=(8, 8))

    if not is_valid_for_plotting or num_nodes == 0:
        # Plot a message indicating no valid route or nothing to plot
        message = "No feasible route to plot"
        if num_nodes == 0:
            message = "No nodes to plot (num_nodes=0)"
        elif not route:
             message = "Route is empty"

        ax.text(0.5, 0.5, f"{message}\n{instance_name}\n{algorithm_type}",
                horizontalalignment='center', verticalalignment='center', fontsize=12, color='red', wrap=True)
        ax.set_xticks([])
        ax.set_yticks([])
        plot_title = f"Route for {instance_name} - {algorithm_type} (No Valid Route)"
    else:
        # Generate node positions in a circular layout
        positions = {}
        for i in range(num_nodes):
            angle = 2 * np.pi * i / num_nodes
            # Use slightly smaller radius for aesthetics if many nodes, but 1 is fine for up to ~50 nodes
            radius = 1.0
            positions[i] = (radius * np.cos(angle), radius * np.sin(angle))

        # Plot nodes
        for i in range(num_nodes):
            x, y = positions[i]
            if i == 0: # Depot
                ax.plot(x, y, 's', markersize=12, color='red', label=f'Depot (Node {i})')
            else: # Customer nodes
                ax.plot(x, y, 'o', markersize=8, color='blue', label=f'Node {i}' if i==1 else None) # Label only one customer node type
            ax.text(x * 1.1, y * 1.1, str(i), fontsize=9, ha='center', va='center') # Adjust text position slightly outward

        # Plot edges
        if len(route) > 1: # Ensure there are edges to draw
            for i in range(len(route) - 1):
                node1 = route[i]
                node2 = route[i+1]
                # Check if nodes exist in positions (they should if route is valid)
                if node1 in positions and node2 in positions:
                    x_values = [positions[node1][0], positions[node2][0]]
                    y_values = [positions[node1][1], positions[node2][1]]
                    ax.plot(x_values, y_values, 'gray', linestyle='-', linewidth=1, alpha=0.7)
                    # Optional: Add arrows for asymmetric problems (more complex, requires orientation)
                    # ax.arrow(x_values[0], y_values[0], x_values[1]-x_values[0], y_values[1]-y_values[0],
                    #          head_width=0.05, head_length=0.05, fc='gray', ec='gray', length_includes_head=True)
        plot_title = f"Route for {instance_name} - {algorithm_type}"
        ax.set_xlabel("X-coordinate")
        ax.set_ylabel("Y-coordinate")
        ax.legend(loc='best', fontsize='small')

    ax.set_title(plot_title, wrap=True)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()

    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_filename)
        if not os.path.exists(output_dir) and output_dir: # Check if output_dir is not empty string
            os.makedirs(output_dir, exist_ok=True)
        elif not output_dir and not os.path.exists(output_filename): # Saving in current dir
             pass # No directory needed if saving to current working dir and it exists

        plt.savefig(output_filename)
        print(f"Route plot saved to {output_filename}")
    except Exception as e:
        print(f"Error saving plot '{output_filename}': {e}")

    plt.close(fig) # Close the figure to free memory

if __name__ == '__main__':
    print("--- Testing tsp_visualizer.py ---")

    # Create a dummy output directory for tests
    test_viz_dir = "test_visualizations_output"
    if not os.path.exists(test_viz_dir):
        os.makedirs(test_viz_dir)

    num_nodes_test = 5
    instance_name_test = "TestInstance1"

    # Test 1: Valid route
    route_test1 = [0, 2, 1, 4, 3, 0]
    algo_type_test1 = "TestTSP_Valid"
    output_fn1 = os.path.join(test_viz_dir, f"{instance_name_test}_{algo_type_test1}.png")
    print(f"\nPlotting test case 1 (valid route): {output_fn1}")
    plot_tsp_route(route_test1, num_nodes_test, instance_name_test, algo_type_test1, output_fn1)

    # Test 2: Empty route (should generate placeholder)
    route_test2 = []
    algo_type_test2 = "TestTSP_EmptyRoute"
    output_fn2 = os.path.join(test_viz_dir, f"{instance_name_test}_{algo_type_test2}.png")
    print(f"\nPlotting test case 2 (empty route): {output_fn2}")
    plot_tsp_route(route_test2, num_nodes_test, instance_name_test, algo_type_test2, output_fn2)

    # Test 3: Route too short for multi-node (should generate placeholder)
    route_test3 = [0,0] # This is valid for num_nodes=1, but not for num_nodes=5
    algo_type_test3 = "TestTSP_ShortRouteMultiNode"
    output_fn3 = os.path.join(test_viz_dir, f"{instance_name_test}_{algo_type_test3}.png")
    print(f"\nPlotting test case 3 (short route [0,0] for {num_nodes_test} nodes): {output_fn3}")
    plot_tsp_route(route_test3, num_nodes_test, instance_name_test, algo_type_test3, output_fn3)

    # Test 4: Single node (depot only) - valid route [0,0]
    algo_type_test4 = "TestTSP_DepotOnly_Valid"
    output_fn4 = os.path.join(test_viz_dir, f"{instance_name_test}_{algo_type_test4}.png")
    print(f"\nPlotting test case 4 (depot only, route [0,0]): {output_fn4}")
    plot_tsp_route([0,0], 1, "DepotOnlyInstance", algo_type_test4, output_fn4)

    # Test 5: Single node (depot only) - valid route [0]
    algo_type_test5 = "TestTSP_DepotOnly_Valid_SingleZero"
    output_fn5 = os.path.join(test_viz_dir, f"{instance_name_test}_{algo_type_test5}.png")
    print(f"\nPlotting test case 5 (depot only, route [0]): {output_fn5}")
    plot_tsp_route([0], 1, "DepotOnlyInstance", algo_type_test5, output_fn5)

    # Test 6: Single node (depot only) - invalid route
    algo_type_test6 = "TestTSP_DepotOnly_Invalid"
    output_fn6 = os.path.join(test_viz_dir, f"{instance_name_test}_{algo_type_test6}.png")
    print(f"\nPlotting test case 6 (depot only, route [0,1,0]): {output_fn6}") # Should skip or placeholder
    plot_tsp_route([0,1,0], 1, "DepotOnlyInstance", algo_type_test6, output_fn6) # Will print info message

    # Test 7: num_nodes = 0
    algo_type_test7 = "TestTSP_ZeroNodes"
    output_fn7 = os.path.join(test_viz_dir, f"{instance_name_test}_{algo_type_test7}.png")
    print(f"\nPlotting test case 7 (zero nodes): {output_fn7}")
    plot_tsp_route([], 0, "ZeroNodeInstance", algo_type_test7, output_fn7)


    print(f"\nVisualizer tests complete. Check for PNG files in '{test_viz_dir}'.")
